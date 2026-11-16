"""Compare automatic sparse FSP with dense FSP, including exact round trips.

Reads are warm-cache CPU/file API timings, not cold-disk throughput. Inputs
are synthetic unless --paths-json supplies a list of paths or {path, group}.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np

import fspcodec


def median_ms(function, repeats):
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(1000 * (time.perf_counter() - start))
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    sources = []
    if args.paths_json:
        for item in json.loads(args.paths_json.read_text()):
            item = {"path": item, "group": "inputs"} if isinstance(item, str) else item
            sources.append((str(item["path"]), item.get("group", "inputs"),
                            fspcodec.read(item["path"])))
    else:
        rng = np.random.default_rng(20261116)
        for shape in ((768, 1024), (375, 1242)):
            for count in (0, 500, 1000, 1500):
                array = np.zeros(shape, np.float32)
                array.flat[rng.choice(array.size, count, replace=False)] = rng.uniform(0.1, 60, count)
                sources.append((f"synthetic_{shape}_{count}", str(shape), array))
    records = []
    with tempfile.TemporaryDirectory(prefix="fsp-sparse-benchmark-") as directory:
        for number, (source, group, array) in enumerate(sources):
            blobs = {kind: fspcodec.compress(array, storage=kind) for kind in ("dense", "auto")}
            paths = {kind: Path(directory) / f"{number}_{kind}.fsp" for kind in blobs}
            formats = {}
            for kind, blob in blobs.items():
                paths[kind].write_bytes(blob)
                decoded = fspcodec.read(paths[kind])
                if decoded.shape != array.shape or not np.array_equal(decoded.view(np.uint32),
                                                                      array.view(np.uint32)):
                    raise AssertionError("Float32 bits changed")
                with fspcodec.open(paths[kind]) as stream:
                    streamed = list(stream)
                if len(streamed) != 1 or not np.array_equal(streamed[0].view(np.uint32),
                                                          array.view(np.uint32)):
                    raise AssertionError("File-stream round trip changed")
                formats[kind] = {
                    "bytes": len(blob), "version": blob[4], "bitwise_equal": True,
                    "encode_ms": median_ms(lambda: fspcodec.compress(array, storage=kind), args.repeats),
                    "read_ms": median_ms(lambda: fspcodec.read(paths[kind]), args.repeats),
                }
            records.append({"source": source, "group": group, "shape": list(array.shape),
                            "explicit_values": int(np.count_nonzero(array.view(np.uint32))),
                            "input_bits_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                            "original_file_bytes": Path(source).stat().st_size if Path(source).is_file() else None,
                            "formats": formats})
    summary = {}
    for group in sorted({r["group"] for r in records}):
        rows = [r for r in records if r["group"] == group]
        dense_bytes = sum(r["formats"]["dense"]["bytes"] for r in rows)
        auto_bytes = sum(r["formats"]["auto"]["bytes"] for r in rows)
        summary[group] = {
            "inputs": len(rows), "size_reduction_percent": 100 * (1 - auto_bytes / dense_bytes),
            "formats": {kind: {key: statistics.mean(r["formats"][kind][key] for r in rows)
                               for key in ("bytes", "encode_ms", "read_ms")}
                        for kind in ("dense", "auto")},
        }
    report = {"fspcodec_version": fspcodec.__version__, "module": fspcodec.__file__,
              "timing_scope": "warm-cache median read including reconstruction, not cold-disk throughput",
              "repeats": args.repeats, "all_bitwise_equal": True, "records": records, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
