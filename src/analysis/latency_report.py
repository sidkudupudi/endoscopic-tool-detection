"""
Parses the JSON timing export from a high-iteration trtexec run and
reports a full latency percentile breakdown -- tail latency (p99, p99.9)
matters more than mean latency for a real-time medical device claim.

First generate the timing data with more iterations than the default run:
    trtexec --loadEngine=tool_detector.engine --iterations=1000 --avgRuns=100 \
        --exportTimes=latency_timings.json

Then:
    python latency_report.py --file latency_timings.json
"""
import argparse
import json

import numpy as np


def report(file: str):
    with open(file) as f:
        data = json.load(f)

    # trtexec JSON export is a list of per-iteration records with a
    # "latencyMs" (or similar) field -- adjust the key below if your
    # TensorRT version names it differently (check one record first).
    latencies = []
    for record in data:
        for key in ("latencyMs", "latency", "computeMs"):
            if key in record:
                latencies.append(record[key])
                break

    if not latencies:
        print("Couldn't find a latency field automatically. First record was:")
        print(json.dumps(data[0], indent=2))
        return

    arr = np.array(latencies)
    percentiles = [50, 90, 95, 99, 99.9]

    print(f"N={len(arr)} iterations")
    print(f"mean = {arr.mean():.4f} ms")
    print(f"min  = {arr.min():.4f} ms")
    print(f"max  = {arr.max():.4f} ms")
    for p in percentiles:
        print(f"p{p:<5} = {np.percentile(arr, p):.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    report(args.file)
