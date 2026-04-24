#!/usr/bin/env python3
"""Analyze Extending basis timing from qmp log files."""

import re
import pathlib
from datetime import datetime
import statistics

def parse_timestamp(line: str) -> datetime | None:
    """Parse timestamp from log line."""
    match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]', line)
    if match:
        ts_str = match.group(1)
        return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')
    return None

def analyze_log(log_path: pathlib.Path) -> list[float]:
    """Get time intervals between Extending basis messages."""
    timestamps = []
    with open(log_path) as f:
        for line in f:
            if "Extending basis" in line:
                ts = parse_timestamp(line)
                if ts:
                    timestamps.append(ts)

    intervals = []
    for i in range(1, len(timestamps)):
        interval = (timestamps[i] - timestamps[i-1]).total_seconds()
        intervals.append(interval)
    return intervals

def main():
    base = pathlib.Path("/home/hzhangxyz/Cloud/Desktop/qmp/multigpu")
    results = {}

    for gpu_count in [1, 2, 3, 4]:
        log_dir = base / str(gpu_count) / "outputs" / "2026-04-24"
        if not log_dir.exists():
            continue
        # Find the most recent log directory
        log_dirs = sorted(log_dir.iterdir())
        if not log_dirs:
            continue
        latest = log_dirs[-1]
        log_file = latest / "__main__.log"
        if not log_file.exists():
            continue

        intervals = analyze_log(log_file)
        # Skip first few iterations (warm-up/initialization)
        intervals = intervals[5:] if len(intervals) > 5 else intervals
        if intervals:
            median = statistics.median(intervals)
            mean = statistics.mean(intervals)
            min_val = min(intervals)
            max_val = max(intervals)
            results[gpu_count] = {
                "count": len(intervals),
                "median": median,
                "mean": mean,
                "min": min_val,
                "max": max_val,
            }
            print(f"{gpu_count} GPU: count={len(intervals)}, median={median:.2f}s, mean={mean:.2f}s, min={min_val:.2f}s, max={max_val:.2f}s")

    # Speedup analysis
    if 1 in results:
        base_median = results[1]["median"]
        print("\nSpeedup (relative to 1 GPU median):")
        for gpu_count, stats in sorted(results.items()):
            if gpu_count > 1:
                speedup = base_median / stats["median"]
                efficiency = speedup / gpu_count * 100
                print(f"{gpu_count} GPU: speedup={speedup:.2f}x, efficiency={efficiency:.1f}%")

if __name__ == "__main__":
    main()