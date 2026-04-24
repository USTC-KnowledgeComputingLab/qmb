#!/usr/bin/env python3
"""
Analyze lanczos timing from qmp log files.

Usage: python analyze_lanczos_timing.py <log_file_or_directory>
"""

import re
import sys
import pathlib
from datetime import datetime

def parse_timestamp(line: str) -> datetime | None:
    """Parse timestamp from log line."""
    match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]', line)
    if match:
        ts_str = match.group(1)
        return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')
    return None

def analyze_log(log_path: pathlib.Path) -> dict:
    """Analyze a single log file for lanczos timing."""

    with open(log_path) as f:
        lines = f.readlines()

    cycles = []
    current_cycle = None
    lanczos_iterations = []

    for line in lines:
        ts = parse_timestamp(line)
        if ts is None:
            continue

        # Start of a new optimization cycle
        if "Starting a new optimization cycle" in line:
            current_cycle = {"start": ts, "lanczos_start": None, "lanczos_end": None, "energy_count": 0}
            lanczos_iterations = []

        # Start of lanczos computation
        if "Computing the target for local optimization" in line and current_cycle:
            current_cycle["lanczos_start"] = ts

        # Each lanczos iteration (energy report)
        if "The current energy is" in line and current_cycle:
            lanczos_iterations.append(ts)
            current_cycle["energy_count"] += 1

        # End of lanczos computation (before local optimization)
        if "Local optimization target calculated" in line and current_cycle:
            current_cycle["lanczos_end"] = ts
            current_cycle["lanczos_iterations"] = lanczos_iterations
            cycles.append(current_cycle)
            current_cycle = None

    return {"log_path": log_path, "cycles": cycles}

def compute_stats(result: dict) -> dict:
    """Compute statistics from analysis result."""
    cycles = result["cycles"]

    lanczos_times = []
    iteration_times = []

    for cycle in cycles:
        if cycle["lanczos_start"] and cycle["lanczos_end"]:
            lanczos_time = (cycle["lanczos_end"] - cycle["lanczos_start"]).total_seconds()
            lanczos_times.append(lanczos_time)

        # Compute time between consecutive energy reports
        iterations = cycle.get("lanczos_iterations", [])
        for i in range(1, len(iterations)):
            iter_time = (iterations[i] - iterations[i-1]).total_seconds()
            iteration_times.append(iter_time)

    stats = {
        "log_path": result["log_path"],
        "num_cycles": len(cycles),
        "total_energy_reports": sum(c["energy_count"] for c in cycles),
    }

    if lanczos_times:
        stats["avg_lanczos_time"] = sum(lanczos_times) / len(lanczos_times)
        stats["min_lanczos_time"] = min(lanczos_times)
        stats["max_lanczos_time"] = max(lanczos_times)
        stats["total_lanczos_time"] = sum(lanczos_times)

    if iteration_times:
        stats["avg_iteration_time"] = sum(iteration_times) / len(iteration_times)
        stats["min_iteration_time"] = min(iteration_times)
        stats["max_iteration_time"] = max(iteration_times)

    return stats

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_lanczos_timing.py <log_file_or_directory>")
        sys.exit(1)

    path = pathlib.Path(sys.argv[1])

    if path.is_file():
        log_files = [path]
    elif path.is_dir():
        log_files = list(path.rglob("__main__.log"))
    else:
        print(f"Path not found: {path}")
        sys.exit(1)

    print("=" * 80)
    print("Lanczos Timing Analysis")
    print("=" * 80)

    all_stats = []
    for log_file in sorted(log_files):
        result = analyze_log(log_file)
        stats = compute_stats(result)
        all_stats.append(stats)

    # Print summary table
    print(f"\n{'GPU Count':<12} {'Log File':<40} {'Avg Lanczos (s)':<15} {'Avg Iteration (ms)':<18} {'Total Reports':<12}")
    print("-" * 100)

    for stats in all_stats:
        # Extract GPU count from path (multigpu/1, multigpu/2, etc.)
        gpu_count = stats["log_path"].parent.parent.parent.parent.name

        avg_lanczos = stats.get("avg_lanczos_time", 0)
        avg_iter_ms = stats.get("avg_iteration_time", 0) * 1000  # Convert to ms
        total_reports = stats.get("total_energy_reports", 0)

        log_name = stats["log_path"].parent.parent.name  # timestamp folder

        print(f"{gpu_count:<12} {log_name:<40} {avg_lanczos:<15.3f} {avg_iter_ms:<18.2f} {total_reports:<12}")

    # Compute speedup
    if len(all_stats) >= 2:
        print("\n" + "=" * 80)
        print("Speedup Analysis (relative to 1 GPU)")
        print("=" * 80)

        base_time = all_stats[0].get("avg_iteration_time", 0) * 1000

        for stats in all_stats:
            gpu_count = stats["log_path"].parent.parent.parent.parent.name
            avg_iter_ms = stats.get("avg_iteration_time", 0) * 1000

            if base_time > 0 and avg_iter_ms > 0:
                speedup = base_time / avg_iter_ms
                efficiency = speedup / int(gpu_count) * 100
                print(f"{gpu_count} GPU: iteration time = {avg_iter_ms:.2f} ms, speedup = {speedup:.2f}x, efficiency = {efficiency:.1f}%")

if __name__ == "__main__":
    main()