"""Benchmark: send simultaneous requests to Triton and FastAPI, compare latencies."""

import argparse
import concurrent.futures
import statistics
import time
from pathlib import Path

import requests


def send_triton_request(url: str) -> float:
    """Send one request to Triton pipeline, return latency in ms."""
    payload = {
        "inputs": [
            {
                "name": "DUMMY",
                "shape": [1],
                "datatype": "FP32",
                "data": [0.0],
            }
        ],
        "outputs": [
            {"name": "DETECTIONS_YOLO1"},
            {"name": "DETECTIONS_YOLO2"},
        ],
    }
    start = time.perf_counter()
    resp = requests.post(f"{url}/v2/models/pipeline/infer", json=payload)
    elapsed = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    return elapsed


def send_fastapi_request(url: str, image_path: str) -> float:
    """Send one request to FastAPI /predict, return latency in ms."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    start = time.perf_counter()
    resp = requests.post(
        f"{url}/predict",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
    )
    elapsed = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    return elapsed


def run_benchmark(target: str, url: str, n: int, workers: int, image_path: str | None) -> list[float]:
    """Fire n requests with given concurrency, return list of latencies (ms)."""
    if target == "triton":
        fn = lambda: send_triton_request(url)
    else:
        if not image_path:
            raise ValueError("--image is required for fastapi target")
        fn = lambda: send_fastapi_request(url, image_path)

    latencies = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn) for _ in range(n)]
        for f in concurrent.futures.as_completed(futures):
            try:
                latencies.append(f.result())
            except Exception as e:
                print(f"  ERROR: {e}")
    return latencies


def print_stats(name: str, latencies: list[float]) -> None:
    if not latencies:
        print(f"  {name}: no successful requests")
        return
    latencies.sort()
    print(f"  {name}:")
    print(f"    requests : {len(latencies)}")
    print(f"    min      : {latencies[0]:.1f} ms")
    print(f"    max      : {latencies[-1]:.1f} ms")
    print(f"    mean     : {statistics.mean(latencies):.1f} ms")
    print(f"    median   : {statistics.median(latencies):.1f} ms")
    print(f"    p95      : {latencies[int(len(latencies) * 0.95)]:.1f} ms")
    print(f"    total    : {sum(latencies):.1f} ms")
    wall_clock = latencies[-1]  # approx — last to finish
    print(f"    throughput: {len(latencies) / (sum(latencies) / 1000 / len(latencies) * 1):.1f} req/s (estimated)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark simultaneous requests to Triton vs FastAPI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--target", choices=["triton", "fastapi", "both"], default="both",
                        help="Which server to benchmark")
    parser.add_argument("--triton-url", default="http://localhost:8000")
    parser.add_argument("--fastapi-url", default="http://localhost:9000")
    parser.add_argument("--image", default=None,
                        help="Image file for FastAPI requests (required for fastapi/both)")
    parser.add_argument("-n", type=int, default=20, help="Total number of requests (default: 20)")
    parser.add_argument("-w", "--workers", type=int, default=10,
                        help="Concurrent workers / threads (default: 10)")

    args = parser.parse_args()

    if args.target in ("fastapi", "both") and not args.image:
        # Try default sample image
        for candidate in ["sample_data/test.jpg", "triton/model_repository/test.jpg"]:
            if Path(candidate).exists():
                args.image = candidate
                break
        if not args.image:
            parser.error("--image is required for fastapi target (no default found)")

    print(f"\n{'='*60}")
    print(f"  Benchmark: {args.n} requests, {args.workers} concurrent workers")
    print(f"{'='*60}\n")

    if args.target in ("triton", "both"):
        print(f"--- Triton ({args.triton_url}) ---")

        # Warmup
        print("  warming up (2 requests)...")
        run_benchmark("triton", args.triton_url, 2, 1, None)

        # Sequential baseline
        print(f"  sequential ({args.n} requests, 1 worker)...")
        wall_start = time.perf_counter()
        seq = run_benchmark("triton", args.triton_url, args.n, 1, None)
        seq_wall = (time.perf_counter() - wall_start) * 1000
        print_stats("sequential", seq)
        print(f"    wall time: {seq_wall:.1f} ms")

        # Concurrent
        print(f"  concurrent ({args.n} requests, {args.workers} workers)...")
        wall_start = time.perf_counter()
        par = run_benchmark("triton", args.triton_url, args.n, args.workers, None)
        par_wall = (time.perf_counter() - wall_start) * 1000
        print_stats("concurrent", par)
        print(f"    wall time: {par_wall:.1f} ms")

        if seq_wall > 0:
            print(f"    speedup  : {seq_wall / par_wall:.2f}x\n")

    if args.target in ("fastapi", "both"):
        print(f"--- FastAPI ({args.fastapi_url}) ---")

        # Warmup
        print("  warming up (2 requests)...")
        run_benchmark("fastapi", args.fastapi_url, 2, 1, args.image)

        # Sequential baseline
        print(f"  sequential ({args.n} requests, 1 worker)...")
        wall_start = time.perf_counter()
        seq = run_benchmark("fastapi", args.fastapi_url, args.n, 1, args.image)
        seq_wall = (time.perf_counter() - wall_start) * 1000
        print_stats("sequential", seq)
        print(f"    wall time: {seq_wall:.1f} ms")

        # Concurrent
        print(f"  concurrent ({args.n} requests, {args.workers} workers)...")
        wall_start = time.perf_counter()
        par = run_benchmark("fastapi", args.fastapi_url, args.n, args.workers, args.image)
        par_wall = (time.perf_counter() - wall_start) * 1000
        print_stats("concurrent", par)
        print(f"    wall time: {par_wall:.1f} ms")

        if seq_wall > 0:
            print(f"    speedup  : {seq_wall / par_wall:.2f}x\n")


if __name__ == "__main__":
    main()
