#!/usr/bin/env python3
"""K-means timing for two distance formulas, on CPU and CUDA when present.

The broadcast form materializes an (n, k, d) tensor. The identity form
materializes an (n, k) distance matrix:

    ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x c^T

Both use the same points, the same initial centroids, and a fixed iteration
count. Inertia is compared with a tolerance. Bitwise equality is not required.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def make_blobs(n: int, k: int, d: int, seed: int, scale: float = 0.8, separation: float = 6.0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=separation, size=(k, d)).astype(np.float32)
    owners = rng.integers(0, k, size=n)
    points = centers[owners] + rng.normal(scale=scale, size=(n, d)).astype(np.float32)
    init_index = rng.choice(n, size=k, replace=False)
    return points, points[init_index].copy()


def pairwise_broadcast(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    diff = points[:, None, :] - centroids[None, :, :]
    return (diff * diff).sum(dim=-1)


def pairwise_identity(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    x2 = (points * points).sum(dim=1, keepdim=True)
    c2 = (centroids * centroids).sum(dim=1).unsqueeze(0)
    # A tiny negative value is rounding error, not a real distance.
    return (x2 + c2 - 2.0 * (points @ centroids.T)).clamp_min(0)


def update_centroids(points: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    k = centroids.shape[0]
    one_hot = torch.nn.functional.one_hot(labels, k).to(points.dtype)
    counts = one_hot.sum(dim=0)
    sums = one_hot.T @ points
    empty = counts == 0
    filled = sums / counts.clamp_min(1).unsqueeze(1)
    return torch.where(empty.unsqueeze(1), centroids, filled)


def kmeans(points: torch.Tensor, init: torch.Tensor, iters: int, method: str):
    pairwise = pairwise_identity if method == "identity" else pairwise_broadcast
    centroids = init.clone()
    labels = None
    dists = None
    for _ in range(iters):
        dists = pairwise(points, centroids)
        labels = dists.argmin(dim=1)
        centroids = update_centroids(points, labels, centroids)
    dists = pairwise(points, centroids)
    labels = dists.argmin(dim=1)
    inertia = dists.gather(1, labels.view(-1, 1)).sum()
    return labels, centroids, inertia


def theoretical_bytes(n: int, k: int, d: int, method: str) -> int:
    distance = n * k * 4
    if method == "broadcast":
        return distance + n * k * d * 4
    return distance + (n + k) * d * 4


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_config(points_np, init_np, iters: int, repeats: int, method: str, device: torch.device):
    points = torch.tensor(points_np, device=device)
    init = torch.tensor(init_np, device=device)
    kmeans(points, init, iters, method)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    inertia = None
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        _, _, inertia = kmeans(points, init, iters, method)
        _sync(device)
        times.append(time.perf_counter() - start)
    peak = None
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
    return statistics.median(times), float(inertia.detach().cpu()), peak


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="10000,100000")
    parser.add_argument("--clusters", default="8,16")
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", default="identity,broadcast")
    parser.add_argument("--output", default="results/kmeans.csv")
    args = parser.parse_args()

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    rows = []
    for n in parse_ints(args.sizes):
        for k in parse_ints(args.clusters):
            points, init = make_blobs(n, k, args.dims, args.seed)
            for method in args.methods.split(","):
                if method not in {"identity", "broadcast"}:
                    raise SystemExit(f"unknown method {method}")
                for device in devices:
                    seconds, inertia, peak = run_config(points, init, args.iters, args.repeats, method, device)
                    rows.append({
                        "method": method,
                        "device": device.type,
                        "size": n,
                        "clusters": k,
                        "dims": args.dims,
                        "iters": args.iters,
                        "repeats": args.repeats,
                        "median_seconds": f"{seconds:.6f}",
                        "inertia": f"{inertia:.6f}",
                        "intermediate_bytes": theoretical_bytes(n, k, args.dims, method),
                        "cuda_peak_bytes": "" if peak is None else peak,
                    })
                    print(
                        f"{device.type:4} {method:9} n={n:<7} k={k:<3} "
                        f"median={seconds:.4f}s inertia={inertia:.2f}"
                    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
