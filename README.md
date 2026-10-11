# Parallel K-means benchmark

Same K-means, two ways of computing squared distance. The broadcast form builds a large `(n, k, d)` tensor. The identity form does not. The script times both on CPU, and on CUDA when PyTorch can see a GPU. Inertia is the quality check. Bitwise equality is not required.

This is the measurement companion to `cuda-kernel-verification` and `cpp-concurrency-lab`. It is not a new model project. The point is to show a fair CPU versus GPU comparison, including a case where the GPU is slower.

## How to run

Use a virtualenv. On the machine where this was built:

```bash
python3 -m venv ~/.venvs/nvidia-portfolio
~/.venvs/nvidia-portfolio/bin/pip install -r requirements.txt
```

The PyTorch CPU wheel, if you have no NVIDIA GPU:

```bash
~/.venvs/nvidia-portfolio/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then, from this directory:

```bash
python3 tests/test_correctness.py
python3 src/benchmark_kmeans.py \
  --sizes 10000,50000 \
  --clusters 8,16 \
  --dims 32 \
  --iters 10 \
  --repeats 3 \
  --output results/kmeans.csv
```

Drop `--sizes` and the other flags to use the script defaults (`10000,100000`, clusters `8,16`, 32 dims, 5 repeats). CUDA rows appear only when `torch.cuda.is_available()` is true. Warmup and `torch.cuda.synchronize()` wrap the timed region.

## How to keep results

See `results/README.md`. Commit `results/kmeans.csv`. Copy the medians into the "Measured" section from that file. Say which machine and whether CUDA was available. Do not commit the virtualenv.

```
||x - c||^2 = ||x||^2 + ||c||^2 - 2 x c^T
```

`pairwise_broadcast` builds an `(n, k, d)` difference tensor. `pairwise_identity` does not. Both still write an `(n, k)` distance matrix. The CSV column `intermediate_bytes` is that difference, counted from the shapes, not from a profiler guess.

Empty clusters keep their previous centroid. Runs use one seed, one initial centroid tensor, and a fixed iteration count. Inertia is the quality metric. CPU and GPU results are not required to be bitwise equal.

Timing warms up once, then measures `repeats` calls. On CUDA the clock starts and stops around `torch.cuda.synchronize()`.

## Commands

```bash
python3 tests/test_correctness.py
python3 src/benchmark_kmeans.py \
  --sizes 10000,100000 \
  --clusters 8,16 \
  --dims 32 \
  --repeats 5
```

`results/kmeans.csv` has method, device, size, median seconds, inertia, and the byte estimate. A slower GPU on a small `n` is a valid row. Kernel launch and the Python centroid update are part of the explanation, not something to hide.

## Measured on this CPU

One WSL run, PyTorch CPU, 32 dimensions, 10 iterations, median of 3 repeats. CUDA was skipped because `torch.cuda.is_available()` was false. Inertia matched across formulas well inside 1e-5 relative difference.

- n=10,000, k=8: identity 0.047s, broadcast 0.281s
- n=10,000, k=16: identity 0.064s, broadcast 0.791s
- n=50,000, k=8: identity 0.775s, broadcast 1.962s
- n=50,000, k=16: identity 0.543s, broadcast 2.914s

The stable result is that the broadcast form was slower in every configuration. Do not promote a single-run ordering between k=8 and k=16 into a general claim.

## What this does not prove

This does not show that K-means "should" run on a GPU. It shows the cost of a particular formula and a particular launch pattern on the device you actually have.
