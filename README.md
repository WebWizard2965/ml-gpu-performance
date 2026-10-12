# Parallel ML benchmark

Two measurements on the same question as the CUDA and C++ labs: did the parallel program do the right thing, and what did it cost?

K-means compares two squared-distance formulas. The broadcast form builds an `(n, k, d)` tensor. The identity form does not. A second command trains a small CIFAR-10 network for one epoch and records accuracy, an eager-versus-compiled logit check, an autocast step on CUDA, and one profiler batch.

One epoch on a subset is a pipeline check. It is not a competitive classifier.

## Colab

Upload `benchmark_kmeans.ipynb` to [Colab](https://colab.research.google.com). Set the runtime to **GPU**, then choose Runtime → Run all. Leave the installed PyTorch alone. The notebook installs `torchvision` only when it is missing.

The last cell downloads `ml-pipeline-results.zip`. Unpack it into `results/` and keep `kmeans.csv`, `train_report.csv`, `profiler.txt`, and `summary.md`. Copy the CIFAR numbers into the blank lines below from `summary.md`. Do not commit `data/`.

## How to run locally

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
  --sizes 10000,100000 \
  --clusters 8,16 \
  --dims 32 \
  --iters 10 \
  --repeats 3 \
  --output results/kmeans.csv
python3 src/train_cifar.py \
  --train-limit 16000 \
  --test-limit 2000 \
  --epochs 1 \
  --batch-size 256 \
  --workers 0 \
  --output results/train_report.csv
```

The unit tests use a synthetic batch and do not download CIFAR-10. `train_cifar.py` downloads CIFAR-10 into `data/` on the first run. `--full` uses the entire train and test splits. CUDA rows, the autocast step, and CUDA profiler time appear only when `torch.cuda.is_available()` is true. Warmup is excluded from the timed epoch, and CUDA timing stops on `torch.cuda.synchronize()`.

## K-means

```
||x - c||^2 = ||x||^2 + ||c||^2 - 2 x c^T
```

`pairwise_broadcast` builds an `(n, k, d)` difference tensor. `pairwise_identity` does not. Both still write an `(n, k)` distance matrix. The CSV column `intermediate_bytes` is that difference, counted from the shapes, not from a profiler guess.

Empty clusters keep their previous centroid. Runs use one seed, one initial centroid tensor, and a fixed iteration count. Inertia is the quality metric. CPU and GPU results are not required to be bitwise equal.

## Measured K-means

Taken from `results/kmeans.csv`, a Colab GPU run: 32 dimensions, 10 iterations, median of 3 repeats. Inertia matched across formulas well inside 1e-4 relative difference. Peak memory is `cuda_peak_bytes`.

n=10,000, k=8:

- identity: CPU 0.030s, CUDA 0.0058s, peak 11.5 MB
- broadcast: CPU 0.236s, CUDA 0.0087s, peak 31.9 MB

n=10,000, k=16:

- identity: CPU 0.022s, CUDA 0.0039s, peak 12.5 MB
- broadcast: CPU 0.252s, CUDA 0.0098s, peak 53.1 MB

n=100,000, k=8:

- identity: CPU 0.148s, CUDA 0.012s, peak 38.5 MB
- broadcast: CPU 1.66s, CUDA 0.042s, peak 234 MB

n=100,000, k=16:

- identity: CPU 0.224s, CUDA 0.015s, peak 48.1 MB
- broadcast: CPU 3.00s, CUDA 0.087s, peak 446 MB

The identity formula is the one that stays small. At n=100,000 and k=16 the broadcast CUDA step is about 6x slower and uses about 9x the memory of the identity step, because it materializes the `(n, k, d)` tensor.

## CIFAR-10 pipeline

`src/train_cifar.py` trains a 3-convolution network: 32, 64, then 128 channels, each with a ReLU, two max-pools, then a linear layer to 10 classes. The default run is the first 16,000 training images and 2,000 test images, one epoch, batch 256, Adam, cross-entropy. It loads data in the main process. `--compile` and `--profile` are optional because those steps can sit for a long time on Colab.

After that epoch the script records:

- test accuracy and loss
- one FP32 step and, on CUDA, one autocast step, both restored to the same weights
- an optional `--compile` logit check and an optional `--profile` batch

Two Colab GPU runs are kept in `results/`. Both used 16,000 training images and one epoch.

`results/colab-run1-inductor-batch128/`, batch 128, `torch.compile` with the default Inductor backend:

- 3,242 images/s, 4.94 s for the timed epoch
- 155 MB peak CUDA memory
- test accuracy 34.5% after one epoch
- eager vs compiled max abs diff: 1.4e-6

`results/colab-run2-aot-eager-batch256/`, batch 256, `torch.compile` with the `aot_eager` backend:

- 3,760 images/s, 4.26 s for the timed epoch
- 269 MB peak CUDA memory
- test accuracy 29.8% after one epoch
- eager vs compiled max abs diff: 0

Each single autocast step was slower than the matching FP32 step, because one step includes autocast setup. Those rows are not a mixed-precision speedup.

## What this does not prove

The K-means table does not show that K-means should run on a GPU. It shows the cost of two formulas. The CIFAR run does not show a trained image model. Two epochs of this network on a subset would still be a weak classifier, and this pipeline stops after one.
