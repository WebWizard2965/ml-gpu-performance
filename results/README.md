# What to keep

- `kmeans.csv` — identity versus broadcast, CPU and CUDA, from a Colab GPU run
- `colab-run1-inductor-batch128/` — CIFAR-10 run at batch 128 with the Inductor compile check
- `colab-run2-aot-eager-batch256/` — CIFAR-10 run at batch 256 with the `aot_eager` compile check

Each run folder has `train_report.csv`, `summary.md`, `profiler.txt`, and that run's `kmeans.csv`. After a new Colab run, unpack `ml-pipeline-results.zip` into a new folder named after its settings, and copy README numbers from that folder's `summary.md`.

Do not commit the downloaded zip, `data/`, a virtualenv, or `__pycache__`.

Local commands, from the repository root:

```bash
python3 src/benchmark_kmeans.py \
  --sizes 10000,100000 \
  --clusters 8,16 \
  --dims 32 \
  --iters 10 \
  --repeats 3 \
  --output results/kmeans.csv
python3 src/train_cifar.py --output results/train_report.csv
```
