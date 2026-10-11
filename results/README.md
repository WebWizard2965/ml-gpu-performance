# What to keep

From this repository root:

```bash
python3 src/benchmark_kmeans.py --output results/kmeans.csv
```

Commit `kmeans.csv` when it is the run you want to show. The README timings should be copied from that file, not remembered. A later run on a GPU adds rows with `device=cuda` in the same file. Commit that version and say which machine it came from.

Do not commit a virtualenv or `__pycache__`.
