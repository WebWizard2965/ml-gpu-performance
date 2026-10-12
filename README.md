# GPU performance in PyTorch

K-means distance formulas, then a one-epoch CIFAR-10 CNN. The notebook is `benchmark_kmeans.ipynb`. Measured outputs are in `results/`.

At 100,000 points, 16 clusters, and 32 dimensions, the algebraic distance was 5.8× faster than the broadcast form on CUDA and used 48 MB instead of 446 MB.

CIFAR-10, batch 128 (`results/colab-run1-inductor-batch128/`): 3,242 images/s, 155 MB peak, torch.compile maximum absolute error 1.4e-6.

CIFAR-10, batch 256 (`results/colab-run2-aot-eager-batch256/`): 3,760 images/s, 269 MB peak, maximum absolute error 0.
