import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_kmeans import kmeans, make_blobs, pairwise_broadcast, pairwise_identity
from train_cifar import SmallCNN, train_step


class KMeansTest(unittest.TestCase):
    def test_identity_matches_broadcast_on_separated_data(self):
        points_np, init_np = make_blobs(500, 4, 8, seed=1, scale=0.3, separation=12.0)
        points = torch.tensor(points_np)
        init = torch.tensor(init_np)
        labels_a, _, inertia_a = kmeans(points, init, iters=12, method="identity")
        labels_b, _, inertia_b = kmeans(points, init, iters=12, method="broadcast")
        self.assertTrue(torch.allclose(inertia_a, inertia_b, rtol=1e-4, atol=1e-3))
        agreement = (labels_a == labels_b).float().mean().item()
        self.assertGreater(agreement, 0.98)

    def test_distance_formulas_agree(self):
        rng = torch.Generator().manual_seed(0)
        points = torch.randn(40, 6, generator=rng)
        centroids = torch.randn(3, 6, generator=rng)
        broadcast = pairwise_broadcast(points, centroids)
        identity = pairwise_identity(points, centroids)
        self.assertTrue(torch.allclose(broadcast, identity, rtol=1e-4, atol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_cuda_inertia(self):
        points_np, init_np = make_blobs(800, 4, 16, seed=2, scale=0.3, separation=10.0)
        cpu_labels, _, cpu_inertia = kmeans(
            torch.tensor(points_np), torch.tensor(init_np), iters=10, method="identity"
        )
        gpu_labels, _, gpu_inertia = kmeans(
            torch.tensor(points_np, device="cuda"),
            torch.tensor(init_np, device="cuda"),
            iters=10,
            method="identity",
        )
        self.assertTrue(torch.allclose(cpu_inertia.cpu(), gpu_inertia.cpu(), rtol=1e-3, atol=1e-2))
        agreement = (cpu_labels.cpu() == gpu_labels.cpu()).float().mean().item()
        self.assertGreater(agreement, 0.95)


class TrainingStepTest(unittest.TestCase):
    def test_one_step_lowers_loss(self):
        torch.manual_seed(0)
        model = SmallCNN()
        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 10, (8,))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        model.eval()
        with torch.no_grad():
            before = float(torch.nn.functional.cross_entropy(model(images), labels))
        train_step(model, images, labels, optimizer)
        model.eval()
        with torch.no_grad():
            after = float(torch.nn.functional.cross_entropy(model(images), labels))
        self.assertLess(after, before)

    def test_two_forward_passes_match(self):
        torch.manual_seed(1)
        model = SmallCNN().eval()
        images = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            first = model(images)
            second = model(images)
        self.assertTrue(torch.allclose(first, second))


if __name__ == "__main__":
    unittest.main()
