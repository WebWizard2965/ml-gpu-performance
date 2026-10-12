#!/usr/bin/env python3
"""One-epoch CIFAR-10 pipeline for a small CNN.

The default subset is 16,000 training images and 2,000 test images, which is
a pipeline check rather than a competitive accuracy run. Eager FP32 training
is compared with torch.compile logits and, on CUDA, with one autocast step.
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.features(images)
        return self.classifier(hidden.flatten(1))


def train_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    # Keep this on-device. Converting a CUDA tensor to float here would force
    # every training batch to wait for the GPU.
    return loss.detach()


def load_cifar(root: Path, train_limit: int | None, test_limit: int | None):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.CIFAR10(root, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=transform)
    if train_limit is not None:
        train_set = torch.utils.data.Subset(train_set, range(train_limit))
    if test_limit is not None:
        test_set = torch.utils.data.Subset(test_set, range(test_limit))
    return train_set, test_set


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    workers: int,
):
    options = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers:
        options.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })
    return torch.utils.data.DataLoader(dataset, **options)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    loss_sum = torch.zeros((), device=device)
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluation", unit="batch", leave=False):
            images = images.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += F.cross_entropy(logits, labels, reduction="sum")
            correct += (logits.argmax(dim=1) == labels).sum()
            total += labels.size(0)
    sync(device)
    return float(correct) / total, float(loss_sum) / total


def one_batch(loader, device: torch.device):
    images, labels = next(iter(loader))
    return (
        images.to(device, non_blocking=True, memory_format=torch.channels_last),
        labels.to(device, non_blocking=True),
    )


def timed_train_step(model, images, labels, optimizer, device: torch.device, use_amp: bool) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sync(device)
    start = time.perf_counter()
    if use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = F.cross_entropy(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
    sync(device)
    return time.perf_counter() - start


def compile_check(
    model: nn.Module,
    images: torch.Tensor,
    backend: str,
) -> tuple[str, float]:
    clone = copy.deepcopy(model)
    clone.eval()
    with torch.no_grad():
        eager = clone(images)
        try:
            # This is a correctness comparison, not a compiled-performance
            # benchmark. aot_eager exercises Dynamo/AOTAutograd without a long
            # first-run Inductor code-generation step on hosted notebooks.
            compiled = torch.compile(clone, backend=backend)
            compiled_logits = compiled(images)
        except Exception as exc:
            return f"{backend} unavailable: {exc}", -1.0
    diff = float((eager - compiled_logits).abs().max())
    if not torch.allclose(eager, compiled_logits, rtol=1e-3, atol=1e-3):
        return f"{backend} mismatch above rtol=1e-3 atol=1e-3", diff
    return f"{backend} ok", diff


def profile_step(model, images, labels, optimizer, device: torch.device) -> str:
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    sort_by = "self_cpu_time_total"
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        sort_by = "self_cuda_time_total"
    with profile(activities=activities, record_shapes=False) as prof:
        train_step(model, images, labels, optimizer)
        sync(device)
    return prof.key_averages().table(sort_by=sort_by, row_limit=8)


def blank_row() -> dict[str, str]:
    return {
        "stage": "",
        "device": "",
        "images": "",
        "batch_size": "",
        "seconds": "",
        "examples_per_sec": "",
        "loss": "",
        "accuracy": "",
        "peak_bytes": "",
        "max_abs_diff": "",
        "detail": "",
    }


def run_training(args: argparse.Namespace) -> list[dict[str, str]]:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] Loading CIFAR-10 on {device}", flush=True)
    train_limit = None if args.full else args.train_limit
    test_limit = None if args.full else args.test_limit
    train_set, test_set = load_cifar(Path(args.data_dir), train_limit, test_limit)
    train_loader = make_loader(train_set, args.batch_size, True, device, args.workers)
    test_loader = make_loader(test_set, args.batch_size, False, device, args.workers)
    model = SmallCNN().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("[2/4] Warmup batch", flush=True)
    warmup_images, warmup_labels = one_batch(train_loader, device)
    first_loss_tensor = train_step(model, warmup_images, warmup_labels, optimizer)
    sync(device)
    first_loss = float(first_loss_tensor)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    seen = 0
    last_loss = first_loss
    print(f"[3/4] Training one epoch ({len(train_loader)} batches)", flush=True)
    start = time.perf_counter()
    for images, labels in tqdm(train_loader, desc="Training", unit="batch"):
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        labels = labels.to(device, non_blocking=True)
        last_loss = train_step(model, images, labels, optimizer)
        seen += labels.size(0)
    sync(device)
    train_seconds = time.perf_counter() - start
    last_loss = float(last_loss)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else ""

    print(f"[4/4] Evaluating ({len(test_loader)} batches)", flush=True)
    accuracy, eval_loss = evaluate(model, test_loader, device)

    check_status, max_diff = "", -1.0
    if args.compile:
        print(f"Optional compile check, backend={args.compile_backend}", flush=True)
        check_status, max_diff = compile_check(model, warmup_images, args.compile_backend)

    print("Timing one FP32 step and, on CUDA, one autocast step", flush=True)
    snapshot = copy.deepcopy(model.state_dict())
    fp32_seconds = timed_train_step(model, warmup_images, warmup_labels, optimizer, device, False)
    model.load_state_dict(snapshot)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    amp_seconds = ""
    if device.type == "cuda":
        amp_seconds = f"{timed_train_step(model, warmup_images, warmup_labels, optimizer, device, True):.6f}"
    model.load_state_dict(snapshot)

    profile_text = ""
    if args.profile:
        print("Optional profiler batch", flush=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        profile_text = profile_step(model, warmup_images, warmup_labels, optimizer, device)

    rows = []
    train_row = blank_row()
    train_row.update({
        "stage": "train_epoch",
        "device": device.type,
        "images": str(seen),
        "batch_size": str(args.batch_size),
        "seconds": f"{train_seconds:.6f}",
        "examples_per_sec": f"{seen / train_seconds:.2f}" if train_seconds else "",
        "loss": f"{last_loss:.6f}",
        "peak_bytes": "" if peak == "" else str(peak),
        "detail": f"first_loss={first_loss:.6f}; epochs={args.epochs}",
    })
    rows.append(train_row)

    eval_row = blank_row()
    eval_row.update({
        "stage": "eval",
        "device": device.type,
        "images": str(len(test_loader.dataset)),
        "batch_size": str(args.batch_size),
        "loss": f"{eval_loss:.6f}",
        "accuracy": f"{accuracy:.6f}",
        "detail": "one epoch is a pipeline check, not a competitive CIFAR-10 score",
    })
    rows.append(eval_row)

    if args.compile:
        compile_row = blank_row()
        compile_row.update({
            "stage": "compile_check",
            "device": device.type,
            "batch_size": str(args.batch_size),
            "max_abs_diff": "" if max_diff < 0 else f"{max_diff:.8f}",
            "detail": check_status,
        })
        rows.append(compile_row)

    fp32_row = blank_row()
    fp32_row.update({
        "stage": "fp32_step",
        "device": device.type,
        "batch_size": str(warmup_images.size(0)),
        "seconds": f"{fp32_seconds:.6f}",
    })
    rows.append(fp32_row)

    if amp_seconds:
        amp_row = blank_row()
        amp_row.update({
            "stage": "amp_step",
            "device": device.type,
            "batch_size": str(warmup_images.size(0)),
            "seconds": amp_seconds,
        })
        rows.append(amp_row)

    if profile_text:
        results = Path(args.output).parent
        results.mkdir(parents=True, exist_ok=True)
        profile_path = results / "profiler.txt"
        profile_path.write_text(profile_text + "\n", encoding="utf-8")
        profile_row = blank_row()
        profile_row.update({
            "stage": "profile",
            "device": device.type,
            "detail": str(profile_path),
        })
        rows.append(profile_row)
    return rows


def write_outputs(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    by_stage = {row["stage"]: row for row in rows}
    train = by_stage["train_epoch"]
    evaluation = by_stage["eval"]
    lines = [
        "# Training summary",
        "",
        f"- device: {train['device']}",
        f"- timed training images: {train['images']}",
        f"- train seconds: {train['seconds']}",
        f"- examples/sec: {train['examples_per_sec']}",
        f"- last training loss: {train['loss']} ({train['detail']})",
        f"- test accuracy: {evaluation['accuracy']}",
        f"- test loss: {evaluation['loss']}",
        "",
        "One epoch on this subset is a pipeline check. It is not a competitive CIFAR-10 result.",
        "",
    ]
    if "compile_check" in by_stage:
        compiled = by_stage["compile_check"]
        lines.insert(
            -3,
            f"- eager vs compiled: {compiled['detail']}, max abs diff {compiled['max_abs_diff'] or 'n/a'}",
        )
    if "amp_step" in by_stage:
        lines.insert(-3, f"- autocast step seconds: {by_stage['amp_step']['seconds']}")
        lines.insert(-3, f"- fp32 step seconds: {by_stage['fp32_step']['seconds']}")
    (output.parent / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=16000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="Also compare one batch with torch.compile")
    parser.add_argument("--profile", action="store_true", help="Also profile one batch")
    parser.add_argument(
        "--compile-backend",
        default="aot_eager",
        choices=("aot_eager", "inductor"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="results/train_report.csv")
    parser.add_argument("--full", action="store_true", help="Use the full CIFAR-10 train and test splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs != 1:
        raise SystemExit("This pipeline runs one epoch. Leave --epochs at 1.")
    write_outputs(run_training(args), Path(args.output))


if __name__ == "__main__":
    main()
