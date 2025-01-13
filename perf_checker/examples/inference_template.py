"""Template script for a typical ML inference pipeline."""

import random
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from device_setup import device
from perf_checker import (
    CollectorManager,
    monitor,
    set_default_manager,
    setup_base_logger,
)

DEVICE = device


def initialize_monitoring(
    enable_time: bool = True,
    enable_precision: bool = True,
    precision_config: str = "configs/precision_config.json",
) -> CollectorManager:
    """Initialize the monitoring system.

    This must be called before using any @monitor decorators.

    Args:
        enable_time: Whether to enable time monitoring
        enable_precision: Whether to enable precision monitoring
        precision_config: Path to precision monitoring config file

    Returns:
        The initialized CollectorManager instance
    """
    manager = CollectorManager(
        enable_time=enable_time,
        enable_precision=enable_precision,
        precision_config=precision_config,
    )
    set_default_manager(manager)
    return manager


# Initialize monitoring - this must be done before any @monitor decorators are used
manager = initialize_monitoring(
    enable_time=True,
    enable_precision=True,
    precision_config="configs/precision_config.json",
)


class DummyDataset(Dataset):
    """A simple dataset that generates random tensors."""

    def __init__(self, num_samples: int = 100, image_size: int = 32):
        """Initialize the dataset.

        Args:
            num_samples: Number of samples in the dataset
            image_size: Size of the square images
        """
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Generate a random image tensor.

        Args:
            idx: Index of the sample

        Returns:
            Random tensor of shape (3, image_size, image_size)
        """
        return torch.randn(3, self.image_size, self.image_size, device=DEVICE)


class SimpleNet(nn.Module):
    """A very simple CNN for testing precision monitoring."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleNetV2(nn.Module):
    """A slightly different CNN architecture for testing."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@monitor("preprocessing", "resize", enable_time=True, enable_precision=True)
def resize_image(data: torch.Tensor) -> torch.Tensor:
    """Resize images to target size."""
    return F.interpolate(data, size=(32, 32), mode="bilinear", align_corners=False)


@monitor("preprocessing", "augment", enable_time=True, enable_precision=False)
def augment_image(data: torch.Tensor) -> torch.Tensor:
    """Apply simple data augmentation."""
    if random.random() > 0.5:
        data = torch.flip(data, dims=[3])
    return data


@monitor("postprocessing", "decode", enable_time=True, enable_precision=False)
def decode_prediction(logits: torch.Tensor) -> List[Dict]:
    """Decode model predictions."""
    classes = ["cat", "dog", "bird"]
    probs = F.softmax(logits, dim=1).cpu()

    results = []
    for i in range(logits.size(0)):
        sample_probs = probs[i].tolist()
        results.append(
            {
                "id": i,
                "predictions": [
                    {"class": cls, "score": score}
                    for cls, score in zip(classes, sample_probs)
                ],
            }
        )
    return results


@monitor("postprocessing", "fuse", enable_time=True, enable_precision=False)
def fuse_predictions(
    pred_1: List[Dict], pred_2: List[Dict], weight_1: float = 0.6
) -> List[Dict]:
    """Fuse predictions from multiple models."""
    weight_2 = 1.0 - weight_1
    results = []

    for p1, p2 in zip(pred_1, pred_2):
        fused = {"id": p1["id"], "fused_predictions": []}

        for pred1, pred2 in zip(p1["predictions"], p2["predictions"]):
            assert pred1["class"] == pred2["class"]
            fused["fused_predictions"].append(
                {
                    "class": pred1["class"],
                    "score": pred1["score"] * weight_1 + pred2["score"] * weight_2,
                }
            )
        results.append(fused)
    return results


def process_batch(
    batch: torch.Tensor,
    model_1: nn.Module,
    model_2: nn.Module,
    manager: CollectorManager,
) -> List[Dict]:
    """Process a single batch of data.

    Args:
        batch: Input tensor batch
        model_1: First model
        model_2: Second model
        manager: Monitor manager

    Returns:
        List of fused predictions
    """
    # Pre-processing (using decorators)
    data = batch.to(DEVICE)
    data = resize_image(data)
    data = augment_image(data)

    # Model inference (using context manager)
    with torch.no_grad():
        # Model 1 inference
        with manager.monitor_context("inference", "model1.forward"):
            logits_1 = model_1(data)

        with manager.monitor_context("inference", "model1.decode"):
            pred_1 = decode_prediction(logits_1)

        # Model 2 inference
        with manager.monitor_context("inference", "model2.forward"):
            logits_2 = model_2(data)

        with manager.monitor_context("inference", "model2.decode"):
            pred_2 = decode_prediction(logits_2)

    # Post-processing (using decorators)
    return fuse_predictions(pred_1, pred_2)


def run_inference(num_samples: int = 4, batch_size: int = 1) -> List[Dict]:
    """Run the inference pipeline, treating each batch as a separate run.

    Args:
        num_samples: Total number of samples to process
        batch_size: Batch size for processing

    Returns:
        List of results for each batch
    """
    # Use the global manager instance
    global manager

    # Initialize models and move to device
    model_1 = SimpleNet().to(DEVICE)
    model_2 = SimpleNetV2().to(DEVICE)

    # Set models to eval mode
    model_1.eval()
    model_2.eval()

    # Create dataset and data loader
    dataset = DummyDataset(num_samples=num_samples)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )  # No shuffle for deterministic batch order

    all_results = []

    # Process each batch as a separate run
    for batch_idx, batch in enumerate(data_loader):
        # Set the batch index as the run ID for monitoring
        manager.set_run(batch_idx)

        print(f"Processing batch {batch_idx + 1}/{len(data_loader)}", end="\r")
        batch_results = process_batch(
            batch=batch, model_1=model_1, model_2=model_2, manager=manager
        )
        all_results.extend(batch_results)

    print("\nCompleted processing all batches")

    # Print sample predictions from first batch
    print("\nSample predictions from first batch:")
    for pred in all_results[:batch_size]:
        for fused_pred in pred["fused_predictions"]:
            print(f"  {fused_pred['class']}: {fused_pred['score']:.3f}")

    # Export monitoring results
    csv_path = manager.export_time_stats("results/time")
    print(f"\nExported timing statistics to: {csv_path}")

    # Print timing statistics
    time_stats = manager.get_time_stats()
    print("\nTiming Statistics:")
    for stage in time_stats:
        print(f"\nStage: {stage}")
        for tag, stats in time_stats[stage].items():
            print(f"  {tag}:")
            for metric, value in stats.items():
                if metric == "times_by_run":
                    print(f"    {metric}: {value}")
                elif isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")

    return all_results


if __name__ == "__main__":
    # Set up logging
    setup_base_logger("results/logs", level="DEBUG")
    # Run with small number of samples and batch size 1
    results = run_inference(num_samples=5, batch_size=1)
