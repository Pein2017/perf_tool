# Performance Monitoring Tool

A Python package for monitoring ML model inference, providing time collection and precision monitoring capabilities with a flexible and intuitive API.

## Features

1. **Time Collection**
   - Thread-safe execution time tracking
   - Hierarchical monitoring support
   - Automatic call stack management
   - Detailed statistics (min, max, mean, per-run times)
   - CSV export functionality

2. **Precision Monitoring**
   - Seamless integration with msprobe.pytorch
   - Automatic dump file organization by run and stage
   - Thread-safe monitoring
   - Configurable monitoring options
   - Clean resource management

3. **Core Features**
   - Thread-safe global state management
   - Flexible monitoring interfaces (decorators and context managers)
   - Hierarchical monitoring with automatic tag suffixing
   - Comprehensive logging system
   - Type-safe implementations with proper type hints

## Project Structure

```
core/
├── __init__.py
├── base/
│   ├── __init__.py
│   ├── base_collector.py      # Abstract base collector with thread safety
│   └── monitor_config.py      # Configuration dataclass
├── collectors/
│   ├── __init__.py
│   ├── time_collector.py      # Time monitoring implementation
│   └── precision_collector.py # Precision monitoring implementation
├── decorators/
│   ├── __init__.py
│   └── monitor.py            # Thread-safe decorator implementation
├── manager/
│   ├── __init__.py
│   ├── collector_manager.py  # Unified monitoring manager
│   └── context.py           # Context manager protocol and implementation
└── storage/
    ├── __init__.py
    └── tag_manager.py       # Tag management for monitoring runs
```

## Key Components

### BaseCollector
- Thread-safe context tracking
- Automatic cleanup on run changes
- Abstract interface for collectors
- Built-in decorator support
- Context manager support

### TimeCollector
- Efficient time tracking with active timers
- Comprehensive statistics calculation
- Per-run time storage
- CSV export functionality

### PrecisionCollector
- Automatic dump path management
- Clean debugger lifecycle
- Resource cleanup on run changes
- Integration with msprobe.pytorch

### CollectorManager
- Unified interface for multiple collectors
- Configuration management
- Automatic registration system
- Flexible monitoring interfaces

### Monitoring Context
- Clean protocol-based design
- Automatic resource management
- Exception-safe implementation

## Usage Examples

### 1. Basic Setup

```python
from refactor_perf_checker import (
    CollectorManager,
    monitor,
    set_default_manager,
    setup_base_logger,
)

def initialize_monitoring(
    enable_time: bool = True,
    enable_precision: bool = True,
    precision_config: str = "configs/precision_config.json",
) -> CollectorManager:
    """Initialize the monitoring system."""
    manager = CollectorManager(
        enable_time=enable_time,
        enable_precision=enable_precision,
        precision_config=precision_config,
    )
    set_default_manager(manager)
    return manager

# Initialize monitoring with logging
setup_base_logger("results/logs", level="DEBUG")
manager = initialize_monitoring(
    enable_time=True,
    enable_precision=True,
    precision_config="configs/precision_config.json",
)
```

### 2. Using Decorators

```python
# Preprocessing with time and precision monitoring
@monitor("preprocessing", "resize", enable_time=True, enable_precision=True)
def resize_image(data: torch.Tensor) -> torch.Tensor:
    """Resize images to target size."""
    return F.interpolate(data, size=(32, 32), mode="bilinear", align_corners=False)

# Postprocessing with time monitoring only
@monitor("postprocessing", "decode", enable_time=True, enable_precision=False)
def decode_prediction(logits: torch.Tensor) -> List[Dict]:
    """Decode model predictions."""
    classes = ["cat", "dog", "bird"]
    probs = F.softmax(logits, dim=1).cpu()
    
    results = []
    for i in range(logits.size(0)):
        sample_probs = probs[i].tolist()
        results.append({
            "id": i,
            "predictions": [
                {"class": cls, "score": score}
                for cls, score in zip(classes, sample_probs)
            ],
        })
    return results
```

### 3. Using Context Managers

```python
def process_batch(batch: torch.Tensor, model_1: nn.Module, model_2: nn.Module, manager: CollectorManager) -> List[Dict]:
    """Example of mixing decorators and context managers."""
    # Pre-processing with decorators
    data = batch.to(DEVICE)
    data = resize_image(data)  # Uses @monitor decorator
    data = augment_image(data)  # Uses @monitor decorator

    # Model inference with context managers
    with torch.no_grad():
        # Model 1 inference
        with manager.monitor_context("inference", "model1.forward"):
            logits_1 = model_1(data)

        with manager.monitor_context("inference", "model1.decode"):
            pred_1 = decode_prediction(logits_1)  # Uses @monitor decorator

        # Model 2 inference
        with manager.monitor_context("inference", "model2.forward"):
            logits_2 = model_2(data)

        with manager.monitor_context("inference", "model2.decode"):
            pred_2 = decode_prediction(logits_2)

    # Post-processing with decorators
    return fuse_predictions(pred_1, pred_2)
```

### 4. Batch Processing and Statistics

```python
def run_inference(num_samples: int = 4, batch_size: int = 1) -> List[Dict]:
    """Complete inference pipeline with monitoring."""
    # Initialize models
    model_1 = SimpleNet().to(DEVICE)
    model_2 = SimpleNetV2().to(DEVICE)
    model_1.eval()
    model_2.eval()

    # Create dataset and dataloader
    dataset = DummyDataset(num_samples=num_samples)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_results = []

    # Process each batch as a separate run
    for batch_idx, batch in enumerate(data_loader):
        # Set the batch index as the run ID for monitoring
        manager.set_run(batch_idx)
        
        # Process batch with monitoring
        batch_results = process_batch(
            batch=batch,
            model_1=model_1,
            model_2=model_2,
            manager=manager
        )
        all_results.extend(batch_results)

    # Export and analyze results
    csv_path = manager.export_time_stats("results/time")
    time_stats = manager.get_time_stats()
    
    # Print timing statistics
    for stage in time_stats:
        print(f"\nStage: {stage}")
        for tag, stats in time_stats[stage].items():
            print(f"  {tag}:")
            for metric, value in stats.items():
                if metric != "times_by_run" and isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.4f}")

    return all_results
```

### 5. Example Model Architectures

```python
class SimpleNet(nn.Module):
    """Example model for precision monitoring."""
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
```

## Thread Safety

The implementation provides comprehensive thread safety through:
- Lock-protected context stacks
- Thread-local monitoring contexts
- Safe run transitions
- Protected global state management

## Best Practices

1. **Initialization**
   - Always set up a default manager for decorators
   - Configure monitoring options at initialization
   - Use appropriate configuration files

2. **Monitoring Patterns**
   - Use decorators for whole-function monitoring
   - Use context managers for specific blocks
   - Set run IDs for batch processing
   - Clean up resources properly

3. **Resource Management**
   - Use context managers for automatic cleanup
   - Update run IDs between batches
   - Export results after completion
   - Monitor memory usage with precision monitoring

4. **Error Handling**
   - Resources are automatically cleaned up
   - Exceptions are properly propagated
   - Logging provides debugging information
   - Thread safety is maintained

## License

This project is licensed under the MIT License.
