# AlexNet on CIFAR-10 (Consumer-GPU Friendly)

A clean, reproducible PyTorch implementation of an AlexNet-style network
adapted for CIFAR-10 (32x32 input), designed to train on low-VRAM consumer
GPUs. The project keeps the classic "5 conv + 3 fc" AlexNet structure while
supporting AMP mixed precision, gradient accumulation, and standard
regularization for small-dataset training.

## Results

Final evaluation on the CIFAR-10 test set (10,000 images):

| Metric | Value |
| :--- | :--- |
| Test loss | **0.6442** |
| Test top-1 accuracy | **88.71%** |
| Test top-5 accuracy | **99.54%** |
| Train top-1 accuracy | 99.54% |

The ~11 point train-test gap is expected for AlexNet on CIFAR-10: the two
4096-dim fully connected layers account for roughly 54.5M of the ~60M
parameters, which makes the model prone to memorizing the 50k training
images. The test top-1 falls within the typical 85%-90% range reported for
AlexNet-style models on CIFAR-10, indicating a healthy reproduction rather
than an implementation bug.

## Features

- 32x32 input adaptation that preserves the 5-conv + 3-fc AlexNet layout
- AMP mixed precision on CUDA (`torch.autocast` + `GradScaler`)
- Gradient accumulation to simulate large batches on small GPUs
- Cosine / Step learning rate schedulers
- best / last checkpointing and JSON training history
- Top-1 and Top-5 evaluation consistent with ImageNet conventions

## Environment

- Python >= 3.9
- PyTorch >= 2.0
- torchvision >= 0.15
- CUDA GPU recommended; CPU training is possible but slow

Install dependencies:

```bash
pip install -r requirements.txt