"""Example: Cross-model knowledge distillation (ResNet101 → ResNet18).

Requires torchvision. Falls back to simple MLPs if not installed.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lobotomizer as lob


def main():
    torch.manual_seed(42)

    try:
        from torchvision.models import resnet18, resnet101

        teacher = resnet101(num_classes=10)
        student = resnet18(num_classes=10)
        x = torch.randn(64, 3, 32, 32)
        input_shape = (1, 3, 32, 32)
    except ImportError:
        print("torchvision not found, using simple MLPs")
        teacher = nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 10))
        student = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        x = torch.randn(64, 128)
        input_shape = (1, 128)

    y = torch.randint(0, 10, (64,))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=16)

    result = lob.Pipeline([
        lob.Distill(
            teacher=teacher,
            method="logit",
            temperature=6.0,
            alpha=0.7,
            task_loss_fn=nn.CrossEntropyLoss(),
            epochs=3,
            lr=1e-3,
        ),
    ]).run(student, training_data=train_loader, input_shape=input_shape)

    print(result.summary())


if __name__ == "__main__":
    main()
