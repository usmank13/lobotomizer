"""Example: Distillation with pre-computed teacher logits.

Useful when the teacher model is too large to hold in memory alongside
the student, or when teacher inference has already been done offline.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lobotomizer as lob


def main():
    torch.manual_seed(42)

    # 1. "Offline" step: generate teacher logits and save them
    teacher = nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 10)
    )
    teacher.eval()

    x_all = torch.randn(200, 64)
    y_all = torch.randint(0, 10, (200,))
    with torch.no_grad():
        teacher_logits = teacher(x_all)

    # 2. Create DataLoader with (inputs, teacher_logits, labels)
    train_loader = DataLoader(
        TensorDataset(x_all, teacher_logits, y_all),
        batch_size=32,
    )

    # 3. Distill student using pre-computed logits — no teacher model loaded
    student = nn.Sequential(
        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10)
    )

    result = lob.Pipeline([
        lob.Distill(
            method="logit",
            precomputed=True,
            alpha=0.8,
            task_loss_fn=nn.CrossEntropyLoss(),
            epochs=5,
            lr=1e-3,
        ),
    ]).run(student, training_data=train_loader, input_shape=(1, 64))

    print(result.summary())


if __name__ == "__main__":
    main()
