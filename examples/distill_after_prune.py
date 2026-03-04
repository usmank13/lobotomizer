"""Example: Prune a model then recover quality with knowledge distillation.

The original (pre-prune) model serves as the teacher automatically.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lobotomizer as lob


def main():
    # 1. Create a simple model and synthetic training data
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    x = torch.randn(200, 128)
    y = torch.randint(0, 10, (200,))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=32)

    # 2. Prune then distill — teacher is automatically the pre-prune model
    result = lob.Pipeline([
        lob.StructuredPrune(sparsity=0.3),
        lob.Distill(method="feature", epochs=5, lr=1e-3),
    ]).run(model, training_data=train_loader, input_shape=(1, 128))

    print(result.summary())


if __name__ == "__main__":
    main()
