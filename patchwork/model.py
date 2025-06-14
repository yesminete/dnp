import torch
import torch.nn as nn
from typing import Callable, Iterable


class PatchWorkModel(nn.Module):
    """Simplified Patchwork model implemented in PyTorch."""

    def __init__(self,
                 block_creator: Callable[[int, int], nn.Module] = None,
                 num_labels: int = 1,
                 num_classes: int = 1,
                 depth: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_channels = num_labels
        for d in range(depth):
            if block_creator is None:
                block = nn.Identity()
            else:
                block = block_creator(in_channels, d)
            if hasattr(block, 'out_channels'):
                in_channels = block.out_channels
            self.blocks.append(block)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        dims = list(range(2, x.dim()))
        x = x.mean(dim=dims)
        return self.classifier(x)


def train_loop(model: nn.Module,
               dataloader: Iterable,
               optimizer: torch.optim.Optimizer,
               criterion: Callable,
               device: str = 'cpu',
               epochs: int = 1):
    """Simple PyTorch training loop."""
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        

__all__ = ['PatchWorkModel', 'train_loop']
