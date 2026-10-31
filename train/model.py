"""DS-CNN keyword spotter: input (B, 1, 49, 10) normalised MFCCs -> (B, 12) logits."""

import torch
from torch import nn

N_CLASSES = 12
CHANNELS = 64
N_BLOCKS = 4


def _block(ch: int) -> nn.Sequential:
    """Depthwise 3x3 -> BN -> ReLU -> pointwise 1x1 -> BN -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(ch, ch, 3, padding=1, groups=ch),
        nn.BatchNorm2d(ch),
        nn.ReLU(),
        nn.Conv2d(ch, ch, 1),
        nn.BatchNorm2d(ch),
        nn.ReLU(),
    )


class DSCNN(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, channels: int = CHANNELS) -> None:
        super().__init__()
        # 10 (time) x 4 (frequency) kernel, stride 2x2; padding keeps a 25 x 5 map.
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, (10, 4), stride=(2, 2), padding=(5, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[_block(channels) for _ in range(N_BLOCKS)])
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        return self.fc(x.mean(dim=(2, 3)))


def count_parameters(model: nn.Module) -> int:
    """Trainable parameters (BatchNorm running statistics are buffers and not counted)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
