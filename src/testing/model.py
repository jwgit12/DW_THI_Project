import torch
import torch.nn as nn


class MRDDenoiser(nn.Module):
    def __init__(self, in_channels=130):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        noise = self.net(x)
        return noise
