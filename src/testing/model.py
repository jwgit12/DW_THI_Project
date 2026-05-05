import torch
import torch.nn as nn


class MRDDenoiser(nn.Module):
    def __init__(self, in_channels=130):
        super().__init__()

        self.net = nn.Sequential(

            # Block 1
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),

            # Block 2
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            # Block 4
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            # Block 5
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),

            # Output
            nn.Conv2d(64, in_channels, 3, padding=1),
        )

    def forward(self, x):
        noise = self.net(x)
        return noise
