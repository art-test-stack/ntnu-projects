import torch
import torch.nn as nn


class VariationalDecoder(nn.Module):
    def __init__(self, latent_space_size: int = 64) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(latent_space_size, 64 * 5 * 5),
            nn.Unflatten(1, (64, 5, 5)),
            nn.ConvTranspose2d(64, 64, kernel_size=3),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )   

    def forward(self, x):
        out = self.main(x)
        return out