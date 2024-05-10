import torch
import torch.nn as nn


class VariationalEncoder(nn.Module):
    def __init__(self, latent_space_size: int = 64) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(.2),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
        )
        self.mu = nn.Linear(64 * 5 * 5, latent_space_size)
        self.log_var = nn.Linear(64 * 5 * 5, latent_space_size)

    def forward(self, x):
        out = self.main(x)

        mu = self.mu(out)
        log_var = self.log_var(out)

        return mu, log_var