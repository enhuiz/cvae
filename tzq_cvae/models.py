import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from .utils import ModuleTraits


class Baseline(nn.Sequential):
    def __init__(self, dim_in=784, dim_out=784):
        super().__init__(
            nn.Linear(dim_in, 500),
            nn.GELU(),
            nn.Linear(500, 500),
            nn.GELU(),
            nn.Linear(500, dim_out),
        )


class EncoderTraits(ModuleTraits):
    @property
    def normal(self) -> Normal:
        return self._saved_for_later["normal"]


class Encoder(EncoderTraits, Baseline):
    def __init__(self):
        super().__init__(dim_out=400)

    def forward(self, x, y_=None):
        if y_ is not None:
            x = x.clone()
            x[x == -1] = y_[x == -1]
        μ, logσ = super().forward(x).chunk(2, dim=-1)
        σ = logσ.exp()
        z = μ + σ * torch.randn_like(μ)
        self.save_for_later(normal=Normal(μ, σ))
        return z


class DumbEncoder(EncoderTraits, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, _):
        μ = torch.zeros(len(x), 200, device=x.device)
        normal = Normal(μ, torch.ones_like(μ))
        self.save_for_later(normal=normal)
        return normal.sample()


class Decoder(Baseline):
    def __init__(self, conditioned=False):
        super().__init__(dim_in=984 if conditioned else 200)


if __name__ == "__main__":
    x = torch.randn(3, 28, 28)
    y = torch.randn(3, 28, 28)

    baseline = Baseline()
    prior = Encoder()
    encoder = Encoder()
    decoder = Decoder()

    x = x.flatten(1)
    y = y.flatten(1)

    y_ = baseline(x)
    z_prior = prior(x, y_)
    z_posterior = encoder(x, y)

    kl = kl_divergence(encoder.normal, prior.normal)

    x = decoder(z_prior)
    x = x.view(3, 28, 28)

    print(x.shape)
