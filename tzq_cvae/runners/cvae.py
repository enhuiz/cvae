import math
import torch
import torch.nn as nn
import torchzq
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from einops import rearrange, repeat

from ..models import Baseline, Encoder, Decoder, kl_divergence
from .baseline import Runner as BaselineRunner


def logmeanexp(x, dim=0):
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])


class Runner(BaselineRunner):
    def __init__(
        self,
        baseline_ckpt: Optional[Path] = None,
        sample_from_prior_when_training: bool = False,
        freeze_baseline: bool = True,
        importance_sampling_s: int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)

    def create_model(self):
        args = self.args

        self.baseline = Baseline()

        if args.baseline_ckpt is not None:
            self.baseline.load_state_dict(
                torch.load(args.baseline_ckpt, "cpu")["model"]
            )

        if args.freeze_baseline:
            for p in self.baseline.parameters():
                p.requires_grad_(False)

        self.prior = Encoder()
        self.encoder = Encoder()
        self.decoder = Decoder()

        return nn.ModuleDict(
            dict(
                baseline=self.baseline,
                prior=self.prior,
                encoder=self.encoder,
                decoder=self.decoder,
            ),
        )

    @torch.no_grad()
    def generate(self, x):
        x = self.flatten(x)
        z_prior = self.prior(x, self.baseline(x))
        logits = self.decoder(z_prior)
        return self.unflatten(logits).sigmoid()

    def training_step(self, batch, _):
        args = self.args
        x, y = batch

        x = self.flatten(x)
        y = self.flatten(y)

        z_posterior = self.encoder(x, y)

        z_prior = self.prior(x, self.baseline(x))

        if not args.sample_from_prior_when_training and self.model.training:
            z = z_posterior
        else:
            z = z_prior

        kl = kl_divergence(self.encoder.normal, self.prior.normal)
        loss_kl = kl.mean(dim=0).sum()

        logits = self.unflatten(self.decoder(z))

        x = self.unflatten(x)
        y = self.unflatten(y)

        masked = x == -1  # (b 1 h w)
        loss_bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss_bce = loss_bce[masked].sum() / len(x)

        loss = loss_bce + loss_kl

        stat_dict = dict(loss_bce=loss_bce.item(), loss_kl=loss_kl.item())

        if not self.model.training:
            z = self.encoder.normal.sample((args.importance_sampling_s,))

            logr = self.prior.normal.log_prob(z) - self.encoder.normal.log_prob(z)
            logr = rearrange(logr.sum(dim=-1), "s b -> (s b)")

            z = rearrange(z, "s b d -> (s b) d")
            y = repeat(y, "b ... -> (s b) ...", s=args.importance_sampling_s)
            logits = self.unflatten(self.decoder(z))

            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            masked = repeat(masked, "b ... -> (s b) ...", s=args.importance_sampling_s)
            bce[~masked] = 0  # consider only masked pixels
            bce = bce.flatten(1).sum(dim=-1)

            logp = rearrange(logr - bce, "(s b) -> s b", b=len(x))
            stat_dict["ncll"] = logmeanexp(logp, dim=0).neg().mean().item()

        return loss, stat_dict


def main():
    torchzq.start(Runner)


if __name__ == "__main__":
    main()
