import math
import torch
import torch.nn as nn
import torchzq
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from einops import rearrange, repeat

from ..models import Baseline, Encoder, DumbEncoder, Decoder, kl_divergence
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
        conditioned_decoder: bool = False,
        conditioned_prior: bool = True,
        use_baseline: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

    def create_model(self):
        args = self.args

        if self.use_baseline:
            self.baseline = Baseline()

            if args.baseline_ckpt is not None:
                self.baseline.load_state_dict(
                    torch.load(args.baseline_ckpt, "cpu")["model"]
                )

            if args.freeze_baseline:
                for p in self.baseline.parameters():
                    p.requires_grad_(False)

        self.prior = Encoder() if args.conditioned_prior else DumbEncoder()
        self.encoder = Encoder()
        self.decoder = Decoder(args.conditioned_decoder)

        mapping = dict(
            prior=self.prior,
            encoder=self.encoder,
            decoder=self.decoder,
        )

        if args.use_baseline:
            mapping["baseline"] = self.baseline

        return nn.ModuleDict(mapping)

    @torch.no_grad()
    def generate(self, x):
        args = self.args
        x = self.flatten(x)
        z = self.prior(x, self.baseline(x) if self.use_baseline else None)
        if args.conditioned_decoder:
            z = torch.cat([z, x], dim=-1)
        logits = self.decoder(z)
        return self.unflatten(logits).sigmoid()

    def compute_ncll(self, x, y):
        args = self.args

        z = self.encoder.normal.sample((args.importance_sampling_s,))

        logr = self.prior.normal.log_prob(z) - self.encoder.normal.log_prob(z)
        logr = rearrange(logr.sum(dim=-1), "s b -> (s b)")

        z = rearrange(z, "s b d -> (s b) d")
        y = repeat(y, "b ... -> (s b) ...", s=args.importance_sampling_s)
        unmasked = repeat(x != -1, "b ... -> (s b) ...", s=args.importance_sampling_s)

        if args.conditioned_decoder:
            x = self.flatten(x)
            x = repeat(x, "b d -> (s b) d", s=args.importance_sampling_s)
            z = torch.cat([z, x], dim=-1)

        logits = self.unflatten(self.decoder(z))

        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        bce[unmasked] = 0  # consider only masked pixels
        bce = bce.flatten(1).sum(dim=-1)

        logp = rearrange(logr - bce, "(s b) -> s b", b=len(x))
        ncll = logmeanexp(logp, dim=0).neg().mean().item()

        return ncll

    @property
    def use_baseline(self):
        args = self.args
        return args.use_baseline and args.conditioned_prior

    def training_step(self, batch, _):
        args = self.args
        x, y = batch

        x = self.flatten(x)
        y = self.flatten(y)

        z_posterior = self.encoder(x, y)

        z_prior = self.prior(x, self.baseline(x) if self.use_baseline else None)

        if not args.sample_from_prior_when_training and self.model.training:
            z = z_posterior
        else:
            z = z_prior

        kl = kl_divergence(self.encoder.normal, self.prior.normal)
        loss_kl = kl.mean(dim=0).sum()

        if args.conditioned_decoder:
            z = torch.cat([z, x], dim=-1)

        logits = self.unflatten(self.decoder(z))

        x = self.unflatten(x)
        y = self.unflatten(y)

        masked = x == -1  # (b 1 h w)
        loss_bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss_bce = loss_bce[masked].sum() / len(x)

        loss = loss_bce + loss_kl

        stat_dict = dict(loss_bce=loss_bce.item(), loss_kl=loss_kl.item())

        if not self.model.training:
            stat_dict["ncll"] = self.compute_ncll(x, y)

        return loss, stat_dict


def main():
    torchzq.start(Runner)


if __name__ == "__main__":
    main()
