import torch
import torch.nn as nn
import torchzq
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from ..models import Baseline, Encoder, Decoder, kl_divergence
from .baseline import Runner as BaselineRunner


class Runner(BaselineRunner):
    def __init__(
        self,
        baseline_ckpt: Optional[Path] = None,
        use_z_posterior_in_training: bool = True,
        freeze_baseline: bool = False,
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

        y_ = self.baseline(x)

        z_prior = self.prior(x, y_)
        z_posterior = self.encoder(x, y)

        if self.model.training and args.use_z_posterior_in_training:
            z = z_posterior
        else:
            z = z_prior

        loss_kl = kl_divergence(
            self.encoder.normal,
            self.prior.normal,
        ).sum()

        logits = self.unflatten(self.decoder(z))
        x = self.unflatten(x)
        y = self.unflatten(y)

        loss_bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss_bce = loss_bce[x == -1].sum()

        self.logits = logits[:16]
        self.images = x[:16]

        loss = loss_bce + loss_kl

        return loss, dict(
            loss_bce=loss_bce.item(),
            loss_kl=loss_kl.item(),
        )


def main():
    torchzq.start(Runner)


if __name__ == "__main__":
    main()
