import torch
import torch.nn as nn
import torchzq
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from ..models import Baseline, Encoder, Decoder, kl_divergence
from .baseline import Runner as BaselineRunner


class Runner(BaselineRunner):
    def __init__(self, baseline_ckpt: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)

    def create_model(self):
        args = self.args

        self.baseline = Baseline()
        if args.baseline_ckpt is not None:
            self.baseline.load_state_dict(
                torch.load(args.baseline_ckpt, "cpu")["model"]
            )
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

    def training_step(self, batch, _):
        x, y = batch

        x = x.flatten(1)
        y = y.flatten(1)

        y_ = self.baseline(x)
        z_prior = self.prior(x, y_)
        z = self.encoder(x, y)

        loss_kl = kl_divergence(
            self.encoder.normal,
            self.prior.normal,
        ).sum()

        logits = self.decoder(z)
        logits = logits.view(len(x), 1, 28, 28)

        x = x.view(len(x), 1, 28, 28)
        y = y.view(len(x), 1, 28, 28)

        loss_recon = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss_recon = loss_recon[x == -1].sum()

        self.logits = logits[:16]
        self.images = x[:16]

        loss = loss_recon + loss_kl

        return loss, dict(
            loss_recon=loss.item(),
            loss_kl=loss_kl.item(),
        )

    def testing_step(self):
        raise NotImplementedError


def main():
    torchzq.start(Runner)


if __name__ == "__main__":
    main()
