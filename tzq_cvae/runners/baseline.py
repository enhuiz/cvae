import numpy as np
import torch
import torchzq
import torch.nn.functional as F

from ..dataset import MaskedMNIST
from ..models import Baseline


class Runner(torchzq.Runner):
    def __init__(self, wandb_project: str = "tzq-cvae", **kwargs):
        super().__init__(**kwargs)

    def create_dataloader(self, mode):
        args = self.args
        return MaskedMNIST("data/mnist").as_dataloader(
            batch_size=args.batch_size,
            drop_last=mode == mode.TRAIN,
            shuffle=mode == mode.TRAIN,
            worker_init_fn=np.random.seed(0),
            num_workers=args.nj,
        )

    def create_model(self):
        return Baseline()

    @staticmethod
    def flatten(x):
        return x.flatten(1)

    @staticmethod
    def unflatten(x):
        return x.view(len(x), 1, 28, 28)

    @torch.no_grad()
    def generate(self, x):
        return self.unflatten(self.model(self.flatten(x))).sigmoid()

    def training_step(self, batch, _):
        x, y = batch

        logits = self.unflatten(self.model(self.flatten(x)))

        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = loss[x == -1].sum()

        # save for vis
        self.images = x[:16]
        self.logits = logits[:16]

        return loss, dict(loss_bce=loss.item())

    def validation_step(self, batch, batch_idx):
        stat_dict = super().validation_step(batch, batch_idx)
        if batch_idx == 0:
            x = batch[0]
            images = self.generate(x)
            images[x != -1] = x[x != -1]
            self.logger.log(
                dict(generated=self.logger.Image(images)),
                self.global_step,
            )
        return stat_dict

    def testing_step(self):
        raise NotImplementedError


def main():
    torchzq.start(Runner)


if __name__ == "__main__":
    main()
