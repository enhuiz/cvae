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
            num_workers=args.nj,
        )

    def create_model(self):
        return Baseline()

    def training_step(self, batch, _):
        x, y = batch

        logits = self.model(x.flatten(1))
        logits = logits.view(len(x), 1, 28, 28)

        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = loss[x == -1].sum()

        # save for vis
        self.images = x[:16]
        self.logits = logits[:16]

        return loss, dict(loss=loss.item())

    def validation_step(self, batch, batch_idx):
        stat_dict = super().validation_step(batch, batch_idx)
        if batch_idx == 0:
            images = self.logits.sigmoid()
            unmasked = self.images != -1
            images[unmasked] = self.images[unmasked]
            self.logger.log(
                dict(recon=self.logger.Image(images)),
                self.global_step,
            )
        return stat_dict

    def testing_step(self):
        raise NotImplementedError


def main():
    torchzq.start(Runner)


if __name__ == "__main__":
    main()
