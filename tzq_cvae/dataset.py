import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from functools import cached_property


class MaskedMNIST(MNIST):
    def __init__(self, root, num_masks=3, **kwargs):
        self.num_masks = num_masks
        kwargs.setdefault("transform", transforms.ToTensor())
        kwargs.setdefault("download", True)
        super().__init__(root, **kwargs)

    @cached_property
    def _quads(self):
        return [
            (slice(None), slice(i, i + 14), slice(j, j + 14))
            for i in [0, 14]
            for j in [0, 14]
        ]

    def _sample_quads(self):
        indices = np.random.choice(4, self.num_masks, False)
        return [self._quads[i] for i in indices]

    def __getitem__(self, index):
        x, _ = super().__getitem__(index)
        y = x.clone()
        quads = self._sample_quads()
        for quad in quads:
            x[quad] = -1
        return x, y

    def as_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, **kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = MaskedMNIST("data/mnist", download=True)

    x, y = dataset[0]
    print(x.shape)

    plt.subplot(121)
    plt.imshow(x[0])
    plt.subplot(122)
    plt.imshow(y[0])

    plt.savefig("test.png")
