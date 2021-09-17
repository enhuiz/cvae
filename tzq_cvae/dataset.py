import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter


class BlurryMNIST(MNIST):
    def __init__(self, root, num_masks=3, **kwargs):
        self.num_masks = num_masks
        kwargs.setdefault("transform", transforms.ToTensor())
        kwargs.setdefault("download", True)
        super().__init__(root, **kwargs)

    def __getitem__(self, index):
        y, _ = super().__getitem__(index)
        x = gaussian_filter(y, sigma=3)
        return x, y

    def as_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, **kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = BlurryMNIST("data/mnist", download=True)

    x, y = dataset[0]
    print(x.shape)

    plt.subplot(121)
    plt.imshow(x[0])
    plt.subplot(122)
    plt.imshow(y[0])

    plt.savefig("test.png")
