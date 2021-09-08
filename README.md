# An implementation of conditional variational encoder (CVAE) based on TorchZQ

This repo partially implements the MNIST experiment in the paper: Learning Structured Output Representation using Deep Conditional Generative Models.

## Setup

```
pip install .
```

## Runs

### Baseline

```
tzq config/baseline.yml train
```

![](https://api.wandb.ai/files/enhuiz/tzq-cvae/1trw1cxc/media/images/generated_37500_0d09c87438665058c544.png)

### CVAE

```
tzq config/cvae.yml train
```

![](https://api.wandb.ai/files/enhuiz/tzq-cvae/nj7zwpbm/media/images/generated_37500_969349d3c67c75077138.png)

### Quantitative comparisons

All models are trained for 20 epochs with batch size 32 and learning rate `1e-3`. CVAE by default is not conditioned on the masked input (i.e. p(y|z) instead of p(y|z, x)).

| Method                                                 | NCLL (Importance Sampling (S = 100) ⬇️ |
| ------------------------------------------------------ | -------------------------------------- |
| Baseline                                               | 112.382                                |
| CVAE (w/ conditioned decoder, w/o baseline)            | 83.745                                 |
| CVAE (w/ conditioned decoder)                          | 79.524                                 |
| CVAE (w/o conditioned prior)                           | 76.024                                 |
| CVAE                                                   | 72.255                                 |
| CVAE (w/o baseline)                                    | 70.868                                 |
| CVAE (w/ jointly trained baseline from the pretrained) | 69.352                                 |
| CVAE (w/ jointly trained baseline from scratch)        | 67.813                                 |

- Baseline seems not helpful when decoder is not conditioned on the masked image.
- Conditioning on the decoder harms NCLL.

More details can be found [here](https://wandb.ai/enhuiz/tzq-cvae?workspace=user-enhuiz).

## Credits

- [Pyro's implementation](https://pyro.ai/examples/cvae.html#Baseline:-Deterministic-Neural-Network).
- [Learning Structured Output Representation using Deep Conditional Generative Models](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf).
