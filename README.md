# An Implementation of Conditional Variational Encoder (CVAE) based on TorchZQ

This repo implements the MNIST experiment in the paper: Learning Structured Output Representation using Deep Conditional Generative Models.

## Setup

```
pip install .
```

## Runs

### Baseline

```
tzq config/baseline.yml train
```

![](https://api.wandb.ai/files/enhuiz/tzq-cvae/1uf2lr09/media/images/recon_37500_69c4b7d9a6318dfdc3a6.png)

### CVAE

```
tzq config/cvae.yml train
```

![](https://api.wandb.ai/files/enhuiz/tzq-cvae/ujztc6o9/media/images/recon_37500_115b76141b2460c52b8b.png)

### CVAE w/o pretrained baselne

```
tzq config/baseline.yml train
```

![](https://api.wandb.ai/files/enhuiz/tzq-cvae/zyyhp9fj/media/images/recon_28125_5f58ad37d6dba07d472c.png)

More running details can be found [here](https://wandb.ai/enhuiz/tzq-cvae?workspace=user-enhuiz).

## Credits

- [Pyro's implementation](https://pyro.ai/examples/cvae.html#Baseline:-Deterministic-Neural-Network)
- [Learning Structured Output Representation using Deep Conditional Generative Models](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf).
