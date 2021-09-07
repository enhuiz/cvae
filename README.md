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

![](https://api.wandb.ai/files/enhuiz/tzq-cvae/xngngp5q/media/images/generated_37500_f10c1354dfbbe13177a7.png)

### CVAE

```
tzq config/cvae.yml train
```

![](https://api.wandb.ai/files/enhuiz/tzq-cvae/1d3yv80a/media/images/generated_37500_8faf8d4d84bf3d886e92.png)

### CVAE w/o pretrained baseline

```
tzq config/cvae-wopt.yml train
```

![](https://api.wandb.ai/files/enhuiz/tzq-cvae/cln5yruy/media/images/generated_37500_91f5f10872277857c404.png)

### CVAE w/ frozen pretrained baseline

```
tzq config/cvae-freeze-baseline.yml train
```

![](https://api.wandb.ai/files/enhuiz/tzq-cvae/r3rrhzor/media/images/generated_37500_85c1afd74377bcd36eb9.png)

### CVAE w/ latent variable comes from prior network fed into decoder during training

```
tzq config/cvae-prior-z.yml
```

![](https://api.wandb.ai/files/enhuiz/tzq-cvae/1enp48ov/media/images/generated_37500_8c5d69cffcde0b0ae29f.png)

More running details can be found [here](https://wandb.ai/enhuiz/tzq-cvae?workspace=user-enhuiz).

## Credits

- [Pyro's implementation](https://pyro.ai/examples/cvae.html#Baseline:-Deterministic-Neural-Network)
- [Learning Structured Output Representation using Deep Conditional Generative Models](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf).
