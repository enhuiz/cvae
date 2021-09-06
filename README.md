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

![](https://storage.googleapis.com/wandb-production.appspot.com/enhuiz/tzq-cvae/ujztc6o9/media/images/recon_37500_115b76141b2460c52b8b.png?Expires=1630950479&GoogleAccessId=wandb-production%40appspot.gserviceaccount.com&Signature=TtM0PRCXcgaUE5qBQX12AKYhG%2B7HJikgsp%2BSi%2FQlMaMrbMLd0wR1BDN3KQNfTPSXj3vipE2aC8tiWQGZpa%2FQHlbgy5aI43OU5VmCmAqOofwxFpiy7UWwm3IwxuE3aIj3%2BERm7xlw%2Bf10V5%2BkP8VC60kHUAIKPZvMJQ06mdZJlwnzlTbnXkp1Q0cCwhMt9NuUtDdYugaoWfIaDK8uX4aolPqx9sznyfzrxUtzDyOuLoK5tIhq7qdUeHVjf0nNHhDH2AmCze3dM9zTcoRkgNJ4DPsqYwSLWvokAjMwfnuc%2FwfGXym3C46zSTJnqUbRsZ2srBbOSwB24s1ChkP8tcvI%2FA%3D%3D)

### CVAE w/o pretrained baselne

```
tzq config/baseline.yml train
```

![](https://storage.googleapis.com/wandb-production.appspot.com/enhuiz/tzq-cvae/zyyhp9fj/media/images/recon_28125_5f58ad37d6dba07d472c.png?Expires=1630950504&GoogleAccessId=wandb-production%40appspot.gserviceaccount.com&Signature=eGU8aS%2F6yvMyWLh206zgOTpwLJYjRoCRskS1IUsFM59KjSOfF59gsOPav%2BPxGUVNEqtw6ku3HY6U2sAiqGVsB6kEVezhmRrhWkaA1M2NvjnXgeTeTt2w5y1pAHEvRza8trvNC71GcnkSXaCH%2B%2Bm4%2Btjc8ohiz9fpj5dMQLf77IQ6rMVCZ8M7aG9h7M0c2frpuW1kIKagO9u6%2BzH4ydtYPvKKpHzXeR846HvGwshkpDtGgoGTCeK%2F4f9wWYsS%2BpTML4qU0h2IINzeGsYnEpkNy2YpIgB4lZfskF3gFNC7%2F40drhygicGYgq0NQW4gm60vmeOzdNxPZezMf5ErmiwROw%3D%3D)

More running details can be found [here](https://wandb.ai/enhuiz/tzq-cvae?workspace=user-enhuiz).

## Credits

- [Pyro's implementation](https://pyro.ai/examples/cvae.html#Baseline:-Deterministic-Neural-Network)
- [Learning Structured Output Representation using Deep Conditional Generative Models](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf).
