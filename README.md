# Learning Optimal Policy in MARL

This repository is an implementation of **LOP: Learning Optimal Policy under Partial Observability in Cooperative Multi-Agent Reinforcement Learning.** The framework for LOP is inherited from [PyMARL](https://github.com/oxwhirl/pymarl).  LOP is written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.


## Setup

Set up the working environment:

```shell
pip install -r requirements.txt 
```

Set up the StarCraftII game core

```shell
bash install_sc2.sh  
```

## Run an experiment 

To train `LOP`  on the map with `MMM2`, 

```shell
python src/main.py --config=lop --env-config=sc2 --map=MMM2
```

All results will be stored in the `Results` folder.



## Licence

The MIT License


