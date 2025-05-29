# HyperCRS

The code is partially referred to [MHCPL](https://github.com/Snnzhao/MHCPL).

## Datasets
https://pan.quark.cn/s/b7110d2f4099#/list/share/1dad4e9516d44758a6717dc22b0096d3-HyperCRS

## Environment Settings
python: 3.8.0

pytorch: 1.8.1 

dgl: 0.8.1

## Training
`python RL_model.py --data_name <data_name>`

## Evaluation
`python evaluate.py --data_name <data_name> --load_rl_epoch <checkpoint_epoch>`

