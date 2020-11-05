# Simplifying Graph Attention Networks with Source-Target Separation

This is a pytorch implementation of the Separated Graph Attention Networks (SepGAT) and Simplified Graph Attention Networks (SimpGAT)
models as described in our paper
Simplifying Graph Attention Networks with Source-Target Separation (ECAI 2020): http://ecai2020.eu/papers/1617_paper.pdf

```
@inproceedings{DBLP:conf/ecai/Guo0FGZ20,
  author    = {Hantao Guo and
               Rui Yan and
               Yansong Feng and
               Xuesong Gao and
               Zhanxing Zhu},
  editor    = {Giuseppe De Giacomo and
               Alejandro Catal{\'{a}} and
               Bistra Dilkina and
               Michela Milano and
               Sen{\'{e}}n Barro and
               Alberto Bugar{\'{\i}}n and
               J{\'{e}}r{\^{o}}me Lang},
  title     = {Simplifying Graph Attention Networks with Source-Target Separation},
  booktitle = {{ECAI} 2020 - 24th European Conference on Artificial Intelligence,
               29 August-8 September 2020, Santiago de Compostela, Spain, August
               29 - September 8, 2020 - Including 10th Conference on Prestigious
               Applications of Artificial Intelligence {(PAIS} 2020)},
  series    = {Frontiers in Artificial Intelligence and Applications},
  volume    = {325},
  pages     = {1166--1173},
  publisher = {{IOS} Press},
  year      = {2020},
  url       = {https://doi.org/10.3233/FAIA200215},
  doi       = {10.3233/FAIA200215},
  timestamp = {Tue, 15 Sep 2020 15:08:42 +0200},
  biburl    = {https://dblp.org/rec/conf/ecai/Guo0FGZ20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Dependencies

This implementation is known to work on Python 3.6 and PyTorch>=1.0.0, with also following packages installed:
* `numpy==1.16.2`
* `scipy==1.2.1`
* `networkx==2.2`
* `scikit-learn==0.20.3`

## Data

We provide the citation network datasets under `data/`, which corresponds to [the public data splits](https://github.com/tkipf/gcn/tree/master/gcn/data).
Due to space limit, please download reddit dataset from [FastGCN](https://github.com/matenure/FastGCN/issues/9) and put `reddit_adj.npz`, `reddit.npz` under `data/`.

## Usage

```
python train.py --dataset cora --model SepGAT
python train.py --dataset cora --model SimpGAT
``` 

Other parameters are documented in `train.py`.

## Acknowledgement

This repo is modified from [gcn](https://github.com/tkipf/gcn), [pygcn](https://github.com/tkipf/pygcn), [GAT](https://github.com/PetarV-/GAT), and [SGC](https://github.com/Tiiiger/SGC)
