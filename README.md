# Chainization
Official pytorch implementation of ECCV 2022 paper, "Order Learning Using Partially Ordered Data via Chainization."

## Dependencies
* Python 3.8
* Pytorch 1.7.1

## Datasets
- [MORPH II](https://ebill.uncw.edu/C20231_ustores/web/classic/product_detail.jsp?PRODUCTID=8) 
* For MORPH II experiments, we follow the same fold settings in this [OL](https://github.com/changsukim-ku/order-learning/tree/master/index) repo.
- [Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html)
* For Adience experiments, we follow the official splits.

## Quick Start : Train Model on Random Edge Cases

You can adjust supervision ratio by changing 'info_ratio' in the parse_option function.

* for Adience dataset

```

    $ python train_chainize_adience.py 
```

* for MORPH II dataset

```

    $ python train_chainize_morph.py
```

## Referecences
1. [FixMatch](https://github.com/google-research/fixmatch)
2. [POE](https://github.com/Li-Wanhua/POEs)
