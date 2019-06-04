# gcn-data-alignment

Implementation of data alignment on Graph Convolutional Networks.

Installation
------------

```python setup.py install```

Run the demo
------------
```
cd alignment
jupyter notebook demo.ipynb
```

How to use the code
------------
```python
from alignment.optimizations import optimize_dim_subspaces

optimize_dim_subspaces(
    dataset="constructive_example",
    num_rdm=2,
    num_k=5,
    num_scanning=1,
    norm_type="Frobenius-Norm",
    log=False,
    heatmap=True
)
```

Cite
------------
Please cite our paper if you use this code in your own work:
```
@article{qian2019quantifying,
  title={Quantifying the alignment of graph and features in deep learning},
  author={Qian, Yifan and Expert, Paul and Rieu, Tom and Panzarasa, Pietro and Barahona, Mauricio},
  journal={arXiv preprint arXiv:1905.12921},
  year={2019}
}
```
