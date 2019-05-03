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
