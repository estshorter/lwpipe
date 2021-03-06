[![PyPI version](https://badge.fury.io/py/lwpipe.svg)](https://badge.fury.io/py/lwpipe)
[![license](https://img.shields.io/pypi/l/lwpipe)](https://github.com/estshorter/lwpipe/blob/master/LICENSE)
[![python](https://img.shields.io/pypi/pyversions/lwpipe)](https://badge.fury.io/py/lwpipe)

# Overview
**lwpipe** provides a lightweight pipeline. lwpipe has fewer features than [luigi](https://github.com/spotify/luigi) or [Kedro](https://github.com/quantumblacklabs/kedro), but you can quickly build and run pipelines.

Note that lwpipe is highly inspired by [Kedro](https://github.com/quantumblacklabs/kedro).

# Installation
from pypi: 
``` sh
pip install lwpipe
```

# Usage
Minimal example (of course, no need to use this library..):
``` python
from lwpipe import Node, Pipeline

nodes = [
    Node(func=lambda x,y: x+y, inputs=(1,2)),
    Node(func=lambda x: x**2),
]

pipe = Pipeline(nodes)
outputs = pipe.run()
assert outputs[0] == 9
```

Example with interim data output:
``` python
from lwpipe import Node, Pipeline
from lwpipe.io import dump_pickle, load_pickle

def time_consuming_func(x):
    return x

nodes = [
    Node(
        func=time_consuming_func,
        inputs=100,
        outputs_dumper=dump_pickle,
        outputs_path="interim_data.pickle",
        outputs_loader=load_pickle, # needed to bypass this node
    ),
    Node(func=lambda x: x**2, name="square"),
]

pipe = Pipeline(nodes)
outputs = pipe.run()
assert outputs[0] == 10000
```
Once the first node is executed, you can bypass the node by `pipe.run(1)` or `pipe.run("square")`.

Multiple outputs with numpy:
``` python
import numpy as np
from lwpipe import Node, Pipeline
from lwpipe.io import dump_npy, load_npy

def split(x):
    return x[:5], x[5:]

nodes = [
    Node(
        func=split,
        inputs=np.arange(10),
        outputs=("former", "latter"),
        outputs_dumper=dump_npy,
        outputs_path=("df1.npy", "df2.npy"),
        outputs_loader=load_npy,
    ),
    Node(
        func=np.mean,
        name="former_mean",
        inputs="former", # calculated at the first node
        outputs="former_mean",
    ),
    Node(
        func=np.mean,
        name="latter_mean",
        inputs="latter", # calculated at the first node
        outputs="latter_mean",
    ),
]

pipe = Pipeline(nodes)
outputs = pipe.run()
assert outputs[0] == 7.0
# You can access interim results by "results" dict
assert pipe.results["former_mean"] == 2.0
```

batch dump example (return values are dumped to one file):
``` python
import numpy as np
from lwpipe import DumpType, Node, Pipeline
from lwpipe.io import (
    dump_dict_pickle,
    dump_savez_compressed,
    load_dict_pickle,
    load_savez_compressed,
)

def divide(x):
    return x[:, 0], x[:, 1]

nodes = [
    Node(
        func=divide,
        inputs=np.arange(1, 7).reshape((3, 2)),
        outputs=("mean_a", "mean_b"),
        outputs_dumper=dump_dict_pickle,
        outputs_dumper_type=DumpType.BATCH,
        outputs_path="1.pickle",
        outputs_loader=load_dict_pickle,
    ),
    Node(
        func=lambda x, y: (x, y),
        outputs=("a", "b"),
        outputs_dumper=dump_dict_pickle,
        outputs_dumper_type=DumpType.BATCH,
        outputs_path="2.pickle",
        outputs_loader=load_dict_pickle,
    ),
    Node(
        func=lambda x, y: (x.max(), y.max()),
        inputs=("a", "b"),
        outputs=("c", "d"),
        outputs_dumper=dump_savez_compressed,
        outputs_dumper_type=DumpType.BATCH,
        outputs_path="3.npz",
        outputs_loader=load_savez_compressed,
    )
]

pipe = Pipeline(nodes)
outputs = pipe.run()
assert outputs == (5, 6)
```

you can pass a config object to a function:
```python
from lwpipe import Node, Pipeline
def add(a, cfg):
    return a + cfg["hyperparam"]

nodes = [Node(func=add, inputs=5, config={"hyperparam": 10})]
# equivalent to
# nodes = [Node(func=lambda a: add(a, {"hyperparam": 10}), inputs=5)]
pipe = Pipeline(nodes)
outputs = pipe.run()
assert outputs[0] == 15

def dumper(data, filepath, cfg):
    filepath = Path(filepath)
    filepath = filepath.with_name(filepath.name+cfg["hyperparam"])
    return dump_pickle(data, filepath)

# also, outputs_dumper can take config as its argument
nodes = [Node(func=add, inputs=5, config={"hyperparam": 10},
              outputs_dumper=dumper,
              outputs_dumper_take_config=True
)]
```

`Pipeline` also accepts a list of functions with no arguments and return-values:
```python
from lwpipe import Pipeline

def func():
    return

funcs = [func, func]
pipe = Pipeline(funcs)
pipe.run()
# equivalent to
# for func in funcs:
#   func()

# you can specify names of functions
pipe = Pipeline(funcs, names=["func1", "func2"])
pipe.run()
```

More examples are included in the [test cases](https://github.com/estshorter/lwpipe/blob/master/tests/test_pipeline.py).