# Overview
**lwpipe** provides a lightweight pipeline for numerical experiments.
For example, this module can be used in preprocessing steps of machine learning. Preprocessing consists of several steps, some of which take time to execute. In this case, it is common in the trial-and-error stage, such as numerical experiments, to dump the calculation results of the computationally-intensive steps and load them in the later programs to reduce the time required when the later steps are changed. This module reduces boilerplate code for file IO in the use cases above.

Note that pipelines in this module do not have the concept of dependency between nodes (tasks), and nodes are executed sequentially.

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
        func=lambda x: x.mean(),
        name="former_mean",
        inputs="former", # calculated at the first node
        outputs="former_mean",
    ),
    Node(
        func=lambda x: x.mean(),
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

batch dump example:
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

More examples are included in the [test cases](https://github.com/estshorter/lwpipe/blob/master/tests/test_basic.py).