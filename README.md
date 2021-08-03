# Overview
**expipe** provides a pipeline for numerical EXperiments.
For example, this module can be used in preprocessing steps of machine learning. Preprocessing consists of several steps, some of which take time to execute. In this case, it is common in the trial-and-error stage, such as numerical experiments, to save the calculation results of the computationally-intensive steps and load them in the later programs to reduce the time required when the later steps are changed. This module reduces boilerplate code for file IO in the use cases above.

Note that pipelines in this module do not have the concept of dependency between nodes (tasks), and nodes will be executed sequentially.

# Installation
from pypi: 
``` sh
pip install expipe
```

# Usage
Minimal example (of course, no need to use this library..):
``` python
from expipe import Node, Pipeline

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
from expipe import Node, Pipeline
from expipe.io import dump_pickle, load_pickle

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
    Node(func=lambda x: x**2),
]

pipe = Pipeline(nodes)
outputs = pipe.run()
assert outputs[0] == 10000
```
Once the first node is executed, you can bypass the node by `pipe.run(1)` or `pipe.run("square")`.

Multiple outputs with numpy:
``` python
import numpy as np
from expipe import InputType, Node, Pipeline
from expipe.io import dump_npy, load_npy

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
        inputs_type=InputType.INTERIM_RESULT,
        outputs="former_mean",
    ),
    Node(
        func=lambda x: x.mean(),
        name="latter_mean",
        inputs="latter", # calculated at the first node
        inputs_type=InputType.INTERIM_RESULT,
        outputs="latter_mean",
    ),
]

pipe = Pipeline(nodes)
outputs = pipe.run()
assert outputs[0] == 7.0
# You can access interim results by "results" dict
assert pipe.results["former_mean"] == 2.0
```

Example with pandas:
``` python
import io
from expipe import Node, Pipeline
from expipe.io import load_csv_as_dataframe

csv = """A,B,C
1,2,3
4,5,6
7,8,9
"""

with io.StringIO(csv) as f:
    nodes = [
        Node(
            func=lambda df: df.mean(),
            inputs=f,
            inputs_loader=load_csv_as_dataframe,
        )
    ]

    pipe = Pipeline(nodes)
    outputs = pipe.run()
    assert outputs[0].to_list() == [4.0, 5.0, 6.0]
```
