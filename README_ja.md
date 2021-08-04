# Overview
lwpipeは数値実験用のパイプラインを提供するモジュールであり、
例えば、機械学習における前処理手順に適用可能です。
前処理は複数の手順から構成されており、
そのうちのいくつかは実行に時間を要します。
この場合、重い処理の計算結果を保存しておき、後段のプログラムではそれを読み込むことで、後段の処理が変更になった時の手間を減らすテクニックが、数値実験等の試行錯誤の段階においては一般的です。本モジュールにより、そのようなユースケースにおける、ファイルIOのボイラープレートコードを減らせます。

なお、本モジュールのパイプラインにはノード（タスク）間の依存性の概念はなく、最初のノードから順次実行されます。


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

More examples are included in the [test cases](https://github.com/estshorter/lwpipe/blob/master/tests/test_basic.py).