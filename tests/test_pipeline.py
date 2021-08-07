from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from lwpipe import DumpType, Node, Pipeline
from lwpipe.io import (
    dump_dict_pickle,
    dump_npy,
    dump_pickle,
    dump_savez_compressed,
    load_dict_pickle,
    load_npy,
    load_pickle,
    load_savez_compressed,
)


def divide(x):
    mid = x.shape[1] // 2
    return x[:, :mid], x[:, mid:]


def ten_times_two_inputs(x, y):
    return x * 10, y * 10


@pytest.fixture
def np_array_2d():
    return np.arange(50).reshape((25, 2))


def test_simple(np_array_2d):
    nodes = [
        Node(
            func=np.add,
            inputs=(np_array_2d, 1),
        ),
        Node(
            func=np.sum,
        ),
    ]

    pipe = Pipeline(nodes)
    outputs = pipe.run()
    outputs_np = outputs[0].tolist()
    assert outputs_np == 1275


def test_output(np_array_2d, tmp_path):
    result1 = tmp_path / "result1.pickle"
    result2 = tmp_path / "result2.pickle"
    nodes = [
        Node(
            func=divide,
            inputs=np_array_2d,
            outputs=("mean1", "mean2"),
            outputs_dumper=dump_pickle,
            outputs_path=(
                result1,
                result2,
            ),
            outputs_loader=load_pickle,
        ),
        Node(
            func=lambda x, y: (x.mean(), y.mean()),
        ),
    ]

    pipe = Pipeline(nodes)
    outputs = pipe.run()
    pipe.clear()
    assert outputs == (24, 25)
    assert result1.exists()
    assert result2.exists()

    outputs = pipe.run(1)
    assert outputs == (24, 25)


def test_ensure_read(np_array_2d, tmp_path):
    result1 = tmp_path / "result1.pickle"
    result2 = tmp_path / "result2.pickle"
    nodes = [
        Node(
            func=divide,
            inputs=np_array_2d,
            outputs=("mean1", "mean2"),
            outputs_dumper=dump_pickle,
            outputs_path=(
                result1,
                result2,
            ),
            outputs_loader=load_pickle,
        ),
        Node(
            func=lambda x, y: (x.mean(), y.mean()),
        ),
    ]

    pipe = Pipeline(nodes)
    outputs = pipe.run()
    pipe.clear()

    assert outputs == (24, 25)
    os.remove(result1)
    with pytest.raises(FileNotFoundError):
        pipe.run(1)
    pipe.clear()
    pipe.run()
    pipe.clear()
    os.remove(result2)
    with pytest.raises(FileNotFoundError):
        pipe.run(1)


def test_base(np_array_2d, tmp_path):
    result1 = tmp_path / "result1.pickle"
    result2 = tmp_path / "result2.npy"

    nodes = [
        Node(
            func=np.mean,
            inputs=np_array_2d,
            outputs="mean",
            outputs_dumper=dump_pickle,
            outputs_path=result1,
            outputs_loader=load_pickle,
        ),
        Node(
            func=lambda x: np.multiply(x, 10),
            name=np.multiply,
            outputs="ntimes",
            outputs_dumper=dump_npy,
            outputs_path=result2,
            outputs_loader=load_npy,
        ),
        Node(
            func=lambda x: x + 10,
            inputs="mean",
        ),
    ]

    pipe = Pipeline(nodes)
    outputs1 = pipe.run()
    pipe.clear()
    assert result1.exists()
    assert result2.exists()
    outputs2 = pipe.run(1)
    pipe.clear()
    outputs3 = pipe.run(2)
    assert outputs1 == outputs2
    assert outputs2 == outputs3


def test_tuple_output(np_array_2d, tmp_path):
    mean1 = tmp_path / "divide1.pickle"
    mean2 = tmp_path / "divide2.pickle"
    mul1 = tmp_path / "mul1.npy"
    mul2 = tmp_path / "mul2.npy"

    nodes = [
        Node(
            func=divide,
            inputs=np_array_2d,
            outputs=("divide1", "divide2"),
            outputs_dumper=dump_pickle,
            outputs_path=(
                mean1,
                mean2,
            ),
            outputs_loader=load_pickle,
        ),
        Node(
            func=ten_times_two_inputs,
            inputs=("divide1", "divide2"),
            outputs_dumper=dump_npy,
            outputs_path=(
                mul1,
                mul2,
            ),
        ),
    ]

    pipe = Pipeline(nodes)
    outputs = pipe.run()
    pipe.clear()
    outputs2 = pipe.run(nodes[1].func.__name__)
    assert np.all(outputs[0] == outputs2[0])
    assert np.all(outputs[1] == outputs2[1])


def test_no_input_at_initial_node():
    nodes = [
        Node(func=lambda: 100),
        Node(
            func=lambda x: 10 * x,
            name="multiply",
        ),
    ]

    pipe = Pipeline(nodes)
    outputs, *_ = pipe.run()
    assert outputs == 1000


def test_none_outputs():
    nodes = [
        Node(func=lambda: 100, outputs=[None]),
        Node(
            func=lambda x: 10 * x,
            name="multiply",
            inputs=[None],
        ),
    ]

    pipe = Pipeline(nodes)
    with pytest.raises(KeyError):
        pipe.run()


def test_resume_duplicate_name():
    nodes = [
        Node(
            func=lambda x: x,
            inputs=100,
            outputs_dumper=dump_pickle,
            outputs_path=".result1.pickle",
            outputs_loader=load_pickle,
        ),
        Node(func=lambda x: x ** 2),
    ]

    pipe = Pipeline(nodes)
    outputs = pipe.run()
    pipe.clear()
    assert outputs[0] == 10000
    outputs = pipe.run(1)
    assert outputs[0] == 10000


def test_duplicate_name():
    add_ = partial(np.add, 1)
    pipe = Pipeline([Node(func=add_, inputs=10), Node(func=add_)])
    outputs = pipe.run()
    assert outputs[0] == 12
    assert pipe.nodes[0].name == "anonymous"
    assert pipe.nodes[1].name == "anonymous__2__"


def test_batch(np_array_2d, tmp_path):
    result1 = tmp_path / "1.pickle"
    result2 = tmp_path / "2.pickle"
    result3 = tmp_path / "3.npz"
    nodes = [
        Node(
            func=divide,
            inputs=np_array_2d,
            outputs=("mean_a", "mean_b"),
            outputs_dumper=dump_dict_pickle,
            outputs_dumper_type=DumpType.BATCH,
            outputs_path=result1,
            outputs_loader=load_dict_pickle,
        ),
        Node(
            func=lambda x, y: (x, y),
            outputs=("a", "b"),
            outputs_dumper=dump_dict_pickle,
            outputs_dumper_type=DumpType.BATCH,
            outputs_path=result2,
            outputs_loader=load_dict_pickle,
        ),
        Node(
            func=ten_times_two_inputs,
            outputs=("c", "d"),
            outputs_dumper=dump_savez_compressed,
            outputs_dumper_type=DumpType.BATCH,
            outputs_path=result3,
            outputs_loader=load_savez_compressed,
        ),
        Node(
            func=lambda x, y: (x, y),
        ),
    ]

    pipe = Pipeline(nodes)
    pipe.run()
    pipe.clear()
    pipe.run(1)
    pipe.clear()
    os.remove(result1)
    pipe.run(2)
    pipe.clear()
    os.remove(result2)
    pipe.run(3)


def test_get_node_names():
    add_ = partial(np.add, 1)
    pipe = Pipeline([Node(func=add_, inputs=10), Node(func=add_)])
    names = pipe.get_node_names()
    assert names == ["anonymous", "anonymous__2__"]


def test_config():
    def add(a, cfg):
        return a + cfg["hyperparam"]

    nodes = [Node(func=add, inputs=5, config={"hyperparam": 10})]
    # equivalent to
    # nodes = [Node(func=lambda a: add(a, {"hyperparam": 10}), inputs=5)]
    pipe = Pipeline(nodes)
    outputs = pipe.run()
    assert outputs[0] == 15


def test_dumper_config(tmp_path):
    def add(a, cfg):
        return a

    def dumper(data, filepath, cfg):
        filepath = Path(filepath)
        filepath = filepath.with_name(cfg["suffix"])
        return dump_pickle(data, filepath)

    def loader(filepath, cfg):
        filepath = Path(filepath)
        filepath = filepath.with_name(cfg["suffix"])
        return load_pickle(filepath)

    def dumper_batch(data, filepath, datalabels, cfg):
        filepath = Path(filepath)
        filepath = filepath.with_name(cfg["suffix"])
        return dump_dict_pickle(data, filepath, datalabels)

    def loader_batch(filepath, datalabels, cfg):
        filepath = Path(filepath)
        filepath = filepath.with_name(cfg["suffix"])
        return load_dict_pickle(filepath, datalabels)

    result1 = tmp_path / "result1.pickle"
    result2 = tmp_path / "result2.pickle"
    nodes = [
        Node(
            func=add,
            inputs=1,
            config={"suffix": "_TEST_"},
            outputs_dumper=dumper,
            outputs_loader=loader,
            outputs_path=result1,
            outputs_dumper_take_config=True,
        ),
        Node(
            func=lambda a, cfg: a,
            outputs_dumper=dumper_batch,
            outputs_dumper_take_config=True,
            config={"suffix": "_TEST_"},
            outputs="hoge",
            outputs_dumper_type=DumpType.BATCH,
            outputs_path=result2,
            outputs_loader=loader_batch,
        ),
        Node(lambda a: a),
    ]
    pipe = Pipeline(nodes)

    pipe.run()
    pipe.clear()
    assert result1.with_name("_TEST_").exists()
    assert result2.with_name("_TEST_").exists()
    pipe.run(1)
    pipe.clear()
    pipe.run(2)


def test_kidou(tmp_path):
    result1 = tmp_path / "result1.pickle"

    nodes = [
        Node(
            func=np.add,
            inputs=(1, 2),
            outputs_dumper=dump_pickle,
            outputs_path=result1,
            outputs_loader=load_pickle,
        ),
        Node(func=lambda x: x * 10),
        Node(func=lambda x: x * 10),
    ]
    pipe = Pipeline(nodes)

    pipe.run()
    pipe.clear()
    pipe.run(1)
    pipe.clear()
    with pytest.raises(ValueError):
        pipe.run(2)


def test_inputs_not_found():
    nodes = [
        Node(
            func=np.add,
            inputs=(1, 2),
        ),
        Node(func=lambda x: x * 10, inputs="hoge"),
    ]

    with pytest.raises(ValueError):
        Pipeline(nodes)


def test_to_from(tmp_path):
    def no_op(a):
        return a

    pipe = Pipeline(
        [
            Node(
                func=no_op,
                inputs=1,
                outputs_dumper=dump_pickle,
                outputs_path=tmp_path / "result1.pickle",
                outputs_loader=load_pickle,
            ),
            Node(func=no_op),
            Node(func=no_op),
        ]
    )
    pipe.run()
    pipe.clear()
    pipe.run(1, 1)
    pipe.clear()
    pipe.run(0, 1)
    with pytest.raises(ValueError):
        pipe.run(0, 3)
    pipe.clear()
    with pytest.raises(ValueError):
        pipe.run(1, 0)
    pipe.clear()
    with pytest.raises(ValueError):
        pipe.run(4, 5)


def test_no_return_value():
    def no_op():
        pass

    def no_op_cfg(cfg):
        pass

    pipe = Pipeline([Node(no_op), Node(no_op_cfg, config={"hoge": 10}), Node(no_op)])
    outputs = pipe.run()
    assert outputs is None
    pipe = Pipeline([Node(no_op), Node(no_op), Node(no_op)])
    outputs = pipe.run()
    assert outputs is None


def no_op():
    return


def test_trivial_pipeline_basic():
    funcs = [no_op, no_op]
    pipe = Pipeline(funcs)
    pipe.run()
    pipe.clear()
    pipe.run(1, 1)
    pipe.clear()
    pipe.run(0, 1)
    pipe.clear()
    with pytest.raises(ValueError):
        pipe.run(0, 2)
    pipe.clear()
    with pytest.raises(ValueError):
        pipe.run(1, 0)
    pipe.clear()
    with pytest.raises(ValueError):
        pipe.run(2, 2)
    pipe.clear()


def test_trivial_pipeline():
    def no_op2():
        return

    funcs = [no_op, no_op, no_op2]
    pipe = Pipeline(funcs)
    assert pipe.get_node_names() == ["no_op", "no_op__2__", "no_op2"]


def test_name_uniqueness():
    funcs = [no_op, no_op]
    with pytest.raises(ValueError):
        Pipeline(funcs, names=["a1", "a1"])


def test_string_from_to():
    funcs = [no_op, no_op, no_op]
    pipe = Pipeline(funcs, names=["func1", "func2", "func3"])
    pipe.run("func1", "func2")
    pipe.clear()
    pipe.run("func2", "func3")
    pipe.clear()
    pipe.run("func1", "func3")
    pipe.clear()
    pipe.run("func3", "func3")
    pipe.clear()
    with pytest.raises(ValueError):
        pipe.run("func3", "func1")
    pipe.clear()
    with pytest.raises(ValueError):
        pipe.run("func2", "func1")


def test_previous_result_to_results_dict(tmp_path):
    nodes = [
        Node(
            func=lambda x: x + "_proc1",
            inputs="INPUT",
            outputs="proc1",
            outputs_dumper=dump_pickle,
            outputs_path=tmp_path / "proc1.pickle",
            outputs_loader=load_pickle,
        ),
        Node(
            func=lambda x: x + "_proc2",
            outputs="proc2",
            name="proc2",
            outputs_dumper=dump_pickle,
            outputs_path=tmp_path / "proc2.pickle",
        ),
        Node(func=lambda x: x + "_proc3", inputs="proc1"),
    ]
    pipe = Pipeline(nodes)
    outputs = pipe.run(0)
    pipe.clear()
    outputs = pipe.run(1)
    assert outputs[0] == "INPUT_proc1_proc3"
    assert pipe.results["proc2"] == "INPUT_proc1_proc2"


def test_previous_result_to_results_dict_batch():
    nodes = [
        Node(
            func=lambda x: x + "_proc1",
            inputs="INPUT",
            outputs="proc1",
            outputs_dumper=dump_dict_pickle,
            outputs_dumper_type=DumpType.BATCH,
            outputs_path="proc1.pickle",
            outputs_loader=load_dict_pickle,
        ),
        Node(
            func=lambda x: x + "_proc2",
            outputs="proc2",
            name="proc2",
            outputs_dumper=dump_dict_pickle,
            outputs_dumper_type=DumpType.BATCH,
            outputs_path="proc2.pickle",
        ),
        Node(func=lambda x: x + "_proc3", inputs="proc1"),
    ]
    pipe = Pipeline(nodes)
    pipe.run()
    pipe.clear()
    outputs = pipe.run(1)
    assert outputs[0] == "INPUT_proc1_proc3"
    assert pipe.results["proc2"] == "INPUT_proc1_proc2"
