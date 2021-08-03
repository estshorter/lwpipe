from __future__ import annotations

import io
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lwpipe import InputType, IOFuncType, Node, Pipeline
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


def mean(x: pd.DataFrame):
    return x.iloc[:, :4].mean()


def multiply(x: pd.DataFrame, n: int | float):
    return (x * n).to_numpy()


def add(x, n: int | float):
    return x + n


def divide(x: pd.DataFrame):
    return x.iloc[:10, :4].mean(), x.iloc[:20, :4].mean()


def divide_multiply(x: pd.DataFrame, y: pd.DataFrame, n: int | float):
    return (x * n).to_numpy(), (y * n).to_numpy()


def ten_times(x: pd.DataFrame, y: pd.DataFrame):
    return (x * 10).to_numpy(), (y * 10).to_numpy()


def sum(x):
    return x.sum()


@pytest.fixture
def df_simple():
    return pd.DataFrame(np.arange(12).reshape(3, 4))


@pytest.fixture
def iris():
    return pd.read_csv(Path(__file__).parent / "data/iris.csv")


def test_simple(df_simple):
    nodes = [
        Node(
            func=add,
            inputs=(df_simple, 1),
        ),
        Node(
            func=sum,
        ),
    ]

    pipe = Pipeline(nodes)
    outputs = pipe.run()
    outputs_np = outputs[0].tolist()
    assert outputs_np == [15, 18, 21, 24]


def test_output(iris, tmp_path):
    nodes = [
        Node(
            func=divide,
            inputs=iris,
            outputs=("mean1", "mean2"),
            outputs_dumper=dump_pickle,
            outputs_path=(
                tmp_path / "result1.pickle",
                tmp_path / "result2.pickle",
            ),
            outputs_loader=load_pickle,
        ),
        Node(
            func=ten_times,
        ),
    ]

    pipe = Pipeline(nodes)
    pipe.run()


def test_base(iris, tmp_path):
    nodes = [
        Node(
            func=mean,
            inputs=iris,
            outputs="mean",
            outputs_dumper=dump_pickle,
            outputs_path=tmp_path / "result1.pickle",
            outputs_loader=load_pickle,
        ),
        Node(
            func=lambda x: multiply(x, 10),
            name=multiply.__name__,
            outputs="ntimes",
            outputs_dumper=dump_npy,
            outputs_path=tmp_path / "result_multiply.pickle",
            outputs_loader=load_npy,
        ),
        Node(
            func=lambda x: add(x, 10),
            name=add.__name__,
            inputs="mean",
        ),
    ]

    pipe = Pipeline(nodes)
    pipe.run()
    pipe.run(1)
    pipe.run(2)


def test_tuple_output(iris, tmp_path):
    nodes = [
        Node(
            func=divide,
            inputs=iris,
            outputs=("mean1", "mean2"),
            outputs_dumper=dump_pickle,
            outputs_path=(
                tmp_path / "mean1.pickle",
                tmp_path / "mean2_pickle",
            ),
            outputs_loader=load_pickle,
        ),
        Node(
            func=ten_times,
            inputs=("mean1", "mean2"),
            outputs_dumper=dump_npy,
            outputs_path=(
                tmp_path / "mul1.npy",
                tmp_path / "mul2.npy",
            ),
        ),
    ]

    pipe = Pipeline(nodes)
    pipe.run()
    pipe.run("ten_times")


def test_in_out(iris, tmp_path):
    nodes = [
        Node(
            func=divide,
            inputs=iris,
            outputs=("mean1", "mean2"),
            outputs_dumper=dump_pickle,
            outputs_path=(
                tmp_path / "mean1",
                tmp_path / "mean2",
            ),
            outputs_loader=load_pickle,
        ),
        Node(
            func=divide_multiply,
            inputs=("mean1", "mean2", 10),
            inputs_type=[
                InputType.INTERIM_RESULT,
                InputType.INTERIM_RESULT,
                InputType.NON_INTERIM_RESULT,
            ],
        ),
    ]

    pipe = Pipeline(nodes)
    pipe.run()


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
    assert outputs[0] == 10000
    outputs = pipe.run(1)
    assert outputs[0] == 10000


def test_pd_things():
    csv = """A,B,C
    1,2,3
    4,5,6
    7,8,9
    """

    with io.StringIO(csv) as f:
        nodes = [
            Node(
                func=lambda df: df.mean(),
                inputs=pd.read_csv(f),
            )
        ]

        pipe = Pipeline(nodes)
        outputs = pipe.run()
        assert outputs[0].to_list() == [4.0, 5.0, 6.0]


def test_no_name_error():
    add_ = partial(add, n=1)
    with pytest.raises(ValueError):
        Node(func=add_, inputs=10)


def test_batch(iris, tmp_path):
    nodes = [
        Node(
            func=divide,
            inputs=iris,
            outputs=("mean_a", "mean_b"),
            outputs_dumper=dump_dict_pickle,
            outputs_dumper_type=IOFuncType.BATCH,
            outputs_path=tmp_path / "1.pickle",
            outputs_loader=load_dict_pickle,
        ),
        Node(
            func=lambda x, y: (x, y),
            name="2",
            outputs=("a", "b"),
            outputs_dumper=dump_dict_pickle,
            outputs_dumper_type=IOFuncType.BATCH,
            outputs_path=tmp_path / "10.npz",
            outputs_loader=load_dict_pickle,
        ),
        Node(
            func=ten_times,
            outputs=("c", "d"),
            inputs=("a", "b"),
            outputs_dumper=dump_savez_compressed,
            outputs_dumper_type=IOFuncType.BATCH,
            outputs_path=tmp_path / "3.npz",
            outputs_loader=load_savez_compressed,
        ),
        Node(
            func=lambda x, y: (x, y),
        ),
    ]

    pipe = Pipeline(nodes)
    pipe.run()
    pipe.run(1)
    pipe.run(2)
    pipe.run(3)
