import pytest
from lwpipe import DumpType, Node
from lwpipe.io import dump_dict_pickle, load_dict_pickle


def test_batch_dumper_not_callable():
    with pytest.raises(ValueError):
        Node(
            func=lambda x: x,
            inputs=1,
            outputs_dumper=[dump_dict_pickle],
            outputs_dumper_type=DumpType.BATCH,
        )


def test_batch_loader_not_callable():
    with pytest.raises(ValueError):
        Node(
            func=lambda x: x,
            inputs=1,
            outputs_dumper=dump_dict_pickle,
            outputs_dumper_type=DumpType.BATCH,
            outputs_loader=[load_dict_pickle],
        )


def test_batch_path_list_tuple():
    with pytest.raises(ValueError):
        Node(
            func=lambda x: x,
            inputs=1,
            outputs_dumper=dump_dict_pickle,
            outputs_dumper_type=DumpType.BATCH,
            outputs_loader=load_dict_pickle,
            outputs_path=["hoge"],
        )
        Node(
            func=lambda x: x,
            inputs=1,
            outputs_dumper=dump_dict_pickle,
            outputs_dumper_type=DumpType.BATCH,
            outputs_loader=load_dict_pickle,
            outputs_path=("hoge"),
        )
