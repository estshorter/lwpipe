from __future__ import annotations

import logging
from enum import IntEnum, auto
from typing import Callable, Optional

from .utils import _assert_same_length

logger = logging.getLogger(__name__)


__version__ = "2.0.0"


class DumpType(IntEnum):
    INDIVIDUAL = auto()
    BATCH = auto()


class Node:
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        inputs: str | list[str] | None = None,
        outputs: str | list[str] | None = None,
        outputs_dumper: Callable | list[Callable] | None = None,
        outputs_dumper_type: DumpType = DumpType.INDIVIDUAL,
        outputs_path: Optional[list[str]] = None,
        outputs_loader: Callable | list[Callable] | None = None,
    ) -> None:
        """
        Parameters
        ------------------
        func: 適用する関数。
        name: 関数の名前。Noneのときはfunc.__name__が代入される。__name__がなければ"anonymous"。
        inputs: 入力データ。最初のノードに対しNoneを設定すると、引数0個の関数をfuncにセットできる。
                それ以外のノードでNoneを設定した場合は、前段の出力を入力として使うという設定になる。
                文字列が渡されているときは、dict型のPipiline.resultsからその名前の中間結果を読もうとする。
        outputs: 出力結果をdictに入れる際のキー。Noneにすると保存されず、次のノードに渡されるのみとなる。
        outputs_dumper: outputsをdumpする関数。リストを渡せば、各変数に対して別々の関数を適用可能。
                        引数は(data, filepath: str | PurePath) を想定。
                        outputs_dumper_typeがBATCHの際は、引数にデータのラベルを表すlist[str]が追加され、
                        (datalist, filepath: str | PurePath, datalabels)
                        となる。
        outputs_dumper_type: 複数の出力データを一つのファイルに保存したいときはDumpType.BATCHを設定する。
        outputs_path: dumpするファイルパス。
        outputs_loader: dumpしたoutputsをloadするための関数。
                        pipelineを途中から実行する場合、中間結果をloadする必要があるが、その際にコールされる。
        """
        self.func = func
        if name is not None:
            self.name = name
        else:
            if hasattr(func, "__name__"):
                self.name = func.__name__
            else:
                self.name = "anonymous"

        self.inputs = _convert_item_to_list(inputs)
        self.outputs = _convert_item_to_list(outputs)

        self.outputs_dumper = outputs_dumper
        self.outputs_dumper_type = outputs_dumper_type
        if outputs_dumper_type == DumpType.BATCH:
            if not callable(outputs_dumper):
                raise ValueError(
                    "For batch loader/dumper, outputs_dumper_type must be callable"
                )
            if (not callable(outputs_loader)) and outputs_loader is not None:
                raise ValueError(
                    "For batch loader/dumper, outputs_loader_type must be callable"
                )
            if isinstance(outputs_path, list) or isinstance(outputs_path, tuple):
                raise ValueError(
                    "For batch loader/dumper, outputs_path must be path-like object"
                )
        self.outputs_path = _convert_item_to_list(outputs_path)
        self.outputs_loader = outputs_loader
        self.outputs_loader_type = outputs_dumper_type


class Pipeline:
    def __init__(self, nodes: list[Node]) -> None:
        self.nodes = nodes
        self.results = dict()

        # node.name -> index in self.nodes
        self.name_to_idx = dict()
        # outputs -> (index in self.nodes, pos in node.outputs)
        self.outputs_to_indexes = dict()

        _assert_non_zero_length(nodes, "nodes")

        name_duplicate_counter = {}

        for idx, node in enumerate(self.nodes):
            if node.name in self.name_to_idx:
                name_duplicate_counter[node.name] += 1
                node.name += f"_{name_duplicate_counter[node.name]}"
            else:
                name_duplicate_counter[node.name] = 1
            self.name_to_idx[node.name] = idx
            if node.outputs is None:
                continue
            for idx_outputs, output in enumerate(node.outputs):
                if output in self.outputs_to_indexes:
                    raise ValueError(f"node.output {output} is not unique")
                self.outputs_to_indexes[output] = (idx, idx_outputs)

    def get_node_names(self) -> list[str]:
        return [node.name for node in self.nodes]

    def run(self, start: int | str = 0):
        """pipelineを実行する。戻り値はlist。
        Parameters
        ----------------
        start: どのノードからパイプラインを開始するか。インデックスかnameで指定可能。
        """
        idx_start = self._get_start_index(start)
        logger.info(
            f"Total {len(self.nodes)} tasks, scheduled {len(self.nodes[idx_start:])} tasks"
        )
        outputs = self._load_interim_output(idx_start)
        for idx, node in enumerate(self.nodes[idx_start:]):
            logger.info(
                f"Running {idx+1}/{len(self.nodes[idx_start:])} tasks ({node.name})"
            )

            # 最初のノードだけ特別扱い
            if idx == 0 and idx_start == 0:
                if node.inputs is not None:
                    outputs = node.func(*node.inputs)
                else:
                    outputs = node.func()
            else:
                inputs = self._get_inputs(node, outputs)
                outputs = node.func(*inputs)

            outputs = _convert_item_to_list(outputs)
            _assert_non_zero_length(outputs, "outputs")
            self._insert_outputs_to_dict(node, outputs)
            self._dump_outputs(node, outputs)

        logger.info("All tasks completed")
        return outputs

    def _get_start_index(self, start: int | str):
        if isinstance(start, int):
            idx_start = start
        elif isinstance(start, str):
            idx_start = self.name_to_idx[start]

        if idx_start < 0 or idx_start >= len(self.nodes):
            raise ValueError(
                f"Wrong argument: idx_start ({idx_start}) must satisfy 0 <= start < {len(self.node)-1}"
            )

        return idx_start

    def _load_interim_output(self, idx_start):
        """
        idx_startから最後のノードまでのノードに対し、
        idx_startより前のノードの結果を使うかチェックし、
        もし使うのであれば必要な結果をロードする。
        """
        if idx_start == 0:
            return None
        last_output = None
        loaded_files_set = set()
        # idx_startから読み込むこと
        # さもないと直前のノードの入力が2回読まれる可能性がある
        for idx_ in range(idx_start, len(self.nodes)):
            # 入力が指定されていない場合（前段の出力を入力とする場合）
            if self.nodes[idx_].inputs is None:
                # idx_start以降のタスクはまだ計算していないのでcontinue
                if idx_ > idx_start:
                    continue
                last_output = self._load_last_output(
                    self.nodes[idx_ - 1], loaded_files_set
                )
            else:  # 前段より前の出力を入力にする場合
                self._load_past_output(idx_, idx_start, loaded_files_set)
        return last_output

    def _load_last_output(self, node_prev, loaded_files_set):
        # 本関数はrange(idx_start, len(self.nodes)): のループの最初に実行されるので、
        # outputs_loaderの呼び出しが冗長ではない
        if node_prev.outputs_loader_type == DumpType.BATCH:
            loaded_files_set.add(*node_prev.outputs_path)
            return node_prev.outputs_loader(*node_prev.outputs_path, node_prev.outputs)

        _assert_non_zero_length(node_prev.outputs_path, "node_prev.outputs_path")

        outputs_loaders = _convert_item_to_list(
            node_prev.outputs_loader, len(node_prev.outputs_path)
        )
        _assert_same_length(
            node_prev.outputs_path,
            outputs_loaders,
            "node_prev.outputs_path",
            "outputs_loaders",
        )

        outputs = []
        for output_path, outputs_loader in zip(node_prev.outputs_path, outputs_loaders):
            loaded_files_set.add(output_path)
            outputs.append(outputs_loader(output_path))
        return outputs

    def _load_past_output(self, idx, idx_start, loaded_files_set):
        for input in self.nodes[idx].inputs:
            idx_dependant_node, idx_in_outputs = self.outputs_to_indexes[input]
            if idx_dependant_node >= idx_start:
                # まだ計算していないのでcontinue
                continue

            node_dep = self.nodes[idx_dependant_node]

            if node_dep.outputs_loader_type == DumpType.BATCH:
                if node_dep.outputs_path[0] in loaded_files_set:
                    continue
                loaded_files_set.add(*node_dep.outputs_path)
                outputs = node_dep.outputs_loader(
                    *node_dep.outputs_path, node_dep.outputs
                )
                for output_key, output in zip(node_dep.outputs, outputs):
                    self.results[output_key] = output
                continue

            outputs_loaders = _convert_item_to_list(
                node_dep.outputs_loader, len(node_dep.outputs)
            )
            _assert_same_length(
                node_dep.outputs,
                outputs_loaders,
                "node_prev.outputs",
                "outputs_loaders",
            )
            idx = idx_in_outputs
            if node_dep.outputs_path[idx] in loaded_files_set:
                return
            loaded_files_set.add(node_dep.outputs_path[idx])
            self.results[input] = outputs_loaders[idx](node_dep.outputs_path[idx])

    def _get_inputs(self, node, previous_outputs):
        """前段までの入力を基に、次のNodeへの入力を決める"""
        # 入力がない場合は前段の結果を持ってくる
        if node.inputs is None:
            if previous_outputs is None:
                raise ValueError("inputs is None")
            return previous_outputs

        inputs = []
        _assert_non_zero_length(node.inputs, "node.inputs")

        for input in node.inputs:
            inputs.append(self.results[input])
        return inputs

    def _insert_outputs_to_dict(self, node, outputs):
        if node.outputs is None:
            return
        _assert_same_length(node.outputs, outputs, "node.outputs", "outputs")
        for key, output in zip(node.outputs, outputs):
            # nodeのoutputsにNoneが入っていたらdictには入れない
            if key is not None:
                self.results[key] = output

    def _dump_outputs(self, node, outputs):
        if node.outputs_dumper is None:
            return

        if node.outputs_dumper_type == DumpType.BATCH:
            return node.outputs_dumper(outputs, *node.outputs_path, node.outputs)

        _assert_same_length(outputs, node.outputs_path, "outputs", "node.outputs_path")
        outputs_dumpers = _convert_item_to_list(node.outputs_dumper, len(outputs))
        _assert_same_length(outputs, outputs_dumpers, "outputs", "outputs_dumper")
        for output, filepath, outputs_dumper in zip(
            outputs, node.outputs_path, outputs_dumpers
        ):
            if outputs_dumper is not None:
                outputs_dumper(output, filepath)


def _assert_non_zero_length(x, x_str):
    if len(x) == 0:
        raise ValueError(f"size of {x_str} is zero")


def _convert_item_to_list(data, length=1) -> list | tuple | None:
    """dataがlistでもtupleでもなければ、同じ要素が詰まった長さlengthの配列にして返す。"""
    if isinstance(data, list) or isinstance(data, tuple) or data is None:
        return data
    else:
        return [data] * length
