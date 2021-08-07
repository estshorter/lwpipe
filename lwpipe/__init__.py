from __future__ import annotations

import logging
from enum import IntEnum, auto
from inspect import signature
from typing import Callable, Optional

from .utils import _assert_same_length

logger = logging.getLogger(__name__)


__version__ = "5.0.0"


class DumpType(IntEnum):
    INDIVIDUAL = auto()
    BATCH = auto()


class Node:
    def __init__(
        self,
        func: Callable,
        inputs: str | list[str] | None = None,
        outputs: str | list[str] | None = None,
        name: Optional[str] = None,
        config: dict = None,
        outputs_dumper: Callable | list[Callable] | None = None,
        outputs_dumper_type: DumpType = DumpType.INDIVIDUAL,
        outputs_dumper_take_config: bool = False,
        outputs_path: Optional[list[str]] = None,
        outputs_loader: Callable | list[Callable] | None = None,
    ) -> None:
        """
        Parameters
        ------------------
        func: 適用する関数。
        inputs: 入力データ。パイプライン中の先頭ノードに対しNoneを設定すると、引数0個の関数をfuncにセットできる。
                先頭ノード以外のものにNoneを設定した場合は、前段の出力を入力として使うという設定になる。
                文字列が渡されているときは、dict型のPipiline.resultsからその名前の中間結果を読もうとする。
        outputs: 出力結果をdict型のPipiline.resultsに入れる際のキー。Noneにすると保存されず、次のノードに渡されるのみとなる。
        name: 関数の名前。Noneのときはfunc.__name__が代入される。__name__がなければ"anonymous"。
        config: funcに与えるconfig、Noneでなければfuncの引数の最後にこれが加わる
        outputs_dumper: outputsをdumpする関数。リストを渡せば、各変数に対して別々の関数を適用可能。
                        引数は(data, filepath: str | PurePath) を想定。
                        outputs_dumper_typeがBATCHの際は、引数にデータのラベルを表すlist[str]が追加され、
                        (datalist, filepath: str | PurePath, datalabels)
                        となる。
        outputs_dumper_type: 複数の出力データを一つのファイルに保存したいときはDumpType.BATCHを設定する。
        outputs_dumper_take_config: outputs_dumperがconfigを最後の引数にとるならTrueに設定する。
        outputs_path: dumpするファイルパス。
        outputs_loader: dumpしたoutputsをloadするための関数。
                        pipelineを途中から実行する場合、中間結果をloadする必要があるが、その際にコールされる。
        """
        self.func = func
        if name is not None:
            self.name = name
        elif hasattr(func, "__name__"):
            self.name = func.__name__
        else:
            self.name = "anonymous"

        self.inputs = _convert_item_to_list(inputs)
        self.outputs = _convert_item_to_list(outputs)

        self.config = config

        self.outputs_dumper = outputs_dumper
        self.outputs_dumper_type = outputs_dumper_type
        self.outputs_dumper_take_config = outputs_dumper_take_config
        if outputs_dumper_type == DumpType.BATCH:
            if not callable(outputs_dumper):
                raise ValueError(
                    "For batch loader/dumper, outputs_dumper must be callable"
                )
            if (not callable(outputs_loader)) and outputs_loader is not None:
                raise ValueError(
                    "For batch loader/dumper, outputs_loader must be callable"
                )
            if isinstance(outputs_path, list) or isinstance(outputs_path, tuple):
                raise ValueError(
                    "For batch loader/dumper, outputs_path must be path-like object"
                )
        self.outputs_path = _convert_item_to_list(outputs_path)
        self.outputs_loader = outputs_loader
        self.outputs_loader_type = outputs_dumper_type


class Pipeline:
    def __init__(
        self, nodes: list[Node] | list[Callable], names: list[str] = None
    ) -> None:
        """
        Parameters
        --------------
        nodes: 実行するノードのリスト。あるいは関数のリスト。
        names: ノードの名前を与える文字列のリスト。nodesがlist[Callable]の場合のみ有効。
        """
        if isinstance(nodes[0], Node):
            if len(nodes) > 1:
                for node in nodes[1:]:
                    if not isinstance(node, Node):
                        raise ValueError(f"node {node} has wrong type {type(node)}")
            self.nodes = nodes
        elif callable(nodes[0]):  # すべてCallableだったらNodeに変換
            if len(nodes) > 1:
                for node in nodes[1:]:
                    if not callable(node):
                        raise ValueError(f"node {node} has wrong type {type(node)}")
            self.nodes = self.convert_callables_to_nodes(nodes, names)
        else:
            raise ValueError(f"node {nodes[0]} has wrong type {type(nodes[0])}")

        self.results = dict()

        # node.name -> index in self.nodes
        self.name_to_idx = dict()
        # outputs -> (index in self.nodes, index in node.outputs)
        self.outputs_to_indexes = dict()

        _assert_non_zero_length(nodes, "nodes")

        name_duplicate_counter = {}

        for idx, node in enumerate(self.nodes):
            if node.name not in self.name_to_idx:
                name_duplicate_counter[node.name] = 1
            else:
                name_duplicate_counter[node.name] += 1
                node.name += f"__{name_duplicate_counter[node.name]}__"
                if node.name not in self.name_to_idx:
                    name_duplicate_counter[node.name] = 1
                else:
                    raise ValueError(
                        f"name: {node.name} is duplicated. Consider change name"
                    )
            self.name_to_idx[node.name] = idx
            if node.outputs is None:
                continue
            for idx_outputs, output in enumerate(node.outputs):
                if output in self.outputs_to_indexes:
                    raise ValueError(f"node.output {output} is not unique")
                self.outputs_to_indexes[output] = (idx, idx_outputs)

        for node in self.nodes[1:]:
            if node.inputs is None:
                continue
            for input in node.inputs:
                if input not in self.outputs_to_indexes:
                    raise ValueError(
                        f"inputs: {input} will not be calculated in this pipeline"
                    )

    def convert_callables_to_nodes(self, funcs, names):
        if names is not None:
            _assert_same_length(funcs, names, "funcs", "names")
            if len(names) != len(set(names)):
                raise ValueError(f"names is not unique: {names}")
        else:
            names = []
            for func in funcs:
                if hasattr(func, "__name__"):
                    name = func.__name__
                else:
                    name = "anonymous"
                names.append(name)
        return [Node(func, name=name) for func, name in zip(funcs, names)]

    def get_node_names(self) -> list[str]:
        return [node.name for node in self.nodes]

    def run(self, from_: int | str = 0, to_: int | str | None = None):
        """pipelineを実行する。戻り値はlist。
        Parameters
        ----------------
        start: どのノードからパイプラインを開始するか。インデックスかnameで指定可能。
        end: どのノードのまでパイプラインを実行するか。インデックスかnameで指定可能。
        """
        if to_ is None:
            to_ = len(self.nodes) - 1
        idx_from = self._get_start_or_end_index(from_, "start")
        self.idx_from = idx_from
        idx_to = self._get_start_or_end_index(to_, "end")
        if idx_from > idx_to:
            raise ValueError(
                f"idx_from must satisfy idx_from ({idx_from}) <= idx_to ({idx_to})"
            )

        logger.info(
            f"Scheduled {len(self.nodes[idx_from:idx_to+1])} tasks "
            + f"(from: {self.nodes[idx_from].name}, to: {self.nodes[idx_to].name})"
        )
        outputs = self._load_interim_output()
        for idx, node in enumerate(self.nodes[idx_from : idx_to + 1]):
            logger.info(
                f"Running {idx+1}/{len(self.nodes[idx_from:idx_to+1])} tasks ({node.name})"
            )

            # 最初のノードだけ特別扱い
            args = []
            if idx == 0 and idx_from == 0:
                if node.inputs is not None:
                    args.extend(node.inputs)
            else:
                inputs = self._get_inputs_from_previous_node(node, outputs)
                if inputs is not None:
                    args.extend(inputs)

            if node.config is not None:
                args.append(node.config)
            outputs = node.func(*args)

            outputs = _convert_item_to_list(outputs)
            if outputs is None:
                continue
            _assert_non_zero_length(outputs, "outputs")
            self._insert_outputs_to_dict(node, outputs)
            self._dump_outputs(node, outputs)

        logger.info(
            "Completed all tasks "
            + f"(from: {self.nodes[idx_from].name}, to: {self.nodes[idx_to].name})"
        )
        return outputs

    def _get_start_or_end_index(self, start_or_end: int | str, start_or_end_str: str):
        if isinstance(start_or_end, int):
            idx = start_or_end
        elif isinstance(start_or_end, str):
            try:
                idx = self.name_to_idx[start_or_end]
            except KeyError:
                raise ValueError(
                    f"specified {start_or_end_str} node ({start_or_end}) is not found"
                )

        if idx < 0 or idx >= len(self.nodes):
            raise ValueError(
                f"0 <= {start_or_end_str} ({idx}) <= {len(self.nodes)-1} must be satisfied"
            )
        return idx

    def _load_interim_output(self):
        """
        idx_fromから最後のノードまでのノードに対し、
        idx_fromより前のノードの結果を使うかチェックし、
        もし使うのであれば必要な結果をロードする。
        """
        if self.idx_from == 0:
            return None
        last_output = None
        loaded_files_set = set()
        # idx_fromから読み込むこと
        # さもないと直前のノードの入力が2回読まれる可能性がある
        for idx_ in range(self.idx_from, len(self.nodes)):
            # 関数の引数がゼロ個の時は何もしない
            num_param_when_no_inputs = 0 if self.nodes[idx_].config is None else 1
            if (
                len(signature(self.nodes[idx_].func).parameters)
                == num_param_when_no_inputs
            ):
                continue
            # 入力が指定されていない場合（前段の出力を入力とする場合）
            if self.nodes[idx_].inputs is None:
                # idx_from以降のタスクはまだ計算していないのでcontinue
                if idx_ > self.idx_from:
                    continue
                last_output = self._load_last_output(
                    self.nodes[idx_ - 1],
                    loaded_files_set,
                    idx_ - 1,
                )
            else:  # 前段より前の出力を入力にする場合
                self._load_past_output(self.nodes[idx_], loaded_files_set)
        return last_output

    def _load_last_output(self, node_prev, loaded_files_set, idx):
        if node_prev.outputs_loader is None:
            raise ValueError(
                f"outputs loader of node {node_prev.name} (index: {idx}) is not set."
            )

        # 本関数はrange(idx_from, len(self.nodes)): のループの最初に実行されるので、
        # outputs_loaderの呼び出しが冗長ではない
        if node_prev.outputs_loader_type == DumpType.BATCH:
            loaded_files_set.add(*node_prev.outputs_path)
            args = [*node_prev.outputs_path, node_prev.outputs]
            if node_prev.outputs_dumper_take_config:
                args.append(node_prev.config)
            results = node_prev.outputs_loader(*args)
            if node_prev.outputs is not None:
                _assert_same_length(
                    node_prev.outputs,
                    results,
                    "node_prev.outputs_path",
                    "results",
                )
                for output, result in zip(node_prev.outputs, results):
                    if output is not None:
                        self.results[output] = result
                        logger.debug(f"Saved '{output}' to memory")
            return results

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
        for idx, (output_path, outputs_loader) in enumerate(
            zip(node_prev.outputs_path, outputs_loaders)
        ):
            loaded_files_set.add(output_path)
            args = [output_path]
            if node_prev.outputs_dumper_take_config:
                args.append(node_prev.config)
            result = outputs_loader(*args)
            outputs.append(result)
            if node_prev.outputs is not None and node_prev.outputs[idx] is not None:
                self.results[node_prev.outputs[idx]] = result
                logger.debug(f"Saved '{node_prev.outputs[idx]}' to memory")
        return outputs

    def _load_past_output(self, node, loaded_files_set):
        for input in node.inputs:
            idx_dependant_node, idx_in_outputs = self.outputs_to_indexes[input]
            if idx_dependant_node >= self.idx_from:
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
                _assert_same_length(
                    node_dep.outputs,
                    outputs,
                    "node_dep.outputs",
                    "outputs",
                )
                for output_key, output in zip(node_dep.outputs, outputs):
                    self.results[output_key] = output
                    logger.debug(f"Saved '{output_key}' to memory")
                continue

            outputs_loaders = _convert_item_to_list(
                node_dep.outputs_loader, len(node_dep.outputs)
            )
            _assert_same_length(
                node_dep.outputs,
                outputs_loaders,
                "node_dep.outputs",
                "outputs_loaders",
            )
            idx_ = idx_in_outputs
            if node_dep.outputs_path[idx_] in loaded_files_set:
                continue
            loaded_files_set.add(node_dep.outputs_path[idx_])
            self.results[input] = outputs_loaders[idx_](node_dep.outputs_path[idx_])
            logger.debug(f"Saved '{input}' to memory")

    def _get_inputs_from_previous_node(self, node, previous_outputs):
        """前段までの入力を基に、次のNodeへの入力を決める"""
        # 入力がない場合は前段の結果を持ってくる
        if node.inputs is None:
            # if previous_outputs is None:
            #     raise ValueError("inputs is None")
            return previous_outputs

        _assert_non_zero_length(node.inputs, "node.inputs")
        return [self.results[input] for input in node.inputs]

    def _insert_outputs_to_dict(self, node, outputs):
        if node.outputs is None:
            return
        _assert_same_length(node.outputs, outputs, "node.outputs", "outputs")
        keys = []
        for key, output in zip(node.outputs, outputs):
            # nodeのoutputsにNoneが入っていたらdictには入れない
            if key is not None:
                self.results[key] = output
                keys.append(f"'{key}'")
        keys_str = ", ".join(keys)
        logger.debug(f"Saved {keys_str} to memory")

    def _dump_outputs(self, node, outputs):
        if node.outputs_dumper is None:
            return

        if node.outputs_dumper_type == DumpType.BATCH:
            args = [outputs, *node.outputs_path, node.outputs]
            if node.outputs_dumper_take_config:
                args.append(node.config)
            return node.outputs_dumper(*args)

        _assert_same_length(outputs, node.outputs_path, "outputs", "node.outputs_path")
        outputs_dumpers = _convert_item_to_list(node.outputs_dumper, len(outputs))
        _assert_same_length(outputs, outputs_dumpers, "outputs", "outputs_dumper")
        for output, filepath, outputs_dumper in zip(
            outputs, node.outputs_path, outputs_dumpers
        ):
            if outputs_dumper is not None:
                args = [output, filepath]
                if node.outputs_dumper_take_config:
                    args.append(node.config)
                outputs_dumper(*args)

    def clear(self):
        self.results.clear()


def _assert_non_zero_length(x, x_str):
    if len(x) == 0:
        raise ValueError(f"size of {x_str} is zero")


def _convert_item_to_list(data, length=1) -> list | tuple | None:
    """dataがlistでもtupleでもなければ、同じ要素が詰まった長さlengthの配列にして返す。"""
    if isinstance(data, list) or isinstance(data, tuple) or data is None:
        return data
    else:
        return [data] * length
