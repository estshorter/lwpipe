from __future__ import annotations

import logging
from enum import IntEnum, auto
from typing import Callable, Optional

logger = logging.getLogger(__name__)


__version__ = "0.0.1"


class InputType(IntEnum):
    NON_INTERIM_RESULT = auto()
    INTERIM_RESULT = auto()


class Node:
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        inputs: str | list[str] | None = None,
        inputs_type: InputType | list[InputType] = InputType.INTERIM_RESULT,
        inputs_loader: Callable | list[Callable] | None | list[None] = None,
        outputs: str | list[str] | None = None,
        outputs_dumper: Callable | list[Callable] | None = None,
        outputs_path: Optional[list[str]] = None,
        outputs_loader: Callable | list[Callable] | None = None,
    ) -> None:
        """
        Parameters
        ------------------
        func: 適用する関数。
        name: 関数の名前。Noneのときはfunc.__name__が代入される。
        inputs: 入力データ。最初のノードに対しNoneを設定すると、引数0個の関数をfuncにセットできる。
                それ以外のノードでNoneを設定した場合は、前段の出力を入力として使うという設定になる。
                文字列を渡されており、かつ、対応するinputs_typeがINTERIM_RESULT
                のときは、この名前の中間結果を読もうとする。
        inputs_type: Pipeline内で計算した結果をfuncへの入力にしたい場合はINTERIM_RESULTに設定する。
                     それ以外の場合はNON_INTERIM_RESULT。
        inputs_loader: inputsに適用する関数。
                       典型的には、引数で受け取ったファイルパスをloadしたデータを返す関数を入れる。
                       Noneの場合は何も適用しない。
        outputs: 出力結果をdictに入れる際のキー。Noneにすると保存されず、次のノードに渡されるのみとなる。
        outputs_dumper: outputsをdumpする関数。リストを渡せば、各変数に対して別々の関数を適用可能。
        outputs_path: dumpするファイルパス。
        outputs_loader: dumpしたoutputsをloadするための関数。
                        pipelineを途中から実行する場合、中間結果をloadする必要があるが、その際に使われる。
        """
        self.func = func
        if name is not None:
            self.name = name
        else:
            self.name = func.__name__

        self.inputs = _convert_item_to_list(inputs)
        self.outputs = _convert_item_to_list(outputs)

        self.inputs_type = inputs_type
        self.inputs_loader = inputs_loader
        self.outputs_dumper = outputs_dumper
        self.outputs_path = _convert_item_to_list(outputs_path)
        self.outputs_loader = outputs_loader


class Pipeline:
    def __init__(self, nodes: list[Node]) -> None:
        self.nodes = nodes
        self.results = dict()

        self.name_to_idx = dict()
        self.outputs_to_idx = dict()

        _assert_non_zero_length(nodes)
        # 最初のノードは中間結果はありえないので設定を上書きする。
        nodes[0].inputs_type = [InputType.NON_INTERIM_RESULT] * len(nodes[0].inputs)

        for idx, node in enumerate(self.nodes):
            # lambda関数のときは名前の重複を許し、nodes内のidxをsuffixに付与する
            if node.name == "<lambda>":
                node.name = f"<lambda>_pos{idx}_"
            if node.name in self.name_to_idx:
                raise ValueError(f"node name '{node.name}' is duplicated.")
            self.name_to_idx[node.name] = idx
            if node.outputs is None:
                continue
            for output in node.outputs:
                self.outputs_to_idx[output] = idx

    def run(self, start: int | str = 0):
        """pipelineを実行する。戻り値はlist。"""
        idx_start = self._get_start_index(start)
        logger.info(
            f"Total {len(self.nodes)} tasks, scheduled {len(self.nodes[idx_start:])} tasks"
        )
        outputs = self._load_interim_output(idx_start)
        for idx, node in enumerate(self.nodes[idx_start:]):
            logger.info(
                f"Running {idx+1}/{len(self.nodes[idx_start:])} tasks ({node.name})"
            )

            # 最初の層に入力データがないときにも対応させる
            if idx == 0 and idx_start == 0 and node.inputs is None:
                outputs = node.func()
            else:
                inputs = self._get_inputs(node, outputs)
                inputs_loaded = self._load_inputs(node, inputs)
                outputs = node.func(*inputs_loaded)

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
        outputs = None
        for idx_ in range(idx_start, len(self.nodes)):
            # 入力が指定されていない場合（前段の出力を入力とする場合）
            if self.nodes[idx_].inputs is None:
                if idx_ > idx_start:
                    # idx_start以降のタスクはまだ計算していないのでcontinue
                    continue
                # 実質 idx_= idx_start
                node_prev = self.nodes[idx_ - 1]
                outputs = []
                _assert_non_zero_length(
                    node_prev.outputs_path, "node_prev.outputs_path"
                )

                outputs_loaders = _convert_item_to_list(
                    node_prev.outputs_loader, len(node_prev.outputs_path)
                )
                _assert_same_length(
                    node_prev.outputs_path,
                    outputs_loaders,
                    "node_prev.outputs_path",
                    "outputs_loaders",
                )

                for output_path, outputs_loader in zip(
                    node_prev.outputs_path, outputs_loaders
                ):
                    outputs.append(outputs_loader(output_path))
            else:  # 前段より前の出力を入力にする場合
                for input in self.nodes[idx_].inputs:
                    dependant_node = self.outputs_to_idx[input]
                    if dependant_node >= idx_start:
                        # まだ計算していないのでcontinue
                        continue
                    node_prev = self.nodes[self.outputs_to_idx[input]]

                    outputs_loaders = _convert_item_to_list(
                        node_prev.outputs_loader, len(node_prev.outputs)
                    )
                    _assert_same_length(
                        node_prev.outputs,
                        outputs_loaders,
                        "node_prev.outputs",
                        "outputs_loaders",
                    )

                    # inputに指定したデータが何番目かを求める
                    index_ = node_prev.outputs.index(input)
                    self.results[input] = outputs_loaders[index_](
                        node_prev.outputs_path[index_]
                    )
        return outputs

    def _get_inputs(self, node, previous_outputs):
        """前段までの入力を基に、次のNodeへの入力を決める"""
        # 入力がない場合は前段の結果を持ってくる
        if node.inputs is None:
            if previous_outputs is None:
                raise ValueError("inputs is None")
            return previous_outputs

        inputs = []
        _assert_non_zero_length(node.inputs, "node.inputs")

        inputs_type = _convert_item_to_list(node.inputs_type, len(node.inputs))
        _assert_same_length(node.inputs, inputs_type, "node.inputs", "inputs_type")
        for input, input_type in zip(node.inputs, inputs_type):
            if input_type == InputType.INTERIM_RESULT:
                inputs.append(self.results[input])
            elif input_type == InputType.NON_INTERIM_RESULT:
                inputs.append(input)
            else:
                raise ValueError(f"unknown input_type: {type(input_type)}")
        return inputs

    def _load_inputs(selfj, node, inputs):
        if node.inputs_loader is None:
            return inputs

        inputs_loaders = _convert_item_to_list(node.inputs_loader, len(inputs))
        _assert_same_length(inputs, inputs_loaders, "inputs", "inputs_loader")

        inputs_loaded = []
        for input, inputs_loader in zip(inputs, inputs_loaders):
            if inputs_loader is not None:
                inputs_loaded.append(inputs_loader(input))
            else:
                inputs_loaded.append(input)
        return inputs_loaded

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


def _assert_same_length(x, y, x_str, y_str):
    if len(x) != len(y):
        raise ValueError(
            f"size of {x_str} ({len(x)}) is different from {y_str} ({len(y)})"
        )


def _convert_item_to_list(data, length=1) -> list | tuple | None:
    """dataがlistでもtupleでもなければ、同じ要素が詰まった長さlengthの配列にして返す。"""
    if isinstance(data, list) or isinstance(data, tuple) or data is None:
        return data
    else:
        return [data] * length
