import pathlib
import tempfile
from typing import List, Literal, Optional, Union, get_args

import openvino as ov
import torch

from replay.data.nn import TensorSchema
from replay.models.nn.sequential.sasrec import (
    SasRec,
    SasRecPredictionBatch,
)
from replay.models.nn.sequential.sasrec.lightning import _prepare_prediction_batch

OptimizedModeType = Literal[
    "batch",
    "one_query",
    "dynamic_batch_size",
]


def _compile_openvino(
    onnx_path: str,
    batch_size: int,
    max_seq_len: int,
    num_candidates_to_score: int,
    num_threads: int,
) -> ov.CompiledModel:
    core = ov.Core()
    core.set_property("CPU", {"INFERENCE_NUM_THREADS": num_threads})
    model_onnx = core.read_model(model=onnx_path)
    inputs_names = [inputs.names.pop() for inputs in model_onnx.inputs]
    del model_onnx

    model_input_scheme = [(input_name, [batch_size, max_seq_len]) for input_name in inputs_names[:2]]
    if num_candidates_to_score is not None:
        model_input_scheme += [(inputs_names[2], [num_candidates_to_score])]
    model_onnx = ov.convert_model(onnx_path, input=model_input_scheme)
    return core.compile_model(model=model_onnx, device_name="CPU")


class OptimizedSasRec:
    """
    SasRec CPU-optimized model for inference via OpenVINO.
    It is recommended to compile model from SasRec checkpoint or the object itself using ``compile`` method.
    It is also possible to compile model by yourself and pass it to the ``__init__``.
    Note that compilation requires disk write permission.
    """

    def __init__(
        self,
        compiled_model: ov.CompiledModel,
        schema: TensorSchema,
    ) -> None:
        """
        :param compiled_model: Compiled model.
        :param schema: Tensor schema of SasRec model.
        """
        self._batch_size: int
        self._max_seq_len: int
        self._inputs_names: List[str]

        self._set_inner_params_from_openvino_model(compiled_model)
        self._schema = schema
        self._model = compiled_model

    def predict(
        self,
        batch: SasRecPredictionBatch,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Inference on one batch.

        :param batch: Prediction input.
        :param candidates_to_score: Item ids to calculate scores.
            Default: ``None``.

        :return: Tensor with scores.
        """
        if self._num_candidates_to_score is None and candidates_to_score is not None:
            msg = (
                "If ``num_candidates_to_score`` is None, "
                "it is impossible to infer the model with passed ``candidates_to_score``."
            )
            raise ValueError(msg)

        if (self._batch_size != -1) and (batch.padding_mask.shape[0] != self._batch_size):
            msg = (
                f"The batch is smaller then defined batch_size={self._batch_size}. "
                "It is impossible to infer the model with dynamic batch size in ``mode`` = ``batch``. "
                "Use ``mode`` = ``dynamic_batch_size``."
            )
            raise ValueError(msg)

        batch = _prepare_prediction_batch(self._schema, self._max_seq_len, batch)
        model_inputs = {
            self._inputs_names[0]: batch.features[self._inputs_names[0]],
            self._inputs_names[1]: batch.padding_mask,
        }
        if self._num_candidates_to_score is not None:
            self._validate_candidates_to_score(candidates_to_score)
            model_inputs[self._inputs_names[2]] = candidates_to_score
        return torch.from_numpy(self._model(model_inputs)[self._output_name])

    def _validate_candidates_to_score(self, candidates: torch.LongTensor):
        if not (isinstance(candidates, torch.Tensor) and candidates.dtype is torch.long):
            msg = (
                "Expected candidates to be of type ``torch.Tensor`` with dtype ``torch.long``, "
                f"got {type(candidates)} with dtype {candidates.dtype}."
            )
            raise ValueError(msg)

    def _set_inner_params_from_openvino_model(self, compiled_model: ov.CompiledModel) -> None:
        input_scheme = compiled_model.inputs
        self._batch_size = input_scheme[0].partial_shape[0].max_length
        self._max_seq_len = input_scheme[0].partial_shape[1].max_length
        self._inputs_names = [input.names.pop() for input in compiled_model.inputs]
        if "candidates_to_score" in self._inputs_names:
            self._num_candidates_to_score = input_scheme[2].partial_shape[0].max_length
        else:
            self._num_candidates_to_score = None
        self._output_name: str = compiled_model.output().names.pop()

    @staticmethod
    def _validate_num_candidates_to_score(num_candidates: int):
        if num_candidates is None:
            return num_candidates
        if num_candidates == -1 or (num_candidates >= 1 and isinstance(num_candidates, int)):
            return int(num_candidates)

        msg = (
            "Expected num_candidates_to_score to be of type ``int``, equal to ``-1``, ``natural number`` or ``None``. "
            f"Got {num_candidates}."
        )
        raise ValueError(msg)

    @staticmethod
    def _get_input_params(
        mode: OptimizedModeType,
        batch_size: Optional[int],
        num_candidates_to_score: Optional[int],
    ) -> None:
        if mode == "one_query":
            batch_size = 1

        if mode == "batch":
            assert batch_size, f"{mode} mode requires `batch_size`"
            batch_size = batch_size

        if mode == "dynamic_batch_size":
            batch_size = -1

        num_candidates_to_score = num_candidates_to_score if num_candidates_to_score else None
        return batch_size, num_candidates_to_score

    @classmethod
    def compile(
        cls,
        model: Union[SasRec, str, pathlib.Path],
        mode: OptimizedModeType = "one_query",
        batch_size: Optional[int] = None,
        num_candidates_to_score: Optional[int] = None,
        num_threads: Optional[int] = 4,
    ) -> "OptimizedSasRec":
        """
        Model compilation.

        :param model: Path to lightning SasRec model saved in .ckpt format or the SasRec object itself.
        :param mode: Inference mode, defines shape of inputs.
            Could be one of [``one_query``, ``batch``, ``dynamic_batch_size``].\n
            ``one_query`` - sets input shape to [1, max_seq_len]\n
            ``batch`` - sets input shape to [batch_size, max_seq_len]\n
            ``dynamic_batch_size`` - sets batch_size to dynamic range [?, max_seq_len]\n
            Default: ``one_query``.
        :param batch_size: Batch size, required for ``batch`` mode.
            Default: ``None``.
        :param num_candidates_to_score: Number of item ids to calculate scores.
            Could be one of [``None``, ``-1``, ``N``].\n
            ``-1`` - sets candidates_to_score shape to dynamic range [1, ?]\n
            ``N`` - sets candidates_to_score shape to [1, N]\n
            ``None`` - disable candidates_to_score usage\n
            Default: ``None``.
        :param num_threads: Number of CPU threads to use.
            Default: ``4``.
        """
        if mode not in get_args(OptimizedModeType):
            msg = f"Parameter ``mode`` could be one of {get_args(OptimizedModeType)}."
            raise ValueError(msg)
        num_candidates_to_score = OptimizedSasRec._validate_num_candidates_to_score(num_candidates_to_score)
        if isinstance(model, SasRec):
            lightning_model = model.cpu()
        elif isinstance(model, (str, pathlib.Path)):
            lightning_model = SasRec.load_from_checkpoint(model, map_location=torch.device("cpu"))

        schema = lightning_model._schema
        item_seq_name = schema.item_id_feature_name
        max_seq_len = lightning_model._model.max_len

        batch_size, num_candidates_to_score = OptimizedSasRec._get_input_params(
            mode, batch_size, num_candidates_to_score
        )

        item_sequence = torch.zeros((1, max_seq_len)).long()
        padding_mask = torch.zeros((1, max_seq_len)).bool()

        model_input_names = [item_seq_name, "padding_mask"]
        model_dynamic_axes_in_input = {
            item_seq_name: {0: "batch_size", 1: "max_len"},
            "padding_mask": {0: "batch_size", 1: "max_len"},
        }
        if num_candidates_to_score:
            candidates_to_score = torch.zeros((1,)).long()
            model_input_names += ["candidates_to_score"]
            model_dynamic_axes_in_input["candidates_to_score"] = {0: "num_candidates_to_score"}
            model_input_sample = ({item_seq_name: item_sequence}, padding_mask, candidates_to_score)
        else:
            model_input_sample = ({item_seq_name: item_sequence}, padding_mask)

        onnx_file = tempfile.NamedTemporaryFile(suffix=".onnx")

        lightning_model.to_onnx(
            onnx_file.name,
            input_sample=model_input_sample,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=model_input_names,
            output_names=["scores"],
            dynamic_axes=model_dynamic_axes_in_input,
        )
        del lightning_model

        compiled_model = _compile_openvino(
            onnx_file.name, batch_size, max_seq_len, num_candidates_to_score, num_threads
        )

        onnx_file.close()

        return cls(compiled_model, schema)
