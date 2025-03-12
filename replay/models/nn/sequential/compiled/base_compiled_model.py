import pathlib
from abc import abstractmethod
from typing import Any, List, Literal, Optional, Union

import lightning
import openvino as ov
import torch

from replay.data.nn import TensorSchema

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
    num_threads: Optional[int],
) -> ov.CompiledModel:
    """
    Method defines compilation strategy for openvino backend.

    :param onnx_path: Path to the model representation in ONNX format.
    :param batch_size: Defines whether batch will be static or dynamic length.
    :param max_seq_len: Defines whether sequence will be static or dynamic length.
    :param num_candidates_to_score: Defines whether candidates will be static or dynamic length.
    :param num_threads: Defines number of CPU threads for which the model will be compiled by the OpenVino core.
        If ``None``, then compiler will set this parameter automatically.
        Default: ``None``.
    """
    core = ov.Core()
    if num_threads is not None:
        core.set_property("CPU", {"INFERENCE_NUM_THREADS": num_threads})
    model_onnx = core.read_model(model=onnx_path)
    inputs_names = [inputs.names.pop() for inputs in model_onnx.inputs]
    del model_onnx

    candidates_input_id = len(inputs_names) - 1 if num_candidates_to_score is not None else len(inputs_names)
    model_input_scheme = [(input_name, [batch_size, max_seq_len]) for input_name in inputs_names[:candidates_input_id]]
    if num_candidates_to_score is not None:
        model_input_scheme += [(inputs_names[candidates_input_id], [num_candidates_to_score])]
    model_onnx = ov.convert_model(onnx_path, input=model_input_scheme)
    return core.compile_model(model=model_onnx, device_name="CPU")


class BaseCompiledModel:
    """
    Base class of CPU-optimized model for inference via OpenVINO.
    It is recommended to use inherited classes and not to use this one.
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
        self._output_name: str

        self._set_inner_params_from_openvino_model(compiled_model)
        self._schema = schema
        self._model = compiled_model

    @abstractmethod
    def predict(
        self,
        batch: Any,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Inference on one batch.

        :param batch: Prediction input.
        :param candidates_to_score: Item ids to calculate scores.
            Default: ``None``.

        :return: Tensor with scores.
        """

    def _validate_candidates_to_score(self, candidates: torch.LongTensor) -> None:
        """Check if candidates param has proper type"""

        if not (isinstance(candidates, torch.Tensor) and candidates.dtype is torch.long):
            msg = (
                "Expected candidates to be of type ``torch.Tensor`` with dtype ``torch.long``, "
                f"got {type(candidates)} with dtype {candidates.dtype}."
            )
            raise ValueError(msg)

    def _set_inner_params_from_openvino_model(self, compiled_model: ov.CompiledModel) -> None:
        """Set params for ``predict`` method"""

        input_scheme = compiled_model.inputs
        self._batch_size = input_scheme[0].partial_shape[0].max_length
        self._max_seq_len = input_scheme[0].partial_shape[1].max_length
        self._inputs_names = [input.names.pop() for input in compiled_model.inputs]
        if "candidates_to_score" in self._inputs_names:
            self._num_candidates_to_score = input_scheme[-1].partial_shape[0].max_length
        else:
            self._num_candidates_to_score = None
        self._output_name = compiled_model.output().names.pop()

    @staticmethod
    def _validate_num_candidates_to_score(num_candidates: int) -> Union[int, None]:
        """Check if num_candidates param is proper"""

        if num_candidates is None:
            return num_candidates
        if isinstance(num_candidates, int) and (num_candidates == -1 or num_candidates >= 1):
            return num_candidates

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
        """Get params for model compilation according to compilation mode"""

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
    @abstractmethod
    def compile(
        cls,
        model: Union[lightning.LightningModule, str, pathlib.Path],
        mode: OptimizedModeType = "one_query",
        batch_size: Optional[int] = None,
        num_candidates_to_score: Optional[int] = None,
        num_threads: Optional[int] = None,
        onnx_path: Optional[str] = None,
    ) -> "BaseCompiledModel":
        """
        Model compilation.

        :param model: Path to lightning model saved in .ckpt format or the model object itself.
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
            Must be a natural number or ``None``.
            If ``None``, then compiler will set this parameter automatically.
            Default: ``None``.
        :param onnx_path: Save ONNX model to path, if defined.
            Default: ``None``.
        """
