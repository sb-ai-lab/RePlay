import pathlib
from typing import Optional, Union, get_args

import openvino as ov
import torch

from replay.data.nn import TensorSchema
from replay.models.nn.sequential.compiled.base_compiled_model import (
    BaseCompiledModel,
    OptimizedModeType,
)
from replay.models.nn.sequential.sasrec import (
    SasRec,
    SasRecPredictionBatch,
)
from replay.models.nn.sequential.sasrec.lightning import _prepare_prediction_batch


class SasRecCompiled(BaseCompiledModel):
    """
    SasRec CPU-optimized model for inference via OpenVINO.
    It is recommended to compile model with ``compile`` method and pass ``SasRec`` checkpoint
    or the model object itself into it.
    It is also possible to compile model by yourself and pass it to the ``__init__`` with ``TensorSchema``.

    **Note** that compilation requires disk write (and maybe delete) permission.
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
        super().__init__(compiled_model, schema)

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
        self._valilade_predict_input(batch, candidates_to_score)

        batch = _prepare_prediction_batch(self._schema, self._max_seq_len, batch)
        model_inputs = {
            self._inputs_names[0]: batch.features[self._inputs_names[0]],
            self._inputs_names[1]: batch.padding_mask,
        }
        if self._num_candidates_to_score is not None:
            self._validate_candidates_to_score(candidates_to_score)
            model_inputs[self._inputs_names[2]] = candidates_to_score
        return torch.from_numpy(self._model(model_inputs)[self._output_name])

    @classmethod
    def compile(
        cls,
        model: Union[SasRec, str, pathlib.Path],
        mode: OptimizedModeType = "one_query",
        batch_size: Optional[int] = None,
        num_candidates_to_score: Optional[int] = None,
        num_threads: Optional[int] = None,
        onnx_path: Optional[str] = None,
    ) -> "SasRecCompiled":
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
            Must be a natural number or ``None``.
            If ``None``, then compiler will set this parameter automatically.
            Default: ``None``.
        :param onnx_path: Save ONNX model to path, if defined.
            Default: ``None``.
        """
        if mode not in get_args(OptimizedModeType):
            msg = f"Parameter ``mode`` could be one of {get_args(OptimizedModeType)}."
            raise ValueError(msg)
        num_candidates_to_score = SasRecCompiled._validate_num_candidates_to_score(num_candidates_to_score)
        if isinstance(model, SasRec):
            lightning_model = model.cpu()
        elif isinstance(model, (str, pathlib.Path)):
            lightning_model = SasRec.load_from_checkpoint(model, map_location=torch.device("cpu"))

        schema = lightning_model._schema
        item_seq_name = schema.item_id_feature_name
        max_seq_len = lightning_model._model.max_len

        batch_size, num_candidates_to_score = SasRecCompiled._get_input_params(
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

        compiled_model = SasRecCompiled._run_model_compilation(
            lightning_model,
            model_input_sample,
            model_input_names,
            model_dynamic_axes_in_input,
            batch_size,
            num_candidates_to_score,
            num_threads,
            onnx_path,
        )

        return cls(compiled_model, schema)
