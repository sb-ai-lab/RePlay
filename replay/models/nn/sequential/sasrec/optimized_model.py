from typing import Optional

import torch



import os
from typing import List, Literal

import openvino as ov
from torch.utils.data import DataLoader
from tqdm import tqdm

from replay.models.nn.sequential.callbacks.prediction_callbacks import (
    BasePredictionCallback,
)
from replay.models.nn.sequential.sasrec import (
    SasRec,
    SasRecPredictionBatch,
)

OptimizedModeType = Literal[
    "batch",
    "one_query",
    "dynamic_batch_size",
    "dynamic_one_query",
]


class OptimizedSasRec:
    """
    SasRec model CPU-optimized for inference via OpenVINO
    """

    def __init__(
        self,
        onnx_path: str,
        mode: OptimizedModeType = "one_query",
        device_name: Optional[str] = "CPU",
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        num_candidates_to_score: Optional[int] = None,
    ) -> None:
        """
        :param onnx_path: Path to SasRec model saved in ONNX format.
        :param mode: Inference mode, defines shape of inputs.
            Could be one of [``one_query``, ``batch``,
            ``dynamic_one_query``, ``dynamic_batch_size``].\n
            ``one_query`` - sets input shape to [1, max_seq_len]\n
            ``batch`` - sets input shape to [batch_size, max_seq_len]\n
            ``dynamic_one_query`` - sets max_seq_len to dynamic range [1, ?]\n
            ``dynamic_batch_size`` - sets batch_size to dynamic range [?, max_seq_len]\n
            Default: ``one_query``.
        :param device_name: Device name according to OpenVINO interfaces.
            Default: ``CPU``.
        :param batch_size: Batch size, required for ``batch`` mode.
            Default: ``None``.
        :param max_seq_len: Max length of sequence. Required for
            ``one_query``, ``batch`` and ``dynamic_batch_size`` modes.
            Default: ``None``.
        :param num_candidates_to_score: Number of item ids to calculate scores.
            Default: ``None``.
        """
        self._mode: OptimizedModeType = mode
        self._batch_size: int
        self._max_seq_len: int
        self._inputs_names: List[str]
        self._output_name: str

        self._core = ov.Core()
        self._set_io_names(onnx_path)
        self._set_input_params(
            self._mode, batch_size, max_seq_len, num_candidates_to_score
        )

        model_input_scheme = [
            (input_name, [self._batch_size, self._max_seq_len])
            for input_name in self._inputs_names[:2]
        ]
        if self._num_candidates_to_score is not None:
            model_input_scheme += [
                (self._inputs_names[2], [self._num_candidates_to_score])
            ]
        model_onnx = ov.convert_model(
            onnx_path,
            input=model_input_scheme,
        )
        self._model = self._core.compile_model(
            model=model_onnx, device_name=device_name
        )

    def _prepare_prediction_batch(
        self, batch: SasRecPredictionBatch
    ) -> SasRecPredictionBatch:
        if self._mode == "dynamic_one_query":
            return batch

        if batch.padding_mask.shape[1] > self._max_seq_len:
            msg = f"The length of the submitted sequence \
                must not exceed the maximum length of the sequence. \
                The length of the sequence is given {batch.padding_mask.shape[1]}, \
                while the maximum length is {self._max_seq_len}"
            raise ValueError(msg)

        if batch.padding_mask.shape[1] < self._max_seq_len:
            query_id, padding_mask, features = batch
            sequence_item_count = padding_mask.shape[1]
            for feature_name, feature_tensor in features.items():
                features[feature_name] = torch.nn.functional.pad(
                    feature_tensor,
                    (self._max_seq_len - sequence_item_count, 0),
                    value=0,
                )
            padding_mask = torch.nn.functional.pad(
                padding_mask, (self._max_seq_len - sequence_item_count, 0), value=0
            )
            batch = SasRecPredictionBatch(query_id, padding_mask, features)
        return batch

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
            msg = "If num_candidates_to_score equals 0, it is impossible to infer the model with passed candidates_to_score."
            raise ValueError(msg)
        batch = self._prepare_prediction_batch(batch)
        model_inputs = {
            self._inputs_names[0]: batch.features[self._inputs_names[0]],
            self._inputs_names[1]: batch.padding_mask,
        }
        if self._num_candidates_to_score is not None:
            model_inputs[self._inputs_names[2]] = candidates_to_score
        return torch.from_numpy(self._model(model_inputs)[self._output_name])

    def predict_dataloader(
        self,
        dataloader: DataLoader,
        callbacks: List[BasePredictionCallback],
        candidates_to_score: Optional[torch.LongTensor],
        show_progress_bar: bool = False,
    ) -> None:
        """
        Inference on PyTorch DataLoader.

        :param dataloader: Prediction input.
        :param callbacks: List of callbacks to process scores.
        :param candidates_to_score: Item ids to calculate scores.
            Default: ``None``.
        :param show_progress_bar: Whether to enable to progress bar.
            Default: ``False``.
        """
        for batch in tqdm(
            dataloader, desc="Predicting Dataloader", disable=not show_progress_bar
        ):
            batch: SasRecPredictionBatch
            if (self._mode == "batch") and (
                batch.padding_mask.shape[0] != self._batch_size
            ):
                continue

            scores = self.predict(batch, candidates_to_score)
            for callback in callbacks:
                callback.on_predict_batch_end(0, 0, scores, batch, 0, 0)

    def _set_io_names(self, onnx_path: str) -> None:
        model_onnx = self._core.read_model(model=onnx_path)
        self._inputs_names = [inputs.names.pop() for inputs in model_onnx.inputs]
        self._output_name = model_onnx.output().names.pop()

        del model_onnx

    def _set_input_params(
        self,
        mode: OptimizedModeType,
        batch_size: Optional[int],
        max_seq_len: Optional[int],
        num_candidates_to_score: Optional[int],
    ) -> None:
        if mode == "one_query":
            assert max_seq_len, f"{mode} mode requires `max_seq_len`"
            self._batch_size = 1
            self._max_seq_len = max_seq_len

        if mode == "batch":
            assert batch_size, f"{mode} mode requires `batch_size`"
            assert max_seq_len, f"{mode} mode requires `max_seq_len`"
            self._batch_size = batch_size
            self._max_seq_len = max_seq_len

        if mode == "dynamic_batch_size":
            assert max_seq_len, f"{mode} mode requires `max_seq_len`"
            self._batch_size = -1
            self._max_seq_len = max_seq_len

        if mode == "dynamic_one_query":
            self._batch_size = 1
            self._max_seq_len = -1

        self._num_candidates_to_score = num_candidates_to_score

    def _validate_num_candidates_to_score(lightning_model, num_candidates_to_score):
        total_item_count = lightning_model._model.item_count
        if not isinstance(num_candidates_to_score, int):
            msg = f"Expected num_candidates_to_score of type int, got {type(num_candidates_to_score)}"
            raise ValueError(msg)
        elif not (0 < num_candidates_to_score <= total_item_count):
            msg = (
                f"Expected number of candidates to be between 1 and {total_item_count=}"
            )
            raise ValueError(msg)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        mode: OptimizedModeType = "one_query",
        device_name: Optional[str] = "CPU",
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        num_candidates_to_score: Optional[int] = None,
        onnx_path: Optional[str] = None,
    ) -> "OptimizedSasRec":
        """
        :param checkpoint_path: Path to lightning SasRec model saved in .ckpt format.
        :param mode: Inference mode, defines shape of inputs.
            Could be one of [``one_query``, ``batch``,
            ``dynamic_one_query``, ``dynamic_batch_size``].\n
            ``one_query`` - sets input shape to [1, max_seq_len]\n
            ``batch`` - sets input shape to [batch_size, max_seq_len]\n
            ``dynamic_one_query`` - sets max_seq_len to dynamic range [1, ?]\n
            ``dynamic_batch_size`` - sets batch_size to dynamic range [?, max_seq_len]\n
            Default: ``one_query``.
        :param device_name: Device name according to OpenVINO interfaces.
            Default: ``CPU``.
        :param batch_size: Batch size, required for ``batch`` mode.
            Default: ``None``.
        :param max_seq_len: Max length of sequence. Required for
            ``one_query``, ``batch`` and ``dynamic_batch_size`` modes.
            Default: ``None``.
        :param num_candidates_to_score: Number of item ids to calculate scores.
            Default: ``None``.
        :param onnx_path: Save ONNX model to path, if defined.
            Default: ``None``.
        """
        lightning_model = SasRec.load_from_checkpoint(
            checkpoint_path, map_location=torch.device("cpu")
        )
        item_seq_name = lightning_model._schema.item_id_feature_name
        max_len = lightning_model._model.max_len

        item_sequence = torch.zeros((1, max_len)).long()
        padding_mask = torch.zeros((1, max_len)).bool()

        model_input_names = [item_seq_name, "padding_mask"]
        model_dynamic_axes_in_input = {
            item_seq_name: {0: "batch_size", 1: "max_len"},
            "padding_mask": {0: "batch_size", 1: "max_len"},
        }
        if num_candidates_to_score is not None:
            cls._validate_num_candidates_to_score(
                lightning_model, num_candidates_to_score
            )
            candidates_to_score = torch.zeros((num_candidates_to_score,)).long()
            model_input_names += ["candidates_to_score"]
            model_dynamic_axes_in_input["candidates_to_score"] = {
                0: "num_candidates_to_score"
            }
            model_input_sample = (
                {item_seq_name: item_sequence},
                padding_mask,
                candidates_to_score,
            )
        else:
            model_input_sample = ({item_seq_name: item_sequence}, padding_mask)

        is_saveble = onnx_path is not None
        if onnx_path is None:
            onnx_path = checkpoint_path.rpartition(".")[0] + "_optimized.onnx"

        lightning_model.to_onnx(
            onnx_path,
            input_sample=model_input_sample,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=model_input_names,
            output_names=["scores"],
            dynamic_axes=model_dynamic_axes_in_input,
        )
        del lightning_model

        if max_seq_len is None:
            max_seq_len = max_len

        optimized_model = cls(
            onnx_path,
            mode,
            device_name,
            batch_size,
            max_seq_len,
            num_candidates_to_score,
        )
        if not is_saveble:
            os.remove(onnx_path)

        return optimized_model
