import abc
from typing import Generic, Optional, TypeVar

import lightning
import torch

from replay.nn.lightning import LightningModule
from replay.nn.lightning.postprocessor import PostprocessorBase
from replay.nn.output import InferenceOutput
from replay.utils import (
    PYSPARK_AVAILABLE,
    MissingImport,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
)

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf
    from pyspark.sql import SparkSession
    from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StructType
else:
    SparkSession = MissingImport


_T = TypeVar("_T")


class TopItemsCallbackBase(lightning.Callback, Generic[_T]):
    """
    The base class for a callback that records the result at the inference stage via ``LightningModule``.
    The result consists of top K the highest logit values, IDs of these top K logit values
    and corresponding query ids (encoded IDs of users).

    For the callback to work correctly, the batch is expected to contain the key passed as a parameter ``query_column``.
    """

    def __init__(
        self,
        top_k: int,
        query_column: str,
        item_column: str,
        rating_column: str = "rating",
        postprocessors: Optional[list[PostprocessorBase]] = None,
    ) -> None:
        """
        :param top_k: Take the ``top_k`` IDs with the highest logit values.
        :param query_column: The name of the query column in the batch and in the resulting dataframe.
        :param item_column: The name of the item column in the resulting dataframe.
        :param rating_column: The name of the rating column in the resulting dataframe.
            This column will contain the ``top_k`` items with the highest logit values.
        :param postprocessors: A list of postprocessors for modifying logits from the model
            before sorting and taking top K ones.
            For example, it can be a softmax operation to logits or set the ``-inf`` value for some IDs.
            Default: ``None``.
        """
        super().__init__()
        self.query_column = query_column
        self.item_column = item_column
        self.rating_column = rating_column
        self._top_k = top_k
        self._postprocessors: list[PostprocessorBase] = postprocessors or []
        self._query_batches: list[torch.Tensor] = []
        self._item_batches: list[torch.Tensor] = []
        self._item_scores: list[torch.Tensor] = []

    def on_predict_epoch_start(
        self,
        trainer: lightning.Trainer,  # noqa: ARG002
        pl_module: LightningModule,
    ) -> None:
        self._query_batches.clear()
        self._item_batches.clear()
        self._item_scores.clear()

        candidates = pl_module.candidates_to_score
        for postprocessor in self._postprocessors:
            postprocessor.candidates = candidates

    def on_predict_batch_end(
        self,
        trainer: lightning.Trainer,  # noqa: ARG002
        pl_module: LightningModule,
        outputs: InferenceOutput,
        batch: dict,
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        logits = self._apply_postproccesors(batch, outputs["logits"])
        top_scores, top_item_ids = torch.topk(logits, k=self._top_k, dim=1)
        if pl_module.candidates_to_score is not None:
            top_item_ids = torch.take(pl_module.candidates_to_score, top_item_ids)

        self._query_batches.append(batch[self.query_column])
        self._item_batches.append(top_item_ids)
        self._item_scores.append(top_scores)

    def get_result(self) -> _T:
        """
        :returns: prediction result
        """
        prediction = self._ids_to_result(
            torch.cat(self._query_batches),
            torch.cat(self._item_batches),
            torch.cat(self._item_scores),
        )
        return prediction

    def _apply_postproccesors(self, batch: dict, logits: torch.Tensor) -> torch.Tensor:
        for postprocessor in self._postprocessors:
            logits = postprocessor.on_prediction(batch, logits)
        return logits

    @abc.abstractmethod
    def _ids_to_result(
        self,
        query_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_scores: torch.Tensor,
    ) -> _T:  # pragma: no cover
        pass


class PandasTopItemsCallback(TopItemsCallbackBase[PandasDataFrame]):
    """
    A callback that records the result of the model's forward function at the inference stage in a Pandas Dataframe.
    """

    def _ids_to_result(
        self,
        query_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_scores: torch.Tensor,
    ) -> PandasDataFrame:
        prediction = PandasDataFrame(
            {
                self.query_column: query_ids.flatten().cpu().numpy(),
                self.item_column: list(item_ids.cpu().numpy()),
                self.rating_column: list(item_scores.cpu().numpy()),
            }
        )
        return prediction.explode([self.item_column, self.rating_column])


class PolarsTopItemsCallback(TopItemsCallbackBase[PolarsDataFrame]):
    """
    A callback that records the result of the model's forward function at the inference stage in a Polars Dataframe.
    """

    def _ids_to_result(
        self,
        query_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_scores: torch.Tensor,
    ) -> PolarsDataFrame:
        prediction = PolarsDataFrame(
            {
                self.query_column: query_ids.flatten().cpu().numpy(),
                self.item_column: list(item_ids.cpu().numpy()),
                self.rating_column: list(item_scores.cpu().numpy()),
            }
        )
        return prediction.explode([self.item_column, self.rating_column])


class SparkTopItemsCallback(TopItemsCallbackBase[SparkDataFrame]):
    """
    A callback that records the result of the model's forward function at the inference stage in a Spark Dataframe.
    """

    def __init__(
        self,
        top_k: int,
        query_column: str,
        item_column: str,
        rating_column: str,
        spark_session: SparkSession,
        postprocessors: Optional[list[PostprocessorBase]] = None,
    ) -> None:
        """
        :param top_k: Take the ``top_k`` IDs with the highest logit values.
        :param query_column: The name of the query column in the resulting dataframe.
        :param item_column: The name of the item column in the resulting dataframe.
        :param rating_column: The name of the rating column in the resulting dataframe.
            This column will contain the ``top_k`` items with the highest logit values.
        :param spark_session: Spark session. Required to create a Spark DataFrame.
        :param postprocessors: A list of postprocessors for modifying logits from the model
            before sorting and taking top K ones.
            For example, it can be a softmax operation to logits or set the ``-inf`` value for some IDs.
            Default: ``None``.
        """
        super().__init__(
            top_k=top_k,
            query_column=query_column,
            item_column=item_column,
            rating_column=rating_column,
            postprocessors=postprocessors,
        )
        self.spark_session = spark_session

    def _ids_to_result(
        self,
        query_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_scores: torch.Tensor,
    ) -> SparkDataFrame:
        schema = (
            StructType()
            .add(self.query_column, IntegerType(), False)
            .add(self.item_column, ArrayType(IntegerType()), False)
            .add(self.rating_column, ArrayType(DoubleType()), False)
        )
        prediction = (
            self.spark_session.createDataFrame(
                data=list(
                    zip(
                        query_ids.flatten().cpu().numpy().tolist(),
                        item_ids.cpu().numpy().tolist(),
                        item_scores.cpu().numpy().tolist(),
                    )
                ),
                schema=schema,
            )
            .withColumn(
                "exploded_columns",
                sf.explode(sf.arrays_zip(self.item_column, self.rating_column)),
            )
            .select(
                self.query_column,
                f"exploded_columns.{self.item_column}",
                f"exploded_columns.{self.rating_column}",
            )
        )
        return prediction


class TorchTopItemsCallback(TopItemsCallbackBase[tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]]):
    """
    A callback that records the result of the model's forward function at the inference stage in a PyTorch Tensors.
    """

    def __init__(
        self,
        top_k: int,
        query_column: str,
        postprocessors: Optional[list[PostprocessorBase]] = None,
    ) -> None:
        """
        :param top_k: Take the ``top_k`` IDs with the highest logit values.
        :param query_column: The name of the query column in the batch.
        :param postprocessors: A list of postprocessors for modifying logits from the model
            before sorting and taking top K.
            For example, it can be a softmax operation to logits or set the ``-inf`` value for some IDs.
            Default: ``None``.
        """
        super().__init__(
            top_k=top_k,
            query_column=query_column,
            item_column="item_id",
            rating_column="rating",
            postprocessors=postprocessors,
        )

    def _ids_to_result(
        self,
        query_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_scores: torch.Tensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
        return (
            query_ids.flatten().cpu().long(),
            item_ids.cpu().long(),
            item_scores.cpu(),
        )


class HiddenStatesCallback(lightning.Callback):
    """
    A callback for getting any hidden state from the model.

    When applying this callback,
    it is expected that the result of the model's forward function contains the ``hidden_states`` key.
    """

    def __init__(self, hidden_state_index: int):
        """
        :param hidden_state_index: It is expected that the result of the model's forward function
            contains the ``hidden_states`` key. ``hidden_states`` key contains Tuple of PyTorch Tensors.
            Therefore, to get a specific hidden state, you need to submit an index from this tuple.
        """
        self._hidden_state_index = hidden_state_index
        self._embeddings_per_batch: list[torch.Tensor] = []

    def on_predict_epoch_start(
        self,
        trainer: lightning.Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        self._embeddings_per_batch.clear()

    def on_predict_batch_end(
        self,
        trainer: lightning.Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        outputs: InferenceOutput,
        batch: dict,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        self._embeddings_per_batch.append(outputs["hidden_states"][self._hidden_state_index].detach().cpu())

    def get_result(self):
        """
        :returns: Hidden states through all batches.
        """
        return torch.cat(self._embeddings_per_batch)
