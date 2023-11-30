import abc
from typing import Generic, List, Optional, Protocol, Tuple, TypeVar, cast

import lightning as L
import torch

from replay.models.nn.sequential.postprocessors import BasePostProcessor
from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame, SparkDataFrame, MissingImportType

if PYSPARK_AVAILABLE:  # pragma: no cover
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StructType
else:
    SparkSession = MissingImportType


# pylint: disable=too-few-public-methods
class PredictionBatch(Protocol):
    """
    Prediction callback batch
    """
    query_id: torch.LongTensor


_T = TypeVar("_T")


# pylint: disable=too-many-instance-attributes
class BasePredictionCallback(L.Callback, Generic[_T]):
    """
    Base callback for prediction stage
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        top_k: int,
        query_column: str,
        item_column: str,
        rating_column: str = "rating",
        postprocessors: Optional[List[BasePostProcessor]] = None,
    ) -> None:
        """
        :param top_k: Takes the highest k scores in the ranking.
        :param query_column: query column name.
        :param item_column: item column name.
        :param rating_column: rating column name.
        :param postprocessors: postprocessors to apply.
        """
        super().__init__()
        self.query_column = query_column
        self.item_column = item_column
        self.rating_column = rating_column
        self._top_k = top_k
        self._postprocessors: List[BasePostProcessor] = postprocessors or []
        self._query_batches: List[torch.Tensor] = []
        self._item_batches: List[torch.Tensor] = []
        self._item_scores: List[torch.Tensor] = []

    # pylint: disable=unused-argument
    def on_predict_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._query_batches.clear()
        self._item_batches.clear()
        self._item_scores.clear()

    # pylint: disable=unused-argument, too-many-arguments
    def on_predict_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor,
        batch: PredictionBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        query_ids, scores = self._compute_pipeline(batch.query_id, outputs)
        top_scores, top_item_ids = torch.topk(scores, k=self._top_k, dim=1)
        self._query_batches.append(query_ids)
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

    def _compute_pipeline(
        self, query_ids: torch.LongTensor, scores: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        for postprocessor in self._postprocessors:
            query_ids, scores = postprocessor.on_prediction(query_ids, scores)
        return query_ids, scores

    @abc.abstractmethod
    def _ids_to_result(
        self,
        query_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_scores: torch.Tensor,
    ) -> _T:  # pragma: no cover
        pass


class PandasPredictionCallback(BasePredictionCallback[PandasDataFrame]):
    """
    Callback for predition stage with pandas data frame
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


class SparkPredictionCallback(BasePredictionCallback[SparkDataFrame]):
    """
    Callback for prediction stage with spark data frame
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        top_k: int,
        query_column: str,
        item_column: str,
        rating_column: str,
        spark_session: SparkSession,
        postprocessors: Optional[List[BasePostProcessor]] = None,
    ) -> None:
        """
        :param top_k: Takes the highest k scores in the ranking.
        :param query_column: query column name.
        :param item_column: item column name.
        :param rating_column: rating column name.
        :param postprocessors: postprocessors to apply.
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
            .withColumn("exploded_columns", F.explode(F.arrays_zip(self.item_column, self.rating_column)))
            .select(self.query_column, f"exploded_columns.{self.item_column}", f"exploded_columns.{self.rating_column}")
        )
        return prediction


class TorchPredictionCallback(BasePredictionCallback[Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]]):
    """
    Callback for predition stage with tuple of tensors
    """

    def __init__(
        self,
        top_k: int,
        postprocessors: Optional[List[BasePostProcessor]] = None,
    ) -> None:
        """
        :param top_k: Takes the highest k scores in the ranking.
        :param postprocessors: postprocessors to apply.
        """
        super().__init__(
            top_k=top_k,
            query_column="query_id",
            item_column="item_id",
            rating_column="rating",
            postprocessors=postprocessors,
        )

    def _ids_to_result(
        self,
        query_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_scores: torch.Tensor,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
        return (
            cast(torch.LongTensor, query_ids.flatten().cpu().long()),
            cast(torch.LongTensor, item_ids.cpu().long()),
            item_scores.cpu(),
        )
