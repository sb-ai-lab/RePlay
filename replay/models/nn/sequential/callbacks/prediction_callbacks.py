import abc
from typing import Generic, List, Optional, Protocol, Tuple, TypeVar, cast

import lightning
import torch

from replay.models.nn.sequential import Bert4Rec
from replay.models.nn.sequential.postprocessors import BasePostProcessor
from replay.utils import PYSPARK_AVAILABLE, MissingImportType, PandasDataFrame, PolarsDataFrame, SparkDataFrame

if PYSPARK_AVAILABLE:  # pragma: no cover
    import pyspark.sql.functions as sf
    from pyspark.sql import SparkSession
    from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StructType
else:
    SparkSession = MissingImportType


class PredictionBatch(Protocol):
    """
    Prediction callback batch
    """

    query_id: torch.LongTensor


_T = TypeVar("_T")


class BasePredictionCallback(lightning.Callback, Generic[_T]):
    """
    Base callback for prediction stage
    """

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

    def on_predict_epoch_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule  # noqa: ARG002
    ) -> None:
        self._query_batches.clear()
        self._item_batches.clear()
        self._item_scores.clear()

    def on_predict_batch_end(
        self,
        trainer: lightning.Trainer,  # noqa: ARG002
        pl_module: lightning.LightningModule,  # noqa: ARG002
        outputs: torch.Tensor,
        batch: PredictionBatch,
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
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


class PolarsPredictionCallback(BasePredictionCallback[PolarsDataFrame]):
    """
    Callback for predition stage with polars data frame
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


class SparkPredictionCallback(BasePredictionCallback[SparkDataFrame]):
    """
    Callback for prediction stage with spark data frame
    """

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
            .withColumn("exploded_columns", sf.explode(sf.arrays_zip(self.item_column, self.rating_column)))
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


class QueryEmbeddingsPredictionCallback(lightning.Callback):
    """
    Callback for prediction stage to get query embeddings.
    """

    def __init__(self):
        self._embeddings_per_batch: List[torch.Tensor] = []

    def on_predict_epoch_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule  # noqa: ARG002
    ) -> None:
        self._embeddings_per_batch.clear()

    def on_predict_batch_end(
        self,
        trainer: lightning.Trainer,  # noqa: ARG002
        pl_module: lightning.LightningModule,
        outputs: torch.Tensor,  # noqa: ARG002
        batch: PredictionBatch,
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        args = [batch.features, batch.padding_mask]
        if isinstance(pl_module, Bert4Rec):
            args.append(batch.tokens_mask)

        query_embeddings = pl_module._model.get_query_embeddings(*args)
        self._embeddings_per_batch.append(query_embeddings)

    def get_result(self):
        """
        :returns: Query embeddings through all batches.
        """
        return torch.cat(self._embeddings_per_batch)
