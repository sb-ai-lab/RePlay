import importlib
import logging
from abc import abstractmethod
from typing import Any, Dict, Iterable, Optional, Union

from replay.data import Dataset
from replay.models.base_rec import BaseRecommender
from replay.utils import PYSPARK_AVAILABLE, SparkDataFrame

from .index_builders.base_index_builder import IndexBuilder

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf

    from replay.utils.spark_utils import get_top_k_recs, return_recs

    from .index_stores.spark_files_index_store import SparkFilesIndexStore


logger = logging.getLogger("replay")


class ANNMixin(BaseRecommender):
    """
    This class overrides the `_fit_wrap` and `_predict_wrap` methods of the base class,
    adding an index construction in the `_fit_wrap` step
    and an index inference in the `_predict_wrap` step.
    """

    index_builder: Optional[IndexBuilder] = None

    @property
    def _use_ann(self) -> bool:
        """
        Property that determines whether the ANN (index) is used.
        If `True`, then the index will be built (at the `fit` stage)
        and index will be inferred (at the `predict` stage).
        """
        return self.index_builder is not None

    @abstractmethod
    def _get_vectors_to_build_ann(self, interactions: SparkDataFrame) -> SparkDataFrame:
        """Implementations of this method must return a dataframe with item vectors.
        Item vectors from this method are used to build the index.

        Args:
            log: DataFrame with interactions

        Returns: DataFrame[item_idx int, vector array<double>] or DataFrame[vector array<double>].
        Column names in dataframe can be anything.
        """

    @abstractmethod
    def _get_ann_build_params(self, interactions: SparkDataFrame) -> Dict[str, Any]:
        """Implementation of this method must return dictionary
        with arguments for `_build_ann_index` method.

        Args:
            interactions: DataFrame with interactions

        Returns: Dictionary with arguments to build index. For example: {
            "id_col": "item_idx",
            "features_col": "item_factors",
            ...
        }

        """

    def _fit_wrap(
        self,
        dataset: Dataset,
    ) -> None:
        """Wrapper extends `_fit_wrap`, adds construction of ANN index by flag.

        Args:
            dataset: historical interactions with query/item features
                ``[user_id, item_id, timestamp, rating]``
        """
        super()._fit_wrap(dataset)

        if self._use_ann:
            vectors = self._get_vectors_to_build_ann(dataset.interactions)
            ann_params = self._get_ann_build_params(dataset.interactions)
            self.index_builder.build_index(vectors, **ann_params)

    @abstractmethod
    def _get_vectors_to_infer_ann_inner(self, interactions: SparkDataFrame, queries: SparkDataFrame) -> SparkDataFrame:
        """Implementations of this method must return a dataframe with user vectors.
        User vectors from this method are used to infer the index.

        Args:
            interactions: DataFrame with interactions
            queries: DataFrame with queries

        Returns: DataFrame[user_idx int, vector array<double>] or DataFrame[vector array<double>].
        Vector column name in dataframe can be anything.
        """

    def _get_vectors_to_infer_ann(
        self, interactions: SparkDataFrame, queries: SparkDataFrame, filter_seen_items: bool
    ) -> SparkDataFrame:
        """This method wraps `_get_vectors_to_infer_ann_inner`
        and adds seen items to dataframe with user vectors by flag.

        Args:
            interactions: DataFrame with interactions
            queries: DataFrame with queries
            filter_seen_items: flag to remove seen items from recommendations based on ``interactions``.

        Returns:

        """
        queries = self._get_vectors_to_infer_ann_inner(interactions, queries)

        # here we add `seen_item_idxs` to filter the viewed items in UDFs (see infer_index_udf)
        if filter_seen_items:
            user_to_max_items = interactions.groupBy(self.query_column).agg(
                sf.count(self.item_column).alias("num_items"),
                sf.collect_set(self.item_column).alias("seen_item_idxs"),
            )
            queries = queries.join(user_to_max_items, on=self.query_column)

        return queries

    @abstractmethod
    def _get_ann_infer_params(self) -> Dict[str, Any]:
        """Implementation of this method must return dictionary
        with arguments for `_infer_ann_index` method.

        Returns: Dictionary with arguments to infer index. For example: {
            "features_col": "user_vector",
            ...
        }

        """

    def _predict_wrap(
        self,
        dataset: Optional[Dataset],
        k: int,
        queries: Optional[Union[SparkDataFrame, Iterable]] = None,
        items: Optional[Union[SparkDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[SparkDataFrame]:
        dataset, queries, items = self._filter_interactions_queries_items_dataframes(dataset, k, queries, items)

        if self._use_ann:
            vectors = self._get_vectors_to_infer_ann(dataset.interactions, queries, filter_seen_items)
            ann_params = self._get_ann_infer_params()
            inferer = self.index_builder.produce_inferer(filter_seen_items)
            recs = inferer.infer(vectors, ann_params["features_col"], k)
        else:
            recs = self._predict(
                dataset,
                k,
                queries,
                items,
                filter_seen_items,
            )

        if not self._use_ann:
            if filter_seen_items and dataset.interactions:
                recs = self._filter_seen(recs=recs, interactions=dataset.interactions, queries=queries, k=k)

            recs = get_top_k_recs(recs, k=k, query_column=self.query_column, rating_column=self.rating_column).select(
                self.query_column, self.item_column, self.rating_column
            )

        output = return_recs(recs, recs_file_path)
        self._clear_model_temp_view("filter_seen_queries_interactions")
        self._clear_model_temp_view("filter_seen_num_seen")
        return output

    def _save_index(self, path):
        self.index_builder.index_store.dump_index(path)

    def _load_index(self, path: str):
        self.index_builder.index_store = SparkFilesIndexStore()
        self.index_builder.index_store.load_from_path(path)

    def init_builder_from_dict(self, init_meta: dict):
        """Inits an index builder instance from a dict with init meta."""

        # index param entity instance initialization
        module = importlib.import_module(init_meta["index_param"]["module"])
        class_ = getattr(module, init_meta["index_param"]["class"])
        index_params = class_(**init_meta["index_param"]["init_args"])

        # index builder instance initialization
        module = importlib.import_module(init_meta["builder"]["module"])
        class_ = getattr(module, init_meta["builder"]["class"])
        index_builder = class_(index_params=index_params, index_store=None)

        self.index_builder = index_builder
