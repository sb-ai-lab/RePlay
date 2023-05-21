from abc import ABC, abstractmethod
from typing import Optional

from pyspark.sql import DataFrame

from replay.ann.entities.base_hnsw_param import BaseHnswParam
from replay.ann.index_inferers.base_inferer import IndexInferer
from replay.ann.index_stores.base_index_store import IndexStore


class IndexBuilder(ABC):
    def __init__(self, index_params: BaseHnswParam, index_store: IndexStore):
        self.index_store = index_store
        self.index_params = index_params

    @abstractmethod
    def build_index(
        self,
        vectors: DataFrame,
        features_col: str,
        ids_col: Optional[str] = None,
    ):
        """
        Method that builds index and stores it using the `IndexStore` class.

        :param vectors: DataFrame with vectors to build index. Schema: [{ids_col}: int, {features_col}: array<float>]
            or [{features_col}: array<float>]
        :param features_col: Name of column from `vectors` dataframe
            that contains vectors to build index
        :param ids_col: Name of column that contains identifiers of vectors.
            None if `vectors` dataframe have no id column.
        """

    @abstractmethod
    def produce_inferer(self, filter_seen_items: bool) -> IndexInferer:
        """Method that produce `IndexInferer`."""
