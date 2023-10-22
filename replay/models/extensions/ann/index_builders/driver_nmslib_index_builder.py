import logging
from typing import Optional

from pyspark.sql import DataFrame

from . import IndexBuilder, NmslibIndexBuilderMixin
from replay.models.extensions.ann.index_inferers import (
    IndexInferer,
    NmslibFilterIndexInferer,
    NmslibIndexInferer,
)

logger = logging.getLogger("replay")


class DriverNmslibIndexBuilder(IndexBuilder):
    """
    Builder that builds nmslib hnsw index on driver.
    """

    def produce_inferer(self, filter_seen_items: bool) -> IndexInferer:
        if filter_seen_items:
            return NmslibFilterIndexInferer(
                self.index_params, self.index_store
            )
        else:
            return NmslibIndexInferer(self.index_params, self.index_store)

    def build_index(
        self,
        vectors: DataFrame,
        features_col: str,
        ids_col: Optional[str] = None,
    ):
        vectors = vectors.toPandas()

        NmslibIndexBuilderMixin.build_and_save_index(
            vectors, self.index_params, self.index_store
        )
