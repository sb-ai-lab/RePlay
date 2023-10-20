import logging
from typing import Optional

import numpy as np
from pyspark.sql import DataFrame

from replay.models.extensions.ann.index_inferers import HnswlibFilterIndexInferer, HnswlibIndexInferer, IndexInferer
from replay.models.extensions.ann.utils import create_hnswlib_index_instance

from .base_index_builder import IndexBuilder

logger = logging.getLogger("replay")


class DriverHnswlibIndexBuilder(IndexBuilder):
    """
    Builder that builds hnswlib index on driver.
    """

    def produce_inferer(self, filter_seen_items: bool) -> IndexInferer:
        if filter_seen_items:
            return HnswlibFilterIndexInferer(self.index_params, self.index_store)
        else:
            return HnswlibIndexInferer(self.index_params, self.index_store)

    def build_index(
        self,
        vectors: DataFrame,
        features_col: str,
        ids_col: Optional[str] = None,
    ):
        vectors = vectors.toPandas()
        vectors_np = np.squeeze(vectors[features_col].values)

        index = create_hnswlib_index_instance(self.index_params, init=True)

        if ids_col:
            index.add_items(np.stack(vectors_np), vectors[ids_col].values)
        else:
            index.add_items(np.stack(vectors_np))

        self.index_store.save_to_store(lambda path: index.save_index(path))  # pylint: disable=unnecessary-lambda)
