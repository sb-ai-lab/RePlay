import logging
from typing import Optional

import numpy as np

from .base_index_builder import IndexBuilder
from replay.models.extensions.ann.index_inferers.base_inferer import IndexInferer
from replay.models.extensions.ann.index_inferers.hnswlib_filter_index_inferer import HnswlibFilterIndexInferer
from replay.models.extensions.ann.index_inferers.hnswlib_index_inferer import HnswlibIndexInferer
from replay.models.extensions.ann.utils import create_hnswlib_index_instance
from replay.utils import SparkDataFrame
from replay.utils.spark_utils import spark_to_pandas

logger = logging.getLogger("replay")


class DriverHnswlibIndexBuilder(IndexBuilder):
    """
    Builder that builds hnswlib index on driver.
    """

    def produce_inferer(self, filter_seen_items: bool) -> IndexInferer:
        if filter_seen_items:
            return HnswlibFilterIndexInferer(
                self.index_params, self.index_store
            )
        else:
            return HnswlibIndexInferer(self.index_params, self.index_store)

    # pylint: disable=no-member
    def build_index(
        self,
        vectors: SparkDataFrame,
        features_col: str,
        ids_col: Optional[str] = None,
    ):
        vectors = spark_to_pandas(vectors, self.allow_collect_to_master)
        vectors_np = np.squeeze(vectors[features_col].values)
        index = create_hnswlib_index_instance(self.index_params, init=True)

        if ids_col:
            index.add_items(np.stack(vectors_np), vectors[ids_col].values)
        else:
            index.add_items(np.stack(vectors_np))

        self.index_store.save_to_store(
            lambda path: index.save_index(  # pylint: disable=unnecessary-lambda)
                path
            )
        )
