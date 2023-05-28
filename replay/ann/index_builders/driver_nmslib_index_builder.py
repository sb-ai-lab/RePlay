import logging
from typing import Optional

from pyspark.sql import DataFrame
from scipy.sparse import csr_matrix

from replay.ann.index_builders.base_index_builder import IndexBuilder
from replay.ann.index_inferers.base_inferer import IndexInferer
from replay.ann.index_inferers.nmslib_filter_index_inferer import (
    NmslibFilterIndexInferer,
)
from replay.ann.index_inferers.nmslib_index_inferer import NmslibIndexInferer
from replay.ann.utils import create_nmslib_index_instance

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
        index_params = {
            "M": self.index_params.m,
            "efConstruction": self.index_params.ef_c,
            "post": self.index_params.post,
        }

        vectors = vectors.toPandas()

        index = create_nmslib_index_instance(self.index_params)

        data = vectors["similarity"].values
        row_ind = vectors["item_idx_two"].values
        col_ind = vectors["item_idx_one"].values

        sim_matrix = csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(
                self.index_params.items_count,
                self.index_params.items_count,
            ),
        )
        index.addDataPointBatch(data=sim_matrix)
        index.createIndex(index_params)

        self.index_store.save_to_store(
            lambda path: index.saveIndex(
                path, save_data=True
            )  # pylint: disable=unnecessary-lambda)
        )
