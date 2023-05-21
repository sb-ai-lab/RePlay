import logging
from typing import Optional, Iterator

import pandas as pd
from pyspark.sql import DataFrame
from scipy.sparse import csr_matrix

from replay.ann.index_builders.base_index_builder import IndexBuilder
from replay.ann.index_inferers.base_inferer import IndexInferer
from replay.ann.index_inferers.nmslib_filter_index_inferer import (
    NmslibFilterIndexInferer,
)
from replay.ann.index_inferers.nmslib_index_inferer import NmslibIndexInferer
from replay.ann.utils import (
    create_nmslib_index_instance,
)

logger = logging.getLogger("replay")


class ExecutorNmslibIndexBuilder(IndexBuilder):
    """
    Builder that build nmslib hnsw index on one executor.
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
            "M": self.index_params.M,
            "efConstruction": self.index_params.efC,
            "post": self.index_params.post,
        }

        # to execution in one executor
        vectors = vectors.repartition(1)

        _index_store = self.index_store
        _index_params = self.index_params

        def build_index_udf(iterator: Iterator[pd.DataFrame]):
            """Builds index on executor and writes it to shared disk or hdfs.

            Args:
                iterator: iterates on dataframes with vectors/features

            """
            index = create_nmslib_index_instance(_index_params)

            pdfs = []
            for pdf in iterator:
                pdfs.append(pdf)

            pdf = pd.concat(pdfs, copy=False)

            # We collect all iterator values into one dataframe,
            # because we cannot guarantee that `pdf` will contain rows
            # with the same `item_idx_two`.
            # And therefore we cannot call the `addDataPointBatch` iteratively.
            data = pdf["similarity"].values
            row_ind = pdf["item_idx_two"].values
            col_ind = pdf["item_idx_one"].values

            sim_matrix_tmp = csr_matrix(
                (data, (row_ind, col_ind)),
                shape=(
                    self.index_params.items_count,
                    self.index_params.items_count,
                ),
            )
            index.addDataPointBatch(data=sim_matrix_tmp)
            index.createIndex(index_params)

            _index_store.save_to_store(
                lambda path: index.saveIndex(path, save_data=True)
            )  # pylint: disable=unnecessary-lambda)

            yield pd.DataFrame(data={"_success": 1}, index=[0])

        # Here we perform materialization (`.collect()`) to build the hnsw index.
        vectors.select(
            "similarity", "item_idx_one", "item_idx_two"
        ).mapInPandas(build_index_udf, "_success int").collect()
