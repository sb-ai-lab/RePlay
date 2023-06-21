import logging
from typing import Optional, Iterator

import pandas as pd
from pyspark.sql import DataFrame
from scipy.sparse import csr_matrix

from replay.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.ann.index_builders.base_index_builder import IndexBuilder
from replay.ann.index_inferers.base_inferer import IndexInferer
from replay.ann.index_inferers.nmslib_filter_index_inferer import (
    NmslibFilterIndexInferer,
)
from replay.ann.index_inferers.nmslib_index_inferer import NmslibIndexInferer
from replay.ann.index_stores.base_index_store import IndexStore
from replay.ann.utils import (
    create_nmslib_index_instance,
)

logger = logging.getLogger("replay")


def make_build_index_udf(
    index_params: NmslibHnswParam, index_store: IndexStore
):
    """
    Method returns udf to build nmslib index.
    This function is implemented as a builder function to be able to test
    the build_index_udf internal function outside of spark,
    because pytest does not see this function call if the function is called by spark.
    :param index_params: index parameters as instance of NmslibHnswParam.
    :param index_store: index store
    :return: `build_index_udf`
    """
    creation_index_params = {
        "M": index_params.m,
        "efConstruction": index_params.ef_c,
        "post": index_params.post,
    }

    def build_index_udf(iterator: Iterator[pd.DataFrame]):
        """Builds index on executor and writes it to shared disk or hdfs.

        Args:
            iterator: iterates on dataframes with vectors/features

        """
        index = create_nmslib_index_instance(index_params)

        pdfs = []
        for pdf in iterator:
            pdfs.append(pdf)

        pdf = pd.concat(pdfs)

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
                index_params.items_count,
                index_params.items_count,
            ),
        )
        index.addDataPointBatch(data=sim_matrix_tmp)
        index.createIndex(creation_index_params)

        index_store.save_to_store(
            lambda path: index.saveIndex(path, save_data=True)
        )  # pylint: disable=unnecessary-lambda)

        yield pd.DataFrame(data={"_success": 1}, index=[0])

    return build_index_udf


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
        # to execution in one executor
        vectors = vectors.repartition(1)

        # this function will build the index
        build_index_udf = make_build_index_udf(
            self.index_params, self.index_store
        )

        # Here we perform materialization (`.collect()`) to build the hnsw index.
        vectors.select(
            "similarity", "item_idx_one", "item_idx_two"
        ).mapInPandas(build_index_udf, "_success int").collect()
