import logging
from typing import Optional, Iterator

import pandas as pd
from pyspark.sql import DataFrame
from scipy.sparse import csr_matrix

from replay.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.ann.index_builders.base_hnsw_index_builder import (
    BaseHnswIndexBuilder,
)
from replay.ann.utils import save_index_to_destination_fs, init_nmslib_index
from replay.utils import get_filesystem

logger = logging.getLogger("replay")


class ExecutorNmslibIndexBuilder(BaseHnswIndexBuilder):
    """
    Builder that build nmslib hnsw index on one executor
    and save it to hdfs or shared disk.
    """

    def _build_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: NmslibHnswParam,
        id_col: Optional[str] = None,
    ):
        index_params = {
            "M": params.M,
            "efConstruction": params.efC,
            "post": params.post,
        }

        # to execution in one executor
        vectors = vectors.repartition(1)

        target_index_file = get_filesystem(params.index_path)

        def build_index_udf(iterator: Iterator[pd.DataFrame]):
            """Builds index on executor and writes it to shared disk or hdfs.

            Args:
                iterator: iterates on dataframes with vectors/features

            """
            index = init_nmslib_index(params)

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
                shape=(params.items_count, params.items_count),
            )
            index.addDataPointBatch(data=sim_matrix_tmp)
            index.createIndex(index_params)

            save_index_to_destination_fs(
                sparse=True,
                save_index=lambda path: index.saveIndex(path, save_data=True),
                target=target_index_file,
            )

            yield pd.DataFrame(data={"_success": 1}, index=[0])

        # Here we perform materialization (`.collect()`) to build the hnsw index.
        vectors.select(
            "similarity", "item_idx_one", "item_idx_two"
        ).mapInPandas(build_index_udf, "_success int").collect()

        # return target_index_file
        return None
