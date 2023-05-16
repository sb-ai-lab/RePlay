import logging
from typing import Optional, Iterator

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

from replay.ann.entities.hnswlib_param import HnswlibParam
from replay.ann.index_builders.base_hnsw_index_builder import (
    BaseHnswIndexBuilder,
)
from replay.ann.utils import save_index_to_destination_fs, init_hnswlib_index
from replay.utils import get_filesystem

logger = logging.getLogger("replay")


class ExecutorHnswlibIndexBuilder(BaseHnswIndexBuilder):
    """
    Builder that build hnswlib index on one executor
    and save it to hdfs or shared disk.
    """

    def _build_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: HnswlibParam,
        id_col: Optional[str] = None,
    ):
        # to execution in one executor
        vectors = vectors.repartition(1)

        target_index_file = get_filesystem(params.index_path)

        def build_index_udf(iterator: Iterator[pd.DataFrame]):
            """Builds index on executor and writes it to shared disk or hdfs.

            Args:
                iterator: iterates on dataframes with vectors/features

            """
            index = init_hnswlib_index(params)

            # pdf is a pandas dataframe that contains ids and features (vectors)
            for pdf in iterator:
                vectors_np = np.squeeze(pdf[features_col].values)
                if id_col:
                    index.add_items(np.stack(vectors_np), pdf[id_col].values)
                else:
                    # ids will be from [0, ..., len(vectors_np)]
                    index.add_items(np.stack(vectors_np))

            save_index_to_destination_fs(
                sparse=False,
                save_index=lambda path: index.save_index(
                    path
                ),  # pylint: disable=unnecessary-lambda
                target=target_index_file,
            )

            yield pd.DataFrame(data={"_success": 1}, index=[0])

        # Here we perform materialization (`.collect()`) to build the hnsw index.
        cols = [id_col, features_col] if id_col else [features_col]
        vectors.select(*cols).mapInPandas(
            build_index_udf, "_success int"
        ).collect()

        # return target_index_file
        return None
