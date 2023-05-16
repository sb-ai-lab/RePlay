import logging
import os
import tempfile
from typing import Optional

import numpy as np
from pyspark.sql import DataFrame, SparkSession

from replay.ann.entities.hnswlib_param import HnswlibParam
from replay.ann.index_builders.base_hnsw_index_builder import (
    BaseHnswIndexBuilder,
)
from replay.ann.utils import init_hnswlib_index
from replay.utils import FileInfo, FileSystem

logger = logging.getLogger("replay")


class DriverHnswlibIndexBuilder(BaseHnswIndexBuilder):
    """
    Builder that build hnswlib index on driver
    and sends it to executors through `SparkContext.addFile()`
    """

    def __init__(self, index_file_name: str):
        self._index_file_name = index_file_name

    def _build_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: HnswlibParam,
        id_col: Optional[str] = None,
    ):
        vectors = vectors.toPandas()
        vectors_np = np.squeeze(vectors[features_col].values)

        index = init_hnswlib_index(params)

        if id_col:
            index.add_items(np.stack(vectors_np), vectors[id_col].values)
        else:
            index.add_items(np.stack(vectors_np))

        # saving index to local temp file and sending it to executors
        temp_dir = tempfile.mkdtemp()
        tmp_file_path = os.path.join(temp_dir, self._index_file_name)
        index.save_index(tmp_file_path)
        spark = SparkSession.getActiveSession()
        spark.sparkContext.addFile("file://" + tmp_file_path)

        return FileInfo(path=temp_dir, filesystem=FileSystem.LOCAL)
