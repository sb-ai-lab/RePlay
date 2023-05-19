import logging
import os
import tempfile
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from scipy.sparse import csr_matrix

from replay.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.ann.index_builders.base_hnsw_index_builder import (
    BaseHnswIndexBuilder,
)
from replay.ann.utils import init_nmslib_index
from replay.utils import FileInfo, FileSystem

logger = logging.getLogger("replay")


class DriverNmslibIndexBuilder(BaseHnswIndexBuilder):
    """
    Builder that build nmslib hnsw index on driver
    and sends it to executors through `SparkContext.addFile()`
    """

    def __init__(self, index_file_name: str):
        self._index_file_name = index_file_name

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

        vectors = vectors.toPandas()

        index = init_nmslib_index(params)

        data = vectors["similarity"].values
        row_ind = vectors["item_idx_two"].values
        col_ind = vectors["item_idx_one"].values

        sim_matrix = csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(params.items_count, params.items_count),
        )
        index.addDataPointBatch(data=sim_matrix)
        index.createIndex(index_params)

        # saving index to local temp file and sending it to executors
        temp_dir = tempfile.mkdtemp()
        tmp_file_path = os.path.join(temp_dir, self._index_file_name)
        # save_data=True https://github.com/nmslib/nmslib/issues/300
        index.saveIndex(tmp_file_path, save_data=True)
        spark = SparkSession.getActiveSession()
        # for the "sparse" type we need to store two files
        spark.sparkContext.addFile("file://" + tmp_file_path)
        spark.sparkContext.addFile("file://" + tmp_file_path + ".dat")

        return FileInfo(path=temp_dir, filesystem=FileSystem.LOCAL)
