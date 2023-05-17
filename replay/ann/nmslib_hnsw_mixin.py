import logging
import os
import shutil
import tempfile
import weakref
from abc import ABC
from typing import Optional

import numpy as np
import pandas as pd
from pyarrow import fs
from pyspark import SparkFiles
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from scipy.sparse import csr_matrix

from replay.ann.ann_mixin import ANNMixin
from replay.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.ann.index_builders.driver_nmslib_index_builder import DriverNmslibIndexBuilder
from replay.ann.index_builders.executor_nmslib_index_builder import ExecutorNmslibIndexBuilder
from replay.ann.index_file_managers.nmslib_index_file_manager import NmslibIndexFileManager
from replay.session_handler import State
from replay.utils import FileSystem, get_filesystem

logger = logging.getLogger("replay")

INDEX_FILENAME = "nmslib_hnsw_index"


class NmslibHnswMixin(ANNMixin, ABC):
    """Mixin that provides methods to build nmslib hnsw index and infer it.
    Also provides methods to saving and loading index to/from disk.
    """

    _nmslib_hnsw_params: Optional[NmslibHnswParam] = None

    def _infer_ann_index(  # pylint: disable=too-many-arguments
        self,
        vectors: DataFrame,
        features_col: str,
        params: NmslibHnswParam,
        k: int,
        filter_seen_items: bool,
    ) -> DataFrame:
        return self._infer_nmslib_hnsw_index(
            vectors,
            params,
            k,
            filter_seen_items,
        )

    def _build_ann_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: NmslibHnswParam,
        id_col: Optional[str] = None,
    ) -> None:
        """Builds hnsw index and dump it to hdfs or disk.

        Args:
            vectors (DataFrame): DataFrame with item vectors to build index.
            params (Dict[str, Any]): index params
        """
        if params.build_index_on == "executor":
            index_builder = ExecutorNmslibIndexBuilder()
        else:
            self.__dict__.pop('_spark_index_file_uid', None)
            index_builder = DriverNmslibIndexBuilder(
                index_file_name=f"{INDEX_FILENAME}_{self._spark_index_file_uid}"
            )
        temp_index_file_info = index_builder.build_index(
            vectors, features_col, params, id_col
        )
        if temp_index_file_info:
            weakref.finalize(self, shutil.rmtree, temp_index_file_info.path)

    def _infer_nmslib_hnsw_index(  # pylint: disable=too-many-locals
        self,
        user_vectors: DataFrame,
        params: NmslibHnswParam,
        k: int,
        filter_seen_items: bool,
    ) -> DataFrame:

        if params.build_index_on == "executor":
            index_file = get_filesystem(params.index_path)
        else:
            index_file = f"{INDEX_FILENAME}_{self._spark_index_file_uid}"

        _index_file_manager = NmslibIndexFileManager(
            params,
            index_file=index_file,
        )

        index_file_manager_broadcast = State().session.sparkContext.broadcast(
            _index_file_manager
        )

        return_type = "item_idx array<int>, distance array<double>"

        def get_csr_matrix(
                user_idx: pd.Series,
                vector_items: pd.Series,
                vector_relevances: pd.Series,
        ) -> csr_matrix:

            return csr_matrix(
                (
                    vector_relevances.explode().values.astype(float),
                    (user_idx.repeat(vector_items.apply(lambda x: len(x))).values,  # pylint: disable=unnecessary-lambda
                     vector_items.explode().values.astype(int)),
                ),
                shape=(user_idx.max() + 1, vector_items.apply(lambda x: max(x)).max() + 1),  # pylint: disable=unnecessary-lambda
            )

        if filter_seen_items:

            @pandas_udf(return_type)
            def infer_index(  # pylint: disable=too-many-locals
                user_idx: pd.Series,
                vector_items: pd.Series,
                vector_relevances: pd.Series,
                num_items: pd.Series,
                seen_item_idxs: pd.Series,
            ) -> pd.DataFrame:
                index_file_manager = index_file_manager_broadcast.value
                index = index_file_manager.index

                # max number of items to retrieve per batch
                max_items_to_retrieve = num_items.max()

                user_vectors = get_csr_matrix(user_idx, vector_items, vector_relevances)

                neighbours = index.knnQueryBatch(
                    user_vectors[user_idx.values, :],
                    k=k + max_items_to_retrieve,
                    num_threads=1
                )

                neighbours_filtered = []
                for i, (item_idxs, distances) in enumerate(neighbours):
                    non_seen_item_indexes = ~np.isin(
                        item_idxs, seen_item_idxs[i], assume_unique=True
                    )
                    neighbours_filtered.append(
                        (
                            (item_idxs[non_seen_item_indexes])[:k],
                            (distances[non_seen_item_indexes])[:k],
                        )
                    )

                pd_res = pd.DataFrame(
                    neighbours_filtered, columns=["item_idx", "distance"]
                )

                # pd_res looks like
                # item_idx       distances
                # [1, 2, 3, ...] [-0.5, -0.3, -0.1, ...]
                # [1, 3, 4, ...] [-0.1, -0.8, -0.2, ...]

                return pd_res

        else:

            @pandas_udf(return_type)
            def infer_index(user_idx: pd.Series,
                            vector_items: pd.Series,
                            vector_relevances: pd.Series,) -> pd.DataFrame:

                index_file_manager = index_file_manager_broadcast.value
                index = index_file_manager.index

                user_vectors = get_csr_matrix(user_idx, vector_items, vector_relevances)
                neighbours = index.knnQueryBatch(
                    user_vectors[user_idx.values, :],
                    num_threads=1
                )

                pd_res = pd.DataFrame(
                    neighbours, columns=["item_idx", "distance"]
                )

                # pd_res looks like
                # item_idx       distances
                # [1, 2, 3, ...] [-0.5, -0.3, -0.1, ...]
                # [1, 3, 4, ...] [-0.1, -0.8, -0.2, ...]

                return pd_res

        cols = ["user_idx", "vector_items", "vector_relevances"]
        if filter_seen_items:
            cols = cols + ["num_items", "seen_item_idxs"]

        res = user_vectors.select(
            "user_idx",
            infer_index(*cols).alias("neighbours"),
        )

        res = self._unpack_infer_struct(res)

        return res

    def _save_nmslib_hnsw_index(self, path, sparse=False):
        """Method save (copy) index from hdfs (or local) to `path` directory.
        `path` can be an hdfs path or a local path.

        Args:
            path (_type_): directory where to dump (copy) the index
        """

        params = self._nmslib_hnsw_params

        if params.build_index_on == "executor":
            index_path = params.index_path
        elif params.build_index_on == "driver":
            index_path = SparkFiles.get(
                f"{INDEX_FILENAME}_{self._spark_index_file_uid}"
            )
        else:
            raise ValueError("Unknown 'build_index_on' param.")

        source = get_filesystem(index_path)
        target = get_filesystem(path)
        self.logger.debug("Index file coping from '%s' to '%s'", index_path, path)

        source_paths = []
        target_paths = []
        if sparse:
            source_paths.append(source.path)
            source_paths.append(source.path + ".dat")
            index_file_target_path = os.path.join(target.path, INDEX_FILENAME)
            target_paths.append(index_file_target_path)
            target_paths.append(index_file_target_path + ".dat")
        else:
            source_paths.append(source.path)
            index_file_target_path = os.path.join(target.path, INDEX_FILENAME)
            target_paths.append(index_file_target_path)

        if source.filesystem == FileSystem.HDFS:
            source_filesystem = fs.HadoopFileSystem.from_uri(source.hdfs_uri)
        else:
            source_filesystem = fs.LocalFileSystem()
        if target.filesystem == FileSystem.HDFS:
            destination_filesystem = fs.HadoopFileSystem.from_uri(
                target.hdfs_uri
            )
        else:
            destination_filesystem = fs.LocalFileSystem()

        for source_path, target_path in zip(source_paths, target_paths):
            fs.copy_files(
                source_path,
                target_path,
                source_filesystem=source_filesystem,
                destination_filesystem=destination_filesystem,
            )
            # param use_threads=True (?)

    def _load_nmslib_hnsw_index(self, path: str, sparse=False):
        """Loads hnsw index from `path` directory to local dir.
        Index file name is 'hnswlib_index'.
        And adds index file to the `SparkFiles`.
        `path` can be a hdfs path or a local path.


        Args:
            path: directory path, where index file is stored
        """
        source = get_filesystem(path + f"/{INDEX_FILENAME}")

        temp_dir = tempfile.mkdtemp()
        weakref.finalize(self, shutil.rmtree, temp_dir)
        target_path = os.path.join(
            temp_dir, f"{INDEX_FILENAME}_{self._spark_index_file_uid}"
        )

        source_paths = []
        target_paths = []
        if sparse:
            source_paths.append(source.path)
            source_paths.append(source.path + ".dat")
            target_paths.append(target_path)
            target_paths.append(target_path + ".dat")
        else:
            source_paths.append(source.path)
            target_paths.append(target_path)

        if source.filesystem == FileSystem.HDFS:
            source_filesystem = fs.HadoopFileSystem.from_uri(source.hdfs_uri)
        else:
            source_filesystem = fs.LocalFileSystem()

        destination_filesystem = fs.LocalFileSystem()

        for source_path, target_path in zip(source_paths, target_paths):
            fs.copy_files(
                source_path,
                target_path,
                source_filesystem=source_filesystem,
                destination_filesystem=destination_filesystem,
            )

        spark = SparkSession.getActiveSession()
        for target_path in target_paths:
            spark.sparkContext.addFile("file://" + target_path)

        self._nmslib_hnsw_params.build_index_on = "driver"
