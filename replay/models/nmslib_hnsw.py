import logging
import os
import shutil
import tempfile
import weakref
from typing import Any, Dict, Optional, Iterator, Union

import nmslib
import numpy as np
import pandas as pd
from pyarrow import fs
from pyspark import SparkFiles
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from scipy.sparse import csr_matrix

from replay.ann.ann_mixin import ANNMixin
from replay.ann.utils import (
    save_index_to_destination_fs,
    load_index_from_source_fs,
)
from replay.session_handler import State
from replay.utils import FileSystem, get_filesystem, FileInfo

logger = logging.getLogger("replay")

INDEX_FILENAME = "nmslib_hnsw_index"


class NmslibIndexFileManager:
    """Loads index from hdfs, local disk or SparkFiles dir and keep it in a memory.
    Instance of `NmslibIndexFileManager` broadcasts to executors and is used in pandas_udf.
    """

    def __init__(
        self,
        index_params,
        index_type: str,
        index_file: Union[FileInfo, str]
    ) -> None:

        self._method = index_params["method"]
        self._space = index_params["space"]
        self._efS = index_params.get("efS")
        self._index_type = index_type
        self._index_file = index_file
        self._index = None

    @property
    def index(self):
        if self._index:
            return self._index

        if self._index_type == "sparse":
            self._index = nmslib.init(
                method=self._method,
                space=self._space,
                data_type=nmslib.DataType.SPARSE_VECTOR,
            )
            if isinstance(self._index_file, FileInfo):
                load_index_from_source_fs(
                    sparse=True,
                    load_index=lambda path: self._index.loadIndex(
                        path, load_data=True
                    ),
                    source=self._index_file
                )
            else:
                self._index.loadIndex(
                    SparkFiles.get(self._index_file), load_data=True
                )
        else:
            self._index = nmslib.init(
                method=self._method,
                space=self._space,
                data_type=nmslib.DataType.DENSE_VECTOR,
            )
            if isinstance(self._index_file, FileInfo):
                load_index_from_source_fs(
                    sparse=False,
                    load_index=lambda path: self._index.loadIndex(path),
                    source=self._index_file
                )
            else:
                self._index.loadIndex(SparkFiles.get(self._index_file))

        if self._efS:
            self._index.setQueryTimeParams({"efSearch": self._efS})
        return self._index


class NmslibHnswMixin(ANNMixin):
    """Mixin that provides methods to build nmslib hnsw index and infer it.
    Also provides methods to saving and loading index to/from disk.
    """

    def _infer_ann_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: Dict[str, Union[int, str]],
        k: int,
        filter_seen_items: bool,
        index_dim: str = None,
        index_type: str = None,
        log: DataFrame = None,
    ) -> DataFrame:
        return self._infer_nmslib_hnsw_index(
            vectors,
            features_col,
            params,
            k,
            filter_seen_items,
            index_type,
            log,
        )

    def _build_ann_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: Dict[str, Union[int, str]],
        dim: int = None,
        num_elements: int = None,
        id_col: Optional[str] = None,
        index_type: str = None,
        items_count: Optional[int] = None,
    ) -> None:
        self._build_nmslib_hnsw_index(
            vectors, features_col, params, index_type, items_count
        )

    def _build_nmslib_hnsw_index(
        self,
        item_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        index_type: str = None,
        items_count: Optional[int] = None,
    ) -> None:
        """Builds hnsw index and dump it to hdfs or disk.

        Args:
            item_vectors (DataFrame): DataFrame with item vectors
            params (Dict[str, Any]): hnsw params
        """

        index_params = {
            "M": params.get("M", 200),
            "efConstruction": params.get("efC", 20000),
            "post": params.get("post", 2000),
        }

        if params["build_index_on"] == "executor":
            # to execution in one executor
            item_vectors = item_vectors.repartition(1)

            target_index_file = get_filesystem(params["index_path"])

            if index_type == "sparse":

                def build_index(iterator: Iterator[pd.DataFrame]):
                    """Builds index on executor and writes it to shared disk or hdfs.

                    Args:
                        iterator: iterates on dataframes with vectors/features

                    """
                    index = nmslib.init(
                        method=params["method"],
                        space=params["space"],
                        data_type=nmslib.DataType.SPARSE_VECTOR,
                    )

                    pdfs = []
                    for pdf in iterator:
                        pdfs.append(pdf)

                    pdf = pd.concat(pdfs, copy=False)

                    # We collect all iterator values into one dataframe,
                    # because we cannot guarantee that `pdf` will contain rows with the same `item_idx_two`.
                    # And therefore we cannot call the `addDataPointBatch` iteratively.
                    data = pdf["similarity"].values
                    row_ind = pdf["item_idx_two"].values
                    col_ind = pdf["item_idx_one"].values

                    sim_matrix_tmp = csr_matrix(
                        (data, (row_ind, col_ind)),
                        shape=(items_count, items_count),
                    )
                    index.addDataPointBatch(data=sim_matrix_tmp)
                    index.createIndex(index_params)

                    save_index_to_destination_fs(
                        sparse=True,
                        save_index=lambda path: index.saveIndex(
                            path, save_data=True
                        ),
                        target=target_index_file,
                    )

                    yield pd.DataFrame(data={"_success": 1}, index=[0])

            else:

                def build_index(iterator: Iterator[pd.DataFrame]):
                    """Builds index on executor and writes it to shared disk or hdfs.

                    Args:
                        iterator: iterates on dataframes with vectors/features

                    """
                    index = nmslib.init(
                        method=params["method"],
                        space=params["space"],
                        data_type=nmslib.DataType.DENSE_VECTOR,
                    )
                    for pdf in iterator:
                        item_vectors_np = np.squeeze(pdf[features_col].values)
                        index.addDataPointBatch(
                            data=np.stack(item_vectors_np),
                            ids=pdf["item_idx"].values,
                        )
                    index.createIndex(index_params)

                    save_index_to_destination_fs(
                        sparse=False,
                        save_index=lambda path: index.saveIndex(path),
                        target=target_index_file,
                    )

                    yield pd.DataFrame(data={"_success": 1}, index=[0])

            # Here we perform materialization (`.collect()`) to build the hnsw index.
            logger.info("Started building the hnsw index")
            if index_type == "sparse":
                item_vectors.select(
                    "similarity", "item_idx_one", "item_idx_two"
                ).mapInPandas(build_index, "_success int").collect()
            else:
                item_vectors.select("item_idx", features_col).mapInPandas(
                    build_index, "_success int"
                ).collect()
            logger.info("Finished building the hnsw index")
        else:
            if index_type == "sparse":
                item_vectors = item_vectors.toPandas()

                index = nmslib.init(
                    method=params["method"],
                    space=params["space"],
                    data_type=nmslib.DataType.SPARSE_VECTOR,
                )

                data = item_vectors["similarity"].values
                row_ind = item_vectors["item_idx_two"].values
                col_ind = item_vectors["item_idx_one"].values

                sim_matrix = csr_matrix(
                    (data, (row_ind, col_ind)),
                    shape=(items_count, items_count),
                )
                index.addDataPointBatch(data=sim_matrix)
                index.createIndex(index_params)

                # saving index to local temp file and sending it to executors
                temp_dir = tempfile.mkdtemp()
                weakref.finalize(self, shutil.rmtree, temp_dir)
                self.__dict__.pop('_spark_index_file_uid', None)
                tmp_file_path = os.path.join(
                    temp_dir, f"{INDEX_FILENAME}_{self._spark_index_file_uid}"
                )
                index.saveIndex(tmp_file_path, save_data=True)
                spark = SparkSession.getActiveSession()
                # for the "sparse" type we need to store two files
                spark.sparkContext.addFile("file://" + tmp_file_path)
                spark.sparkContext.addFile("file://" + tmp_file_path + ".dat")

            else:
                item_vectors = item_vectors.toPandas()
                item_vectors_np = np.squeeze(item_vectors[features_col].values)
                index = nmslib.init(
                    method=params["method"],
                    space=params["space"],
                    data_type=nmslib.DataType.DENSE_VECTOR,
                )
                index.addDataPointBatch(
                    data=np.stack(item_vectors_np),
                    ids=item_vectors["item_idx"].values,
                )
                index.createIndex(index_params)

                # saving index to local temp file and sending it to executors
                temp_dir = tempfile.mkdtemp()
                weakref.finalize(self, shutil.rmtree, temp_dir)
                tmp_file_path = os.path.join(
                    temp_dir, f"{INDEX_FILENAME}_{self._spark_index_file_uid}"
                )
                index.saveIndex(tmp_file_path)
                spark = SparkSession.getActiveSession()
                spark.sparkContext.addFile("file://" + tmp_file_path)

    def _update_hnsw_index(
        self,
        item_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
    ):
        index = nmslib.init(
            method=params["method"],
            space=params["space"],
            data_type=nmslib.DataType.DENSE_VECTOR,
        )
        index_path = SparkFiles.get(
            f"{INDEX_FILENAME}_{self._spark_index_file_uid}"
        )
        index.loadIndex(index_path)
        item_vectors = item_vectors.toPandas()
        item_vectors_np = np.squeeze(item_vectors[features_col].values)
        index.addDataPointBatch(
            data=np.stack(item_vectors_np),
            ids=item_vectors["id"].values,
        )
        index_params = {
            "M": params.get("M", 200),
            "efConstruction": params.get("efC", 20000),
            "post": params.get("post", 2000),
        }
        index.createIndex(index_params)

        # saving index to local temp file and sending it to executors
        temp_dir = tempfile.mkdtemp()
        tmp_file_path = os.path.join(
            temp_dir, f"{INDEX_FILENAME}_{self._spark_index_file_uid}"
        )
        index.saveIndex(tmp_file_path)
        spark = SparkSession.getActiveSession()
        spark.sparkContext.addFile("file://" + tmp_file_path)

    def _infer_nmslib_hnsw_index(
        self,
        user_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        k: int,
        filter_seen_items: bool,
        index_type: str = None,
        log: DataFrame = None,
    ) -> DataFrame:

        if params["build_index_on"] == "executor":
            index_file = get_filesystem(params["index_path"])
        else:
            index_file = f"{INDEX_FILENAME}_{self._spark_index_file_uid}"

        _index_file_manager = NmslibIndexFileManager(
            params,
            index_type,
            index_file=index_file,
        )

        index_file_manager_broadcast = State().session.sparkContext.broadcast(
            _index_file_manager
        )

        return_type = "item_idx array<int>, distance array<double>"

        if index_type == "sparse":

            def get_csr_matrix(
                    user_idx: pd.Series,
                    vector_items: pd.Series,
                    vector_relevances: pd.Series,
            ) -> csr_matrix:

                return csr_matrix(
                    (
                        vector_relevances.explode().values.astype(float),
                        (user_idx.repeat(vector_items.apply(lambda x: len(x))).values,
                         vector_items.explode().values.astype(int)),
                    ),
                    shape=(user_idx.max() + 1, vector_items.apply(lambda x: max(x)).max() + 1),
                )

            if filter_seen_items:

                @pandas_udf(return_type)
                def infer_index(
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

                    # take slice
                    m = user_vectors[user_idx.values, :]

                    neighbours = index.knnQueryBatch(
                        m, k=k + max_items_to_retrieve, num_threads=1
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
                    # take slice
                    m = user_vectors[user_idx.values, :]
                    neighbours = index.knnQueryBatch(m, num_threads=1)

                    pd_res = pd.DataFrame(
                        neighbours, columns=["item_idx", "distance"]
                    )

                    # pd_res looks like
                    # item_idx       distances
                    # [1, 2, 3, ...] [-0.5, -0.3, -0.1, ...]
                    # [1, 3, 4, ...] [-0.1, -0.8, -0.2, ...]

                    return pd_res

        else:
            if filter_seen_items:

                @pandas_udf(return_type)
                def infer_index(
                    vectors: pd.Series,
                    num_items: pd.Series,
                    seen_item_idxs: pd.Series,
                ) -> pd.DataFrame:
                    index_file_manager = index_file_manager_broadcast.value
                    index = index_file_manager.index

                    # max number of items to retrieve per batch
                    max_items_to_retrieve = num_items.max()

                    neighbours = index.knnQueryBatch(
                        np.stack(vectors.values),
                        k=k + max_items_to_retrieve,
                        num_threads=1,
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

                    return pd_res

            else:

                @pandas_udf(return_type)
                def infer_index(vectors: pd.Series) -> pd.DataFrame:
                    index_file_manager = index_file_manager_broadcast.value
                    index = index_file_manager.index

                    neighbours = index.knnQueryBatch(
                        np.stack(vectors.values),
                        k=k,
                        num_threads=1,
                    )
                    pd_res = pd.DataFrame(
                        neighbours, columns=["item_idx", "distance"]
                    )

                    return pd_res

        cols = []
        if index_type == "sparse":
            cols += ["user_idx", "vector_items", "vector_relevances"]
        else:
            cols.append(features_col)
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

        if params["build_index_on"] == "executor":
            index_path = params["index_path"]
        elif params["build_index_on"] == "driver":
            index_path = SparkFiles.get(
                f"{INDEX_FILENAME}_{self._spark_index_file_uid}"
            )
        else:
            raise ValueError("Unknown 'build_index_on' param.")

        source = get_filesystem(index_path)
        target = get_filesystem(path)
        self.logger.debug(f"Index file coping from '{index_path}' to '{path}'")

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

        self._nmslib_hnsw_params["build_index_on"] = "driver"
