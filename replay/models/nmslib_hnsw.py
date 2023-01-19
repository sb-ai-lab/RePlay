import logging
import os
from typing import Any, Dict, Optional, Iterator
import uuid

import numpy as np
import pandas as pd
import nmslib
import tempfile

from pyarrow import fs
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame, Window, functions as sf
from pyspark.sql.functions import pandas_udf
from scipy.sparse import csr_matrix
from replay.session_handler import State

from replay.utils import FileSystem, JobGroup, get_filesystem

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
        index_path: Optional[str] = None,
        filesystem: Optional[FileSystem] = None,
        hdfs_uri: Optional[str] = None,
        index_filename: Optional[str] = None,
    ) -> None:

        self._method = index_params["method"]
        self._space = index_params["space"]
        self._efS = index_params.get("efS")
        self._index_type = index_type
        self._index_path = index_path
        self._filesystem = filesystem
        self._hdfs_uri = hdfs_uri
        self._index_filename = index_filename
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
            if self._index_path:
                if self._filesystem == FileSystem.HDFS:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        tmp_file_path = os.path.join(
                            temp_dir, INDEX_FILENAME
                        )
                        source_filesystem = fs.HadoopFileSystem.from_uri(
                            self._hdfs_uri
                        )
                        fs.copy_files(
                            self._index_path,
                            "file://" + tmp_file_path,
                            source_filesystem=source_filesystem,
                        )
                        fs.copy_files(
                            self._index_path + ".dat",
                            "file://" + tmp_file_path + ".dat",
                            source_filesystem=source_filesystem,
                        )
                        self._index.loadIndex(tmp_file_path, load_data=True)
                elif self._filesystem == FileSystem.LOCAL:
                    self._index.loadIndex(self._index_path, load_data=True)
                else:
                    raise ValueError(
                        "`filesystem` must be specified if `index_path` is specified!"
                    )
            else:
                self._index.loadIndex(
                    SparkFiles.get(self._index_filename), load_data=True
                )
        else:
            self._index = nmslib.init(
                method=self._method,
                space=self._space,
                data_type=nmslib.DataType.DENSE_VECTOR,
            )
            if self._index_path:
                if self._filesystem == FileSystem.HDFS:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        tmp_file_path = os.path.join(
                            temp_dir, INDEX_FILENAME
                        )
                        source_filesystem = fs.HadoopFileSystem.from_uri(
                            self._hdfs_uri
                        )
                        fs.copy_files(
                            self._index_path,
                            "file://" + tmp_file_path,
                            source_filesystem=source_filesystem,
                        )
                        self._index.loadIndex(tmp_file_path)
                else:
                    self._index.loadIndex(self._index_path)
            else:
                self._index.loadIndex(SparkFiles.get(self._index_filename))

        if self._efS:
            self._index.setQueryTimeParams({"efSearch": self._efS})
        return self._index


class NmslibHnswMixin:
    """Mixin that provides methods to build nmslib hnsw index and infer it.
    Also provides methods to saving and loading index to/from disk.
    """

    def __init__(self):
        # A unique id for the object.
        self.uid = uuid.uuid4().hex[-12:]

    def _build_nmslib_hnsw_index(
        self,
        item_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        index_type: str = None,
        items_count: Optional[int] = None,
    ):
        """Builds hnsw index and dump it to hdfs or disk.

        Args:
            item_vectors (DataFrame): DataFrame with item vectors
            params (Dict[str, Any]): hnsw params
        """

        with JobGroup(
            f"{self.__class__.__name__}._build_hnsw_index()",
            "all _build_hnsw_index()",
        ):
            if params["build_index_on"] == "executor":
                # to execution in one executor
                item_vectors = item_vectors.repartition(1)

                filesystem, hdfs_uri, index_path = get_filesystem(
                    params["index_path"]
                )

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

                        data = pdf["similarity"].values
                        row_ind = pdf["item_idx_two"].values
                        col_ind = pdf["item_idx_one"].values

                        # M = pdf['item_idx_one'].max() + 1
                        # N = pdf['item_idx_two'].max() + 1

                        sim_matrix_tmp = csr_matrix(
                            (data, (row_ind, col_ind)),
                            shape=(items_count, items_count),
                        )
                        index.addDataPointBatch(data=sim_matrix_tmp)

                        index_params = {}
                        if "M" in params:
                            index_params["M"] = params["M"]
                        if "efC" in params:
                            index_params["efConstruction"] = params["efC"]
                        if "post" in params:
                            index_params["post"] = params["post"]
                        if index_params:
                            index.createIndex(index_params)
                        else:
                            index.createIndex()

                        if filesystem == FileSystem.HDFS:
                            temp_dir = tempfile.mkdtemp()
                            tmp_file_path = os.path.join(
                                temp_dir, INDEX_FILENAME
                            )
                            index.saveIndex(tmp_file_path, save_data=True)

                            destination_filesystem = (
                                fs.HadoopFileSystem.from_uri(hdfs_uri)
                            )
                            fs.copy_files(
                                "file://" + tmp_file_path,
                                index_path,
                                destination_filesystem=destination_filesystem,
                            )
                            fs.copy_files(
                                "file://" + tmp_file_path + ".dat",
                                index_path + ".dat",
                                destination_filesystem=destination_filesystem,
                            )
                            # param use_threads=True (?)
                        else:
                            index.saveIndex(index_path, save_data=True)

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
                            item_vectors_np = np.squeeze(
                                pdf[features_col].values
                            )
                            index.addDataPointBatch(
                                data=np.stack(item_vectors_np),
                                ids=pdf["item_idx"].values,
                            )
                        index_params = {}
                        if "M" in params:
                            index_params["M"] = params["M"]
                        if "efC" in params:
                            index_params["efConstruction"] = params["efC"]
                        if "post" in params:
                            index_params["post"] = params["post"]
                        if index_params:
                            index.createIndex(index_params)
                        else:
                            index.createIndex()

                        if filesystem == FileSystem.HDFS:
                            temp_dir = tempfile.mkdtemp()
                            tmp_file_path = os.path.join(
                                temp_dir, INDEX_FILENAME
                            )
                            index.saveIndex(tmp_file_path)

                            destination_filesystem = (
                                fs.HadoopFileSystem.from_uri(hdfs_uri)
                            )
                            fs.copy_files(
                                "file://" + tmp_file_path,
                                index_path,
                                destination_filesystem=destination_filesystem,
                            )
                            # param use_threads=True (?)
                        else:
                            index.saveIndex(index_path)

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
                    index_params = {}
                    if "M" in params:
                        index_params["M"] = params["M"]
                    if "efC" in params:
                        index_params["efConstruction"] = params["efC"]
                    if "post" in params:
                        index_params["post"] = params["post"]
                    if index_params:
                        index.createIndex(index_params)
                    else:
                        index.createIndex()
                    # saving index to local temp file and sending it to executors
                    temp_dir = tempfile.mkdtemp()
                    tmp_file_path = os.path.join(
                        temp_dir, f"{INDEX_FILENAME}_{self.uid}"
                    )
                    index.saveIndex(tmp_file_path, save_data=True)
                    spark = SparkSession.getActiveSession()
                    spark.sparkContext.addFile("file://" + tmp_file_path)
                    spark.sparkContext.addFile(
                        "file://" + tmp_file_path + ".dat"
                    )

                else:
                    item_vectors = item_vectors.toPandas()
                    item_vectors_np = np.squeeze(
                        item_vectors[features_col].values
                    )
                    index = nmslib.init(
                        method=params["method"],
                        space=params["space"],
                        data_type=nmslib.DataType.DENSE_VECTOR,
                    )
                    index.addDataPointBatch(
                        data=np.stack(item_vectors_np),
                        ids=item_vectors["item_idx"].values,
                    )
                    index_params = {}
                    if "M" in params:
                        index_params["M"] = params["M"]
                    if "efC" in params:
                        index_params["efConstruction"] = params["efC"]
                    if "post" in params:
                        index_params["post"] = params["post"]
                    if index_params:
                        index.createIndex(index_params)
                    else:
                        index.createIndex()

                    # saving index to local temp file and sending it to executors
                    temp_dir = tempfile.mkdtemp()
                    tmp_file_path = os.path.join(
                        temp_dir, f"{INDEX_FILENAME}_{self.uid}"
                    )
                    index.saveIndex(tmp_file_path)
                    spark = SparkSession.getActiveSession()
                    spark.sparkContext.addFile("file://" + tmp_file_path)

    def _update_hnsw_index(
        self,
        item_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any]
    ):
        index = nmslib.init(
            method=params["method"],
            space=params["space"],
            data_type=nmslib.DataType.DENSE_VECTOR,
        )
        index_path = SparkFiles.get(f"{INDEX_FILENAME}_{self.uid}")
        index.loadIndex(index_path)
        item_vectors = item_vectors.toPandas()
        item_vectors_np = np.squeeze(
            item_vectors[features_col].values
        )
        index.addDataPointBatch(
            data=np.stack(item_vectors_np),
            ids=item_vectors["id"].values,
        )
        index_params = {}
        if "M" in params:
            index_params["M"] = params["M"]
        if "efC" in params:
            index_params["efConstruction"] = params["efC"]
        if "post" in params:
            index_params["post"] = params["post"]
        if index_params:
            index.createIndex(index_params)
        else:
            index.createIndex()

        # saving index to local temp file and sending it to executors
        temp_dir = tempfile.mkdtemp()
        tmp_file_path = os.path.join(
            temp_dir, f"{INDEX_FILENAME}_{self.uid}"
        )
        index.saveIndex(tmp_file_path)
        spark = SparkSession.getActiveSession()
        spark.sparkContext.addFile("file://" + tmp_file_path)

    def _build_hnsw_index_on_sparse(
        self,
        item_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        items_count: int,
        users_count: int
    ):
        """ "Builds hnsw index and dump it to hdfs or disk.

        Args:
            item_vectors (DataFrame): DataFrame with item vectors
            params (Dict[str, Any]): hnsw params
        """

        with JobGroup(
            f"{self.__class__.__name__}._build_hnsw_index()",
            "all _build_hnsw_index()",
        ):
            if params["build_index_on"] == "executor":
                # to execution in one executor
                item_vectors = item_vectors.repartition(1)

                filesystem, hdfs_uri, index_path = get_filesystem(
                    params["index_path"]
                )

                def build_index(iterator):
                    index = nmslib.init(
                        method=params["method"],
                        space=params["space"],
                        data_type=nmslib.DataType.SPARSE_VECTOR,
                    )

                    pdfs = []
                    for pdf in iterator:
                        pdfs.append(pdf)

                    pdf = pd.concat(pdfs, copy=False)

                    data = pdf["relevance"].values
                    row_ind = pdf["item_idx"].values
                    col_ind = pdf["user_idx"].values

                    interactions_matrix = csr_matrix(
                        (data, (row_ind, col_ind)),
                        shape=(items_count, users_count),
                    )
                    index.addDataPointBatch(data=interactions_matrix)

                    index_params = {}
                    if "M" in params:
                        index_params["M"] = params["M"]
                    if "efC" in params:
                        index_params["efConstruction"] = params["efC"]
                    if "post" in params:
                        index_params["post"] = params["post"]
                    if index_params:
                        index.createIndex(index_params)
                    else:
                        index.createIndex()

                    if filesystem == FileSystem.HDFS:
                        temp_path = tempfile.mkdtemp()
                        tmp_file_path = os.path.join(
                            temp_path, "nmslib_hnsw_index"
                        )
                        index.saveIndex(tmp_file_path, save_data=True)

                        destination_filesystem = (
                            fs.HadoopFileSystem.from_uri(hdfs_uri)
                        )
                        fs.copy_files(
                            "file://" + tmp_file_path,
                            index_path,
                            destination_filesystem=destination_filesystem,
                        )
                        fs.copy_files(
                            "file://" + tmp_file_path + ".dat",
                            index_path + ".dat",
                            destination_filesystem=destination_filesystem,
                        )
                        # param use_threads=True (?)
                    else:
                        index.saveIndex(index_path, save_data=True)

                    yield pd.DataFrame(data={"_success": 1}, index=[0])


                # builds index on executor and writes it to shared disk or hdfs
                item_vectors.select(
                    "item_idx", "user_idx", "relevance"
                ).mapInPandas(build_index, "_success int").show()
            else:
                item_vectors = item_vectors.toPandas()

                index = nmslib.init(
                    method=params["method"],
                    space=params["space"],
                    data_type=nmslib.DataType.SPARSE_VECTOR,
                )

                data = item_vectors["relevance"].values
                row_ind = item_vectors["item_idx"].values
                col_ind = item_vectors["user_idx"].values

                sim_matrix = csr_matrix(
                    (data, (row_ind, col_ind)),
                    shape=(items_count, items_count),
                )
                index.addDataPointBatch(data=sim_matrix)
                index_params = {}
                if "M" in params:
                    index_params["M"] = params["M"]
                if "efC" in params:
                    index_params["efConstruction"] = params["efC"]
                if "post" in params:
                    index_params["post"] = params["post"]
                if index_params:
                    index.createIndex(index_params)
                else:
                    index.createIndex()
                # saving index to local temp file and sending it to executors
                temp_path = tempfile.mkdtemp()
                tmp_file_path = os.path.join(
                    temp_path, "nmslib_hnsw_index_" + self.uid
                )
                index.saveIndex(tmp_file_path, save_data=True)
                spark = SparkSession.getActiveSession()
                spark.sparkContext.addFile("file://" + tmp_file_path)
                spark.sparkContext.addFile(
                    "file://" + tmp_file_path + ".dat"
                )

    def _infer_nmslib_hnsw_index(
        self,
        user_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        k: int,
        index_type: str = None,
    ):

        if params["build_index_on"] == "executor":
            filesystem, hdfs_uri, index_path = get_filesystem(
                params["index_path"]
            )
            _index_file_manager = NmslibIndexFileManager(
                params, index_type, index_path, filesystem, hdfs_uri
            )
        else:
            print(f"Creation NmslibIndexFileManager instance with index_filename=nmslib_hnsw_index_{self.uid}")
            _index_file_manager = NmslibIndexFileManager(params, index_type, index_filename=f"{INDEX_FILENAME}_{self.uid}")

        index_file_manager_broadcast = State().session.sparkContext.broadcast(
            _index_file_manager
        )

        return_type = (
            "user_idx int, item_idx array<int>, distance array<double>"
        )

        if index_type == "sparse":
            interactions_matrix_broadcast = self._interactions_matrix_broadcast

            @pandas_udf(return_type)
            def infer_index(
                user_idx: pd.Series, num_items: pd.Series
            ) -> pd.DataFrame:
                index_file_manager = index_file_manager_broadcast.value
                interactions_matrix = interactions_matrix_broadcast.value

                index = index_file_manager.index

                # max number of items to retrieve per batch
                max_items_to_retrieve = num_items.max()

                # take slice
                m = interactions_matrix[user_idx.values, :]
                neighbours = index.knnQueryBatch(
                    m, k=k + max_items_to_retrieve,
                    num_threads=1
                )
                pd_res = pd.DataFrame(
                    neighbours, columns=["item_idx", "distance"]
                )
                # which is better?
                pd_res["user_idx"] = user_idx.values
                # pd_res = pd_res.assign(user_idx=user_idx.values)

                # pd_res looks like
                # user_id item_idx  distances
                # 0       [1, 2, 3, ...] [-0.5, -0.3, -0.1, ...]
                # 1       [1, 3, 4, ...] [-0.1, -0.8, -0.2, ...]

                return pd_res

        else:

            @pandas_udf(return_type)
            def infer_index(
                user_ids: pd.Series, vectors: pd.Series, num_items: pd.Series
            ) -> pd.DataFrame:
                index_file_manager = index_file_manager_broadcast.value
                index = index_file_manager.index

                # max number of items to retrieve per batch
                max_items_to_retrieve = num_items.max()

                neighbours = index.knnQueryBatch(
                    np.stack(vectors.values),
                    k=k + max_items_to_retrieve,
                    num_threads=1
                )
                pd_res = pd.DataFrame(
                    neighbours, columns=["item_idx", "distance"]
                )
                # which is better?
                # pd_res['user_idx'] = user_ids
                pd_res = pd_res.assign(user_idx=user_ids.values)

                return pd_res

        with JobGroup(
            "infer_index()",
            "infer_hnsw_index (inside 1)",
        ):
            if index_type == "sparse":
                res = user_vectors.select(
                    infer_index("user_idx", "num_items").alias("r")
                )
            else:
                res = user_vectors.select(
                    infer_index("user_idx", features_col, "num_items").alias("r")
                )
            # res = res.cache()
            # res.write.mode("overwrite").format("noop").save()

        with JobGroup(
            "res.withColumn('zip_exp', ...",
            "infer_hnsw_index (inside 2)",
        ):
            res = res.select('*',
                sf.explode(sf.arrays_zip("r.item_idx", "r.distance")).alias('zip_exp')
            )
            
            # Fix arrays_zip random behavior. It can return zip_exp.0 or zip_exp.item_idx in different machines
            item_idx_field_name: str = res.schema["zip_exp"].jsonValue()["type"]["fields"][0]["name"]
            distance_field_name: str = res.schema["zip_exp"].jsonValue()["type"]["fields"][1]["name"]

            res = res.select(
                sf.col("r.user_idx").alias("user_idx"),
                sf.col(f"zip_exp.{item_idx_field_name}").alias("item_idx"),
                (sf.lit(-1.0) * sf.col(f"zip_exp.{distance_field_name}")).alias("relevance")
            )
            # res = res.cache()
            # res.write.mode("overwrite").format("noop").save()

        return res

    def _infer_hnsw_index_on_sparse(
        self,
        user_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        k: int,
        index_type: str = None,
    ):

        if params["build_index_on"] == "executor":
            filesystem, hdfs_uri, index_path = get_filesystem(
                params["index_path"]
            )
            _index_file_manager = NmslibIndexFileManager(
                params, index_type, index_path, filesystem, hdfs_uri
            )
        else:
            _index_file_manager = NmslibIndexFileManager(params, index_type, index_filename="nmslib_hnsw_index_" + self.uid)

        index_file_manager_broadcast = State().session.sparkContext.broadcast(
            _index_file_manager
        )

        return_type = (
            "item_idx int, item_idxs array<int>, distances array<double>"
        )

        interactions_matrix_broadcast = self._interactions_matrix_broadcast

        @pandas_udf(return_type)
        def infer_index(
            item_idx: pd.Series
        ) -> pd.DataFrame:
            index_file_manager = index_file_manager_broadcast.value
            interactions_matrix = interactions_matrix_broadcast.value

            index = index_file_manager.index

            # take slice
            m = interactions_matrix[item_idx.values, :]
            neighbours = index.knnQueryBatch(
                m, k=k,
                num_threads=1
            )
            pd_res = pd.DataFrame(
                neighbours, columns=["item_idxs", "distances"]
            )
            # which is better?
            pd_res["item_idx"] = item_idx.values
            # pd_res = pd_res.assign(user_idx=user_idx.values)

            # pd_res looks like
            # user_id item_idxs  distances
            # 0       [1, 2, 3, ...] [-0.5, -0.3, -0.1, ...]
            # 1       [1, 3, 4, ...] [-0.1, -0.8, -0.2, ...]

            return pd_res


        with JobGroup(
            "infer_index()",
            "infer_hnsw_index (inside 1)",
        ):
            res = user_vectors.select(
                infer_index("item_idx").alias("r")
            )
            # res = res.cache()
            # res.write.mode("overwrite").format("noop").save()

        with JobGroup(
            "res.withColumn('zip_exp', ...",
            "infer_hnsw_index (inside 2)",
        ):
            res = res.select('*',
                sf.explode(sf.arrays_zip("r.item_idxs", "r.distances")).alias('zip_exp')
            )
            
            # Fix arrays_zip random behavior. It can return zip_exp.0 or zip_exp.item_idx in different machines
            item_idx_field_name: str = res.schema["zip_exp"].jsonValue()["type"]["fields"][0]["name"]
            distance_field_name: str = res.schema["zip_exp"].jsonValue()["type"]["fields"][1]["name"]

            res = res.select(
                sf.col("r.item_idx").alias("item_idx_one"),
                sf.col(f"zip_exp.{item_idx_field_name}").alias("item_idx_two"),
                (sf.lit(-1.0) * sf.col(f"zip_exp.{distance_field_name}")).alias("similarity")
            )
            # res = res.cache()
            # res.write.mode("overwrite").format("noop").save()

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
            index_path = SparkFiles.get(f"{INDEX_FILENAME}_{self.uid}")
        else:
            raise ValueError("Unknown 'build_index_on' param.")

        from_filesystem, from_hdfs_uri, from_path = get_filesystem(
            index_path
        )
        to_filesystem, to_hdfs_uri, to_path = get_filesystem(path)
        print(f"Index file coping from '{from_path}' to '{to_path}'")

        from_paths = []
        target_paths = []
        if sparse:
            from_paths.append(from_path)
            from_paths.append(from_path + ".dat")
            print(from_paths)
            index_file_target_path = os.path.join(to_path, INDEX_FILENAME)
            target_paths.append(index_file_target_path)
            target_paths.append(index_file_target_path + ".dat")
            print(target_paths)
        else:
            from_paths.append(from_path)
            print(from_paths)
            index_file_target_path = os.path.join(to_path, INDEX_FILENAME)
            target_paths.append(index_file_target_path)
            print(target_paths)

        if from_filesystem == FileSystem.HDFS:
            source_filesystem = fs.HadoopFileSystem.from_uri(from_hdfs_uri)
        else:
            source_filesystem = fs.LocalFileSystem()
        if to_filesystem == FileSystem.HDFS:
            destination_filesystem = fs.HadoopFileSystem.from_uri(
                to_hdfs_uri
            )
        else:
            destination_filesystem = fs.LocalFileSystem()

        for from_path, target_path in zip(from_paths, target_paths):
            fs.copy_files(
                from_path,
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
        from_filesystem, from_hdfs_uri, from_path = get_filesystem(
            path + '/nmslib_hnsw_index'
        )

        to_path = tempfile.mkdtemp()
        to_path = os.path.join(to_path, f"{INDEX_FILENAME}_{self.uid}")

        from_paths = []
        target_paths = []
        if sparse:
            from_paths.append(from_path)
            from_paths.append(from_path + ".dat")
            print(from_paths)
            target_paths.append(to_path)
            target_paths.append(to_path + ".dat")
            print(target_paths)
        else:
            from_paths.append(from_path)
            print(from_paths)
            target_paths.append(to_path)
            print(target_paths)

        if from_filesystem == FileSystem.HDFS:
            source_filesystem = fs.HadoopFileSystem.from_uri(from_hdfs_uri)
        else:
            source_filesystem = fs.LocalFileSystem()

        destination_filesystem = fs.LocalFileSystem()

        for from_path, to_path in zip(from_paths, target_paths):
            fs.copy_files(
                from_path,
                to_path,
                source_filesystem=source_filesystem,
                destination_filesystem=destination_filesystem,
            )

        spark = SparkSession.getActiveSession()
        for target_path in target_paths:
            spark.sparkContext.addFile("file://" + target_path)

        self._nmslib_hnsw_params["build_index_on"] = "driver"
