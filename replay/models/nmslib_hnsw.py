import logging
import os
from typing import Any, Dict, Optional

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
    ) -> None:

        self._method = index_params["method"]
        self._space = index_params["space"]
        self._efS = index_params.get("efS")
        self._index_type = index_type
        self._index_path = index_path
        self._filesystem = filesystem
        self._hdfs_uri = hdfs_uri
        self._index = None

    @property
    def index(self):
        if self._index:
            print("using loaded index")
            return self._index

        print("*load index*")
        if self._index_type == "sparse":
            self._index = nmslib.init(
                method=self._method,
                space=self._space,
                data_type=nmslib.DataType.SPARSE_VECTOR,
            )
            if self._index_path:
                if self._filesystem == FileSystem.HDFS:
                    with tempfile.TemporaryDirectory() as temp_path:
                        tmp_file_path = os.path.join(
                            temp_path, "nmslib_hnsw_index"
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
                    SparkFiles.get("nmslib_hnsw_index"), load_data=True
                )
        else:
            self._index = nmslib.init(
                method=self._method,
                space=self._space,
                data_type=nmslib.DataType.DENSE_VECTOR,
            )
            if self._index_path:
                if self._filesystem == FileSystem.HDFS:
                    with tempfile.TemporaryDirectory() as temp_path:
                        tmp_file_path = os.path.join(
                            temp_path, "nmslib_hnsw_index"
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
                self._index.loadIndex(SparkFiles.get("nmslib_hnsw_index"))

        if self._efS:
            self._index.setQueryTimeParams({"efSearch": self._efS})
        return self._index


class NmslibHnsw:
    def _build_hnsw_index(
        self,
        item_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        index_type: str = None,
        items_count: Optional[int] = None,
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

                if index_type == "sparse":

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

                        data = pdf["similarity"].values
                        row_ind = pdf["item_idx_two"].values  # 'item_idx_one'
                        col_ind = pdf["item_idx_one"].values  # 'item_idx_two'

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

                else:

                    def build_index(iterator):
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
                            temp_path = tempfile.mkdtemp()
                            tmp_file_path = os.path.join(
                                temp_path, "nmslib_hnsw_index"
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

                            # hdfs = fs.HadoopFileSystem.from_uri(hdfs_uri)
                            # fs.copy_files("file://" + tmp_file_path, index_path, destination_filesystem=hdfs)
                            # param use_threads=True (?)
                        else:
                            index.saveIndex(index_path)

                        yield pd.DataFrame(data={"_success": 1}, index=[0])

                # builds index on executor and writes it to shared disk or hdfs
                if index_type == "sparse":
                    item_vectors.select(
                        "similarity", "item_idx_one", "item_idx_two"
                    ).mapInPandas(build_index, "_success int").show()

                    # logger.debug(f"filesystem: {filesystem}")
                    # logger.debug(f"hdfs_uri: {hdfs_uri}")
                    # logger.debug(f"index_path: {index_path}")
                    # # share index to executors
                    # spark = SparkSession.getActiveSession()
                    # if filesystem == FileSystem.HDFS:
                    #     full_path = hdfs_uri + index_path
                    # else:
                    #     full_path = index_path
                    # spark.sparkContext.addFile(full_path)
                    # # if index is sparse then we need include .dat file also!
                    # spark.sparkContext.addFile(full_path + ".dat")
                else:
                    item_vectors.select("item_idx", features_col).mapInPandas(
                        build_index, "_success int"
                    ).show()

                    # # share index to executors
                    # spark = SparkSession.getActiveSession()
                    # if filesystem == FileSystem.HDFS:
                    #     full_path = hdfs_uri + index_path
                    # else:
                    #     full_path = index_path
                    # spark.sparkContext.addFile(full_path)
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
                    temp_path = tempfile.mkdtemp()
                    tmp_file_path = os.path.join(
                        temp_path, "nmslib_hnsw_index"
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
                    temp_path = tempfile.mkdtemp()
                    tmp_file_path = os.path.join(
                        temp_path, "nmslib_hnsw_index"
                    )
                    index.saveIndex(tmp_file_path)
                    spark = SparkSession.getActiveSession()
                    spark.sparkContext.addFile("file://" + tmp_file_path)

    # def _filter_seen_hnsw_res(
    #     self, log: DataFrame, pred: DataFrame, k: int, id_type="idx"
    # ):
    #     """
    #     filter items seen in log and leave top-k most relevant
    #     """

    #     user_id = "user_" + id_type
    #     item_id = "item_" + id_type

    #     recs = pred.join(log, on=[user_id, item_id], how="anti")

    #     recs = (
    #         recs.withColumn(
    #             "temp_rank",
    #             sf.row_number().over(
    #                 Window.partitionBy(user_id).orderBy(
    #                     sf.col("relevance").desc()
    #                 )
    #             ),
    #         )
    #         .filter(sf.col("temp_rank") <= k)
    #         .drop("temp_rank")
    #     )

    #     return recs

    def _infer_hnsw_index(
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
            _index_file_manager = NmslibIndexFileManager(params, index_type)

        index_file_manager_broadcast = State().session.sparkContext.broadcast(
            _index_file_manager
        )

        # k_udf = k + self._max_items_to_retrieve
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
                    m, k=k + max_items_to_retrieve
                )  # , num_threads=1  k_udf
                pd_res = pd.DataFrame(
                    neighbours, columns=["item_idx", "distance"]
                )
                # which is better?
                pd_res["user_idx"] = user_idx.values
                # pd_res = pd_res.assign(user_idx=user_idx.values)

                # pd_res looks like
                # user_id item_idx  distances
                # 0       [1, 2, 3] [-0.5, -0.3, -0.1]
                # 1       [1, 3, 4] [-0.1, -0.8, -0.2]

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
                    k=k + max_items_to_retrieve
                ) # num_threads=1
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

    def _save_nmslib_hnsw_index(self, path):
        """Method save (copy) index from hdfs (or local) to `path` directory.
        `path` can be an hdfs path or a local path.

        Args:
            path (_type_): directory where to dump (copy) the index
        """

        params = self._nmslib_hnsw_params

        if params["build_index_on"] == "executor":
            index_filename = params["index_path"].split("/")[-1]
            # print(SparkFiles.get(index_filename))
            from_path = SparkFiles.get(index_filename)
            logger.debug(f"index local path: {from_path}")
        else:
            # print(SparkFiles.get("nmslib_hnsw_index"))
            from_path = SparkFiles.get("nmslib_hnsw_index")
            logger.debug(f"index local path: {from_path}")

        # from_filesystem, from_hdfs_uri, from_path = get_filesystem(
        #     params["index_path"]
        # )
        to_filesystem, to_hdfs_uri, to_path = get_filesystem(path)

        source_filesystem = fs.LocalFileSystem()
        if to_filesystem == FileSystem.HDFS:
            destination_filesystem = fs.HadoopFileSystem.from_uri(to_hdfs_uri)
            fs.copy_files(
                from_path,
                os.path.join(to_path, "nmslib_hnsw_index"),
                source_filesystem=source_filesystem,
                destination_filesystem=destination_filesystem,
            )
        else:
            destination_filesystem = fs.LocalFileSystem()
            fs.copy_files(
                from_path,
                os.path.join(to_path, "nmslib_hnsw_index"),
                source_filesystem=source_filesystem,
                destination_filesystem=destination_filesystem,
            )

        # if from_filesystem == FileSystem.HDFS:
        #     source_filesystem = fs.HadoopFileSystem.from_uri(from_hdfs_uri)
        #     if to_filesystem == FileSystem.HDFS:
        #         destination_filesystem = fs.HadoopFileSystem.from_uri(
        #             to_hdfs_uri
        #         )
        #         fs.copy_files(
        #             from_path,
        #             os.path.join(to_path, "nmslib_hnsw_index"),
        #             source_filesystem=source_filesystem,
        #             destination_filesystem=destination_filesystem,
        #         )
        #     else:
        #         destination_filesystem = fs.LocalFileSystem()
        #         fs.copy_files(
        #             from_path,
        #             os.path.join(to_path, "nmslib_hnsw_index"),
        #             source_filesystem=source_filesystem,
        #             destination_filesystem=destination_filesystem,
        #         )
        # else:
        #     source_filesystem = fs.LocalFileSystem()
        #     if to_filesystem == FileSystem.HDFS:
        #         destination_filesystem = fs.HadoopFileSystem.from_uri(
        #             to_hdfs_uri
        #         )
        #         fs.copy_files(
        #             from_path,
        #             os.path.join(to_path, "nmslib_hnsw_index"),
        #             source_filesystem=source_filesystem,
        #             destination_filesystem=destination_filesystem,
        #         )
        #     else:
        #         destination_filesystem = fs.LocalFileSystem()
        #         fs.copy_files(
        #             from_path,
        #             os.path.join(to_path, "nmslib_hnsw_index"),
        #             source_filesystem=source_filesystem,
        #             destination_filesystem=destination_filesystem,
        #         )

        # param use_threads=True (?)
