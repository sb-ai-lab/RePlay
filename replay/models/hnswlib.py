import logging
import os
from typing import Any, Dict, Optional
import uuid

import numpy as np
import pandas as pd
import hnswlib
import tempfile

from pyarrow import fs
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame, functions as sf
from pyspark.sql.functions import pandas_udf
from replay.session_handler import State

from replay.utils import FileSystem, JobGroup, get_filesystem

logger = logging.getLogger("replay")


class HnswlibIndexFileManager:
    """Loads index from hdfs, local disk or SparkFiles dir and keep it in a memory.
    Instance of `HnswlibIndexFileManager` broadcasts to executors and is used in pandas_udf.
    """

    def __init__(
        self,
        index_params,
        index_dim: int,
        index_path: Optional[str] = None,
        filesystem: Optional[FileSystem] = None,
        hdfs_uri: Optional[str] = None,
        index_filename: Optional[str] = None,
    ) -> None:

        self._space = index_params["space"]
        # self._efS = index_params.get("efS")
        self._dim = index_dim
        self._index_path = index_path
        self._filesystem = filesystem
        self._hdfs_uri = hdfs_uri
        self._index_filename = index_filename
        self._index = None

    @property
    def index(self):
        if self._index:
            return self._index

        # self._index = nmslib.init(
        #     method=self._method,
        #     space=self._space,
        #     data_type=nmslib.DataType.DENSE_VECTOR,
        # )
        self._index = hnswlib.Index(space=self._space, dim=self._dim)
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
                    # self._index.loadIndex(tmp_file_path)
                    self._index.load_index(tmp_file_path)
            else:
                # self._index.loadIndex(self._index_path)
                self._index.load_index(self._index_path)
        else:
            # self._index.loadIndex(SparkFiles.get(self._index_filename))
            self._index.load_index(SparkFiles.get(self._index_filename))

        # if self._efS:
        #     # self._index.setQueryTimeParams({"efSearch": self._efS})
        #     self._index.set_ef(self._efS)
        return self._index


class HnswlibMixin:
    """Mixin that provides methods to build nmslib hnsw index and infer it.
    Also provides methods to saving and loading index to/from disk.
    """

    def __init__(self):
        #: A unique id for the object.
        self.uid = uuid.uuid4().hex[-12:]

    def _build_hnsw_index(
        self,
        item_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        dim: int,
        num_elements: int,
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
                    # index = nmslib.init(
                    #     method=params["method"],
                    #     space=params["space"],
                    #     data_type=nmslib.DataType.DENSE_VECTOR,
                    # )
                    index = hnswlib.Index(space=params["space"], dim=dim)

                    # Initializing index - the maximum number of elements should be known beforehand
                    index.init_index(max_elements = num_elements, ef_construction = params["efC"], M = params["M"])

                    for pdf in iterator:
                        item_vectors_np = np.squeeze(
                            pdf[features_col].values
                        )
                        index.add_items(np.stack(item_vectors_np), pdf["item_idx"].values)
                        # index.addDataPointBatch(
                        #     data=np.stack(item_vectors_np),
                        #     ids=pdf["item_idx"].values,
                        # )
                    # index_params = {}
                    # if "M" in params:
                    #     index_params["M"] = params["M"]
                    # if "efC" in params:
                    #     index_params["efConstruction"] = params["efC"]
                    # if "post" in params:
                    #     index_params["post"] = params["post"]
                    # if index_params:
                    #     index.createIndex(index_params)
                    # else:
                    #     index.createIndex()

                    if filesystem == FileSystem.HDFS:
                        temp_path = tempfile.mkdtemp()
                        tmp_file_path = os.path.join(
                            temp_path, "nmslib_hnsw_index"
                        )
                        # index.saveIndex(tmp_file_path)
                        index.save_index(tmp_file_path)

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
                        # index.saveIndex(index_path)
                        index.save_index(index_path)

                    yield pd.DataFrame(data={"_success": 1}, index=[0])

                # builds index on executor and writes it to shared disk or hdfs
                item_vectors.select("item_idx", features_col).mapInPandas(
                    build_index, "_success int"
                ).show()
            else:
                item_vectors = item_vectors.toPandas()
                item_vectors_np = np.squeeze(
                    item_vectors[features_col].values
                )

                index = hnswlib.Index(space=params["space"], dim=dim)

                # Initializing index - the maximum number of elements should be known beforehand
                index.init_index(max_elements = num_elements, ef_construction = params["efC"], M = params["M"])

                index.add_items(np.stack(item_vectors_np), item_vectors["item_idx"].values)                

                # index = nmslib.init(
                #     method=params["method"],
                #     space=params["space"],
                #     data_type=nmslib.DataType.DENSE_VECTOR,
                # )
                # index.addDataPointBatch(
                #     data=np.stack(item_vectors_np),
                #     ids=item_vectors["item_idx"].values,
                # )
                # index_params = {}
                # if "M" in params:
                #     index_params["M"] = params["M"]
                # if "efC" in params:
                #     index_params["efConstruction"] = params["efC"]
                # if "post" in params:
                #     index_params["post"] = params["post"]
                # if index_params:
                #     index.createIndex(index_params)
                # else:
                #     index.createIndex()

                # saving index to local temp file and sending it to executors
                temp_path = tempfile.mkdtemp()
                tmp_file_path = os.path.join(
                    temp_path, "nmslib_hnsw_index_" + self.uid
                )
                # index.saveIndex(tmp_file_path)
                index.save_index(tmp_file_path)
                spark = SparkSession.getActiveSession()
                spark.sparkContext.addFile("file://" + tmp_file_path)

    def _update_hnsw_index(
        self,
        item_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        dim: int,
        num_elements: int
    ):
        # index = nmslib.init(
        #     method=params["method"],
        #     space=params["space"],
        #     data_type=nmslib.DataType.DENSE_VECTOR,
        # )
        index = hnswlib.Index(space=params["space"], dim=dim)
        index_path = SparkFiles.get("nmslib_hnsw_index_" + self.uid)
        # index.loadIndex(index_path)
        index.load_index(index_path, max_elements = num_elements)
        item_vectors = item_vectors.toPandas()
        item_vectors_np = np.squeeze(
            item_vectors[features_col].values
        )
        index.add_items(np.stack(item_vectors_np), item_vectors["id"].values)
        # index.addDataPointBatch(
        #     data=np.stack(item_vectors_np),
        #     ids=item_vectors["id"].values,
        # )
        # index_params = {}
        # if "M" in params:
        #     index_params["M"] = params["M"]
        # if "efC" in params:
        #     index_params["efConstruction"] = params["efC"]
        # if "post" in params:
        #     index_params["post"] = params["post"]
        # if index_params:
        #     index.createIndex(index_params)
        # else:
        #     index.createIndex()

        self.uid = uuid.uuid4().hex[-12:]
        # saving index to local temp file and sending it to executors
        temp_path = tempfile.mkdtemp()
        tmp_file_path = os.path.join(
            temp_path, "nmslib_hnsw_index_" + self.uid
        )
        # index.saveIndex(tmp_file_path)
        index.save_index(tmp_file_path)
        spark = SparkSession.getActiveSession()
        spark.sparkContext.addFile("file://" + tmp_file_path)

    def _infer_hnsw_index(
        self,
        user_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        k: int,
        index_dim: str = None,
    ):

        if params["build_index_on"] == "executor":
            filesystem, hdfs_uri, index_path = get_filesystem(
                params["index_path"]
            )
            _index_file_manager = HnswlibIndexFileManager(
                params, index_dim, index_path, filesystem, hdfs_uri
            )
        else:
            print(f"Creation HnswlibIndexFileManager instance with index_filename=nmslib_hnsw_index_{self.uid}")
            _index_file_manager = HnswlibIndexFileManager(params, index_dim, index_filename="nmslib_hnsw_index_" + self.uid)

        index_file_manager_broadcast = State().session.sparkContext.broadcast(
            _index_file_manager
        )

        return_type = (
            "user_idx int, item_idx array<int>, distance array<double>"
        )

        @pandas_udf(return_type)
        def infer_index(
            user_ids: pd.Series, vectors: pd.Series, num_items: pd.Series
        ) -> pd.DataFrame:
            index_file_manager = index_file_manager_broadcast.value
            index = index_file_manager.index

            # max number of items to retrieve per batch
            max_items_to_retrieve = num_items.max()

            labels, distances = index.knn_query(np.stack(vectors.values), k = k + max_items_to_retrieve)
            # neighbours = index.knnQueryBatch(
            #     np.stack(vectors.values),
            #     k=k + max_items_to_retrieve,
            #     num_threads=1
            # )
            # pd_res = pd.DataFrame(
            #     neighbours, columns=["item_idx", "distance"]
            # )
            # pd_res = pd.DataFrame({'labels': list(labels), 'distances': list(distances)})
            pd_res = pd.DataFrame({'item_idx': list(labels), 'distance': list(distances)})

            # which is better?
            # pd_res['user_idx'] = user_ids
            pd_res = pd_res.assign(user_idx=user_ids.values)

            return pd_res

        with JobGroup(
            "infer_index()",
            "infer_hnsw_index (inside 1)",
        ):
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

        from_filesystem, from_hdfs_uri, from_path = get_filesystem(
            params["index_path"]
        )
        to_filesystem, to_hdfs_uri, to_path = get_filesystem(path)

        if from_filesystem == FileSystem.HDFS:
            source_filesystem = fs.HadoopFileSystem.from_uri(from_hdfs_uri)
            if to_filesystem == FileSystem.HDFS:
                destination_filesystem = fs.HadoopFileSystem.from_uri(
                    to_hdfs_uri
                )
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
        else:
            source_filesystem = fs.LocalFileSystem()
            if to_filesystem == FileSystem.HDFS:
                destination_filesystem = fs.HadoopFileSystem.from_uri(
                    to_hdfs_uri
                )
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

        # param use_threads=True (?)
