import logging
import os
import shutil
import tempfile
import uuid
import weakref
from typing import Any, Dict, Iterator, Optional, Union

import hnswlib
import numpy as np
import pandas as pd
from pyarrow import fs
from pyspark import SparkFiles
from pyspark.sql import DataFrame, functions as sf
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf

from replay.ann.ann_mixin import ANNMixin
from replay.ann.utils import save_index_to_destination_fs
from replay.session_handler import State
from replay.utils import FileSystem, get_filesystem

logger = logging.getLogger("replay")

INDEX_FILENAME = "hnswlib_index"


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
        self._efS = index_params.get("efS")
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

        self._index = hnswlib.Index(space=self._space, dim=self._dim)
        if self._index_path:
            if self._filesystem == FileSystem.HDFS:
                with tempfile.TemporaryDirectory() as temp_path:
                    tmp_file_path = os.path.join(
                        temp_path, INDEX_FILENAME
                    )
                    source_filesystem = fs.HadoopFileSystem.from_uri(
                        self._hdfs_uri
                    )
                    fs.copy_files(
                        self._index_path,
                        "file://" + tmp_file_path,
                        source_filesystem=source_filesystem,
                    )
                    self._index.load_index(tmp_file_path)
            else:
                self._index.load_index(self._index_path)
        else:
            self._index.load_index(SparkFiles.get(self._index_filename))

        if self._efS:
            self._index.set_ef(self._efS)
        return self._index


class HnswlibMixin(ANNMixin):
    """Mixin that provides methods to build hnswlib index and infer it.
    Also provides methods to saving and loading index to/from disk.
    """

    def _infer_ann_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: Dict[str, Union[int, str]],
        k: int,
        index_dim: str = None,
        index_type: str = None,
        log: DataFrame = None,
    ) -> DataFrame:
        return self._infer_hnsw_index(
            vectors, features_col, params, k, index_dim
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
        self._build_hnsw_index(
            vectors, features_col, params, dim, num_elements, id_col
        )

    def _build_hnsw_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        dim: int,
        num_elements: int,
        id_col: Optional[str] = None,
    ) -> None:
        """Builds hnsw index and dump it to hdfs or disk.

        Args:
            vectors: DataFrame with vectors. Schema: [{id_col}: int, {features_col}: array<float>]
            features_col: the name of the column in the `vectors` dataframe that contains features (vectors).
            params: index params
            dim: feature (vector) length
            num_elements: how many elements will be stored in the index
            id_col: the name of the column in the `vectors` dataframe that contains ids (of vectors)
        """

        if params["build_index_on"] == "executor":
            # to execution in one executor
            vectors = vectors.repartition(1)

            filesystem, hdfs_uri, index_path = get_filesystem(
                params["index_path"]
            )

            def build_index(iterator: Iterator[pd.DataFrame]):
                """Builds index on executor and writes it to shared disk or hdfs.

                Args:
                    iterator: iterates on dataframes with vectors/features

                """
                index = hnswlib.Index(space=params["space"], dim=dim)

                # Initializing index - the maximum number of elements should be known beforehand
                index.init_index(
                    max_elements=num_elements,
                    ef_construction=params["efC"],
                    M=params["M"],
                )

                # pdf is a pandas dataframe that contains ids and features (vectors)
                for pdf in iterator:
                    vectors_np = np.squeeze(pdf[features_col].values)
                    if id_col:
                        index.add_items(
                            np.stack(vectors_np), pdf[id_col].values
                        )
                    else:
                        # ids will be from [0, ..., len(vectors_np)]
                        index.add_items(np.stack(vectors_np))

                save_index_to_destination_fs(
                    index,
                    sparse=False,
                    save_index=lambda path: index.save_index(path),
                    filesystem=filesystem,
                    destination_path=index_path,
                    hdfs_uri=hdfs_uri,
                )

                yield pd.DataFrame(data={"_success": 1}, index=[0])

            # Here we perform materialization (`.collect()`) to build the hnsw index.
            logger.info("Started building the hnsw index")
            cols = [id_col, features_col] if id_col else [features_col]
            vectors.select(*cols).mapInPandas(
                build_index, "_success int"
            ).collect()
            logger.info("Finished building the hnsw index")
        else:
            vectors = vectors.toPandas()
            vectors_np = np.squeeze(vectors[features_col].values)

            index = hnswlib.Index(space=params["space"], dim=dim)

            # Initializing index - the maximum number of elements should be known beforehand
            index.init_index(
                max_elements=num_elements,
                ef_construction=params["efC"],
                M=params["M"],
            )

            if id_col:
                index.add_items(
                    np.stack(vectors_np), vectors[id_col].values
                )
            else:
                index.add_items(np.stack(vectors_np))

            # saving index to local temp file and sending it to executors
            temp_dir = tempfile.mkdtemp()
            weakref.finalize(self, shutil.rmtree, temp_dir)
            tmp_file_path = os.path.join(
                temp_dir, f"{INDEX_FILENAME}_{self._spark_index_file_uid}"
            )
            index.save_index(tmp_file_path)
            spark = SparkSession.getActiveSession()
            spark.sparkContext.addFile("file://" + tmp_file_path)

    def _update_hnsw_index(
        self,
        item_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        dim: int,
        num_elements: int,
    ):
        index = hnswlib.Index(space=params["space"], dim=dim)
        index_path = SparkFiles.get(f"{INDEX_FILENAME}_{self._spark_index_file_uid}")
        index.load_index(index_path, max_elements=num_elements)
        item_vectors = item_vectors.toPandas()
        item_vectors_np = np.squeeze(item_vectors[features_col].values)
        index.add_items(np.stack(item_vectors_np), item_vectors["id"].values)

        self._spark_index_file_uid = uuid.uuid4().hex[-12:]
        # saving index to local temp file and sending it to executors
        temp_dir = tempfile.mkdtemp()
        tmp_file_path = os.path.join(
            temp_dir, f"{INDEX_FILENAME}_{self._spark_index_file_uid}"
        )
        index.save_index(tmp_file_path)
        spark = SparkSession.getActiveSession()
        spark.sparkContext.addFile("file://" + tmp_file_path)

    def _infer_hnsw_index(
        self,
        vectors: DataFrame,
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
            _index_file_manager = HnswlibIndexFileManager(
                params,
                index_dim,
                index_filename=f"{INDEX_FILENAME}_{self._spark_index_file_uid}",
            )

        index_file_manager_broadcast = State().session.sparkContext.broadcast(
            _index_file_manager
        )

        return_type = (
            "item_idx array<int>, distance array<double>"
        )

        @pandas_udf(return_type)
        def infer_index(
            vectors: pd.Series, num_items: pd.Series
        ) -> pd.DataFrame:
            index_file_manager = index_file_manager_broadcast.value
            index = index_file_manager.index

            # max number of items to retrieve per batch
            max_items_to_retrieve = num_items.max()

            labels, distances = index.knn_query(
                np.stack(vectors.values), k=k + max_items_to_retrieve,
                num_threads=1
            )

            pd_res = pd.DataFrame(
                {"item_idx": list(labels), "distance": list(distances)}
            )

            return pd_res

        res = vectors.select(
            "user_idx",
            infer_index(features_col, "num_items").alias("r")
        )

        res = res.select(
            "user_idx",
            sf.explode(sf.arrays_zip("r.item_idx", "r.distance")).alias(
                "zip_exp"
            ),
        )

        # Fix arrays_zip random behavior. It can return zip_exp.0 or zip_exp.item_idx in different machines
        fields = res.schema["zip_exp"].jsonValue()["type"]["fields"]
        item_idx_field_name: str = fields[0]["name"]
        distance_field_name: str = fields[1]["name"]

        res = res.select(
            "user_idx",
            sf.col(f"zip_exp.{item_idx_field_name}").alias("item_idx"),
            (
                sf.lit(-1.0) * sf.col(f"zip_exp.{distance_field_name}")
            ).alias("relevance"),
        )

        return res

    def _save_hnswlib_index(self, path: str):
        """Method save (copy) index from hdfs (or local) to `path` directory.
        `path` can be a hdfs path or a local path.

        Args:
            path (str): directory where to dump (copy) the index
        """

        params = self._hnswlib_params

        if params["build_index_on"] == "executor":
            index_path = params["index_path"]
        elif params["build_index_on"] == "driver":
            index_path = SparkFiles.get(f"{INDEX_FILENAME}_{self._spark_index_file_uid}")
        else:
            raise ValueError("Unknown 'build_index_on' param.")

        from_filesystem, from_hdfs_uri, from_path = get_filesystem(
            index_path
        )
        to_filesystem, to_hdfs_uri, to_path = get_filesystem(path)
        self.logger.debug(f"Index file coping from '{index_path}' to '{path}'")

        if from_filesystem == FileSystem.HDFS:
            source_filesystem = fs.HadoopFileSystem.from_uri(from_hdfs_uri)
        else:
            source_filesystem = fs.LocalFileSystem()
        if to_filesystem == FileSystem.HDFS:
            destination_filesystem = fs.HadoopFileSystem.from_uri(to_hdfs_uri)
        else:
            destination_filesystem = fs.LocalFileSystem()

        fs.copy_files(
            from_path,
            os.path.join(to_path, INDEX_FILENAME),
            source_filesystem=source_filesystem,
            destination_filesystem=destination_filesystem,
        )
        # param use_threads=True (?)

    def _load_hnswlib_index(self, path: str):
        """Loads hnsw index from `path` directory to local dir.
        Index file name is 'hnswlib_index'.
        And adds index file to the `SparkFiles`.
        `path` can be a hdfs path or a local path.


        Args:
            path: directory path, where index file is stored
        """
        from_filesystem, from_hdfs_uri, from_path = get_filesystem(
            path + f"/{INDEX_FILENAME}"
        )

        to_path = tempfile.mkdtemp()
        weakref.finalize(self, shutil.rmtree, to_path)
        to_path = os.path.join(to_path, f"{INDEX_FILENAME}_{self._spark_index_file_uid}")

        if from_filesystem == FileSystem.HDFS:
            source_filesystem = fs.HadoopFileSystem.from_uri(from_hdfs_uri)
        else:
            source_filesystem = fs.LocalFileSystem()
        destination_filesystem = fs.LocalFileSystem()
        fs.copy_files(
            from_path,
            to_path,
            source_filesystem=source_filesystem,
            destination_filesystem=destination_filesystem,
        )

        spark = SparkSession.getActiveSession()
        spark.sparkContext.addFile("file://" + to_path)

        self._hnswlib_params["build_index_on"] = "driver"
