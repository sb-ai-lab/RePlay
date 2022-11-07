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

from replay.utils import FileSystem, JobGroup, get_filesystem

logger = logging.getLogger("replay")


class NmslibHnsw:
    def _build_hnsw_index(
        self,
        item_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        index_type: str = None
    ):
        """ "Builds hnsw index and dump it to hdfs or disk.

        Args:
            item_vectors (DataFrame): DataFrame with item vectors
            params (Dict[str, Any]): hnsw params
        """

        if params["build_index_on"] == "executor":
            # to execution in one executor
            item_vectors = item_vectors.repartition(1)

            filesystem, hdfs_uri, index_path = get_filesystem(params["index_path"])

            if index_type == "sparse":
                def build_index(iterator):
                    index = nmslib.init(
                            method=params["method"],
                            space=params["space"], # cosinesimil_sparse negdotprod_sparse
                            data_type=nmslib.DataType.SPARSE_VECTOR
                    )
                    
                    pdfs = []
                    for pdf in iterator:
                        pdfs.append(pdf)
                    
                    pdf = pd.concat(pdfs)
                    
                    data = pdf['similarity'].values
                    row_ind = pdf['item_idx_two'].values # 'item_idx_one'
                    col_ind = pdf['item_idx_one'].values # 'item_idx_two'

                    # M = pdf['item_idx_one'].max() + 1
                    # N = pdf['item_idx_two'].max() + 1

                    sim_matrix_tmp = csr_matrix((data, (row_ind, col_ind))) # , shape=(M, N)
                    index.addDataPointBatch(
                        data=sim_matrix_tmp
                    )

                    print(f"max(rowIds): {row_ind.max()}")
                    print(f"max(col_ind): {col_ind.max()}")
                    print(f"len(index): {len(index)}")

                    index.createIndex({
                        'M': params["M"],
                        'efConstruction': params["efC"],
                        'post' : params["post"]
                    })
                    
                    if filesystem == FileSystem.HDFS:
                        temp_path = tempfile.mkdtemp()
                        tmp_file_path = os.path.join(
                            temp_path, "nmslib_hnsw_index"
                        )
                        index.saveIndex(tmp_file_path, save_data=True)

                        destination_filesystem = fs.HadoopFileSystem.from_uri(hdfs_uri)
                        fs.copy_files(
                            "file://" + tmp_file_path, index_path,
                            destination_filesystem=destination_filesystem
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
                        item_vectors_np = np.squeeze(pdf[features_col].values)
                        index.addDataPointBatch(
                            data=np.stack(item_vectors_np),
                            ids=pdf["item_idx"].values,
                        )
                    index.createIndex({
                        'M': params["M"],
                        'efConstruction': params["efC"],
                        'post' : params["post"]
                    })

                    if filesystem == FileSystem.HDFS:
                        temp_path = tempfile.mkdtemp()
                        tmp_file_path = os.path.join(
                            temp_path, "nmslib_hnsw_index"
                        )
                        index.saveIndex(tmp_file_path)

                        destination_filesystem = fs.HadoopFileSystem.from_uri(hdfs_uri)
                        fs.copy_files(
                            "file://" + tmp_file_path, index_path,
                            destination_filesystem=destination_filesystem
                        )

                        # hdfs = fs.HadoopFileSystem.from_uri(hdfs_uri)
                        # fs.copy_files("file://" + tmp_file_path, index_path, destination_filesystem=hdfs)
                        # param use_threads=True (?)
                    else:
                        index.saveIndex(index_path)

                    yield pd.DataFrame(data={"_success": 1}, index=[0])

            if index_type == "sparse":
                item_vectors.select('similarity', 'item_idx_one', 'item_idx_two').mapInPandas(
                    build_index, "_success int"
                ).show()
            else:
                item_vectors.select("item_idx", features_col).mapInPandas(
                    build_index, "_success int"
                ).show()
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
            index.createIndex()

            # saving index to local temp file and sending it to executors
            temp_path = tempfile.mkdtemp()
            tmp_file_path = os.path.join(temp_path, "nmslib_hnsw_index")
            index.saveIndex(tmp_file_path)
            spark = SparkSession.getActiveSession()
            # self._broadcast_index = spark.sparkContext.broadcast(index)
            spark.sparkContext.addFile("file://" + tmp_file_path)

    def _filter_seen_hnsw_res(
        self, log: DataFrame, pred: DataFrame, k: int, id_type="idx"
    ):
        """
        filter items seen in log and leave top-k most relevant
        """

        user_id = "user_" + id_type
        item_id = "item_" + id_type

        recs = pred.join(log, on=[user_id, item_id], how="anti")

        recs = (
            recs.withColumn(
                "temp_rank",
                sf.row_number().over(
                    Window.partitionBy(user_id).orderBy(
                        sf.col("relevance").desc()
                    )
                ),
            )
            .filter(sf.col("temp_rank") <= k)
            .drop("temp_rank")
        )

        return recs

    def _infer_hnsw_index(
        self,
        log: DataFrame,
        user_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        k: int,
        filter_seen_items: bool = True,
        index_type: str = None,
        max_user_id: Optional[int] = None,
        items_count: Optional[int] = None,
        test_unique_user_idx: Optional[int] = None
    ):

        if params["build_index_on"] == "executor":
            filesystem, hdfs_uri, index_path = get_filesystem(params["index_path"])

        k_udf = k + self._max_items_to_retrieve
        return_type = "item_idx array<int>, distance array<double>, user_idx int"

        if index_type == "sparse":
            # test_unique_user_idx = 
            @pandas_udf(return_type)
            def infer_index(
                user_idx: pd.Series, item_idx: pd.Series
            ) -> pd.DataFrame:
                index = nmslib.init(
                    method=params["method"],
                    space=params["space"],
                    data_type=nmslib.DataType.SPARSE_VECTOR,
                )
                if params["build_index_on"] == "executor":
                    if filesystem == FileSystem.HDFS:
                        temp_path = tempfile.mkdtemp()
                        tmp_file_path = os.path.join(
                            temp_path, "nmslib_hnsw_index"
                        )
                        source_filesystem = fs.HadoopFileSystem.from_uri(hdfs_uri)
                        fs.copy_files(
                            index_path,
                            "file://" + tmp_file_path,
                            source_filesystem=source_filesystem
                        )
                        index.loadIndex(tmp_file_path, load_data=True)
                    else:
                        index.loadIndex(index_path, load_data=True)
                else:
                    index.loadIndex(SparkFiles.get("nmslib_hnsw_index"))

                index.setQueryTimeParams({'efSearch': params["efS"]})
                ones = pd.Series(1 for _ in range(len(user_idx)))
                print(f"len(ones): {len(ones)}")
                print(f"len(user_idx): {len(user_idx)}")
                print(f"len(item_idx): {len(item_idx)}")
                print(f"max_user_id: {max_user_id}")
                print(f"items_count: {items_count}")
                interactions_matrix = csr_matrix(
                    (ones, (user_idx, item_idx)),
                    shape=(max_user_id+1, items_count), # user_idx.max()+1
                )
                neighbours = index.knnQueryBatch(interactions_matrix, k=k_udf)
                pd_res = pd.DataFrame(neighbours, columns=['item_idx', 'distance'])
                pd_res['user_idx'] = [x for x in range(interactions_matrix.shape[0])]
                pd_res = pd_res[pd_res['user_idx'].isin(test_unique_user_idx)]
                # which is better?
                # pd_res['user_idx'] = user_ids_list
                # pd_res = pd_res.assign(user_idx=user_idx.values)

                return pd_res
        else:
            @pandas_udf(return_type)
            def infer_index(
                user_ids_list: pd.Series, vectors: pd.Series
            ) -> pd.DataFrame:
                index = nmslib.init(
                    method=params["method"],
                    space=params["space"],
                    data_type=nmslib.DataType.DENSE_VECTOR,
                )
                if params["build_index_on"] == "executor":
                    if filesystem == FileSystem.HDFS:
                        temp_path = tempfile.mkdtemp()
                        tmp_file_path = os.path.join(
                            temp_path, "nmslib_hnsw_index"
                        )
                        source_filesystem = fs.HadoopFileSystem.from_uri(hdfs_uri)
                        fs.copy_files(
                            index_path,
                            "file://" + tmp_file_path,
                            source_filesystem=source_filesystem
                        )
                        index.loadIndex(tmp_file_path)
                    else:
                        index.loadIndex(index_path)
                else:
                    index.loadIndex(SparkFiles.get("nmslib_hnsw_index"))

                index.setQueryTimeParams({'efSearch': params["efS"]})
                neighbours = index.knnQueryBatch(np.stack(vectors.values), k=k_udf)
                pd_res = pd.DataFrame(neighbours, columns=["item_idx", "distance"])
                # which is better?
                # pd_res['user_idx'] = user_ids_list
                pd_res = pd_res.assign(user_idx=user_ids_list.values)

                return pd_res

        with JobGroup(
            "infer_index()",
            "infer_hnsw_index (inside 1)",
        ):
            if index_type == "sparse":
                res = user_vectors.select(
                    infer_index("user_idx", "item_idx").alias("r")
                )
            else:
                res = user_vectors.select(
                    infer_index("user_idx", features_col).alias("r")
                )
            res = res.cache()
            res.write.mode("overwrite").format("noop").save()

        with JobGroup(
            "res.withColumn('zip_exp', ...",
            "infer_hnsw_index (inside 2)",
        ):
            res = res.withColumn(
                "zip_exp",
                sf.explode(sf.arrays_zip("r.item_idx", "r.distance")),
            ).select(
                sf.col("r.user_idx").alias("user_idx"),
                sf.col("zip_exp.item_idx").alias("item_idx"),
                (sf.lit(-1.0) * sf.col("zip_exp.distance")).alias(
                    "relevance"
                ),  # -1
            )
            res = res.cache()
            res.write.mode("overwrite").format("noop").save()

        if filter_seen_items:
            with JobGroup(
                "filter_seen_hnsw_res()",
                "infer_hnsw_index (inside 3)",
            ):
                res = self._filter_seen_hnsw_res(log, res, k)
                res = res.cache()
                res.write.mode("overwrite").format("noop").save()
        else:
            res = res.cache()

        return res
   

    def _save_nmslib_hnsw_index(self, path):
        """Method save (copy) index from hdfs (or local) to `path` directory.
        `path` can be an hdfs path or a local path.

        Args:
            path (_type_): directory where to dump (copy) the index
        """
        
        params = self._nmslib_hnsw_params

        from_filesystem, from_hdfs_uri, from_path = get_filesystem(params["index_path"])
        to_filesystem, to_hdfs_uri, to_path = get_filesystem(path)

        if from_filesystem == FileSystem.HDFS:
            source_filesystem = fs.HadoopFileSystem.from_uri(from_hdfs_uri)
            if to_filesystem == FileSystem.HDFS:
                destination_filesystem = fs.HadoopFileSystem.from_uri(to_hdfs_uri)
                fs.copy_files(
                    from_path,
                    os.path.join(to_path, "nmslib_hnsw_index"),
                    source_filesystem=source_filesystem,
                    destination_filesystem=destination_filesystem
                )
            else:
                destination_filesystem = fs.LocalFileSystem()
                fs.copy_files(
                    from_path,
                    os.path.join(to_path, "nmslib_hnsw_index"),
                    source_filesystem=source_filesystem,
                    destination_filesystem=destination_filesystem
                )
        else:
            source_filesystem = fs.LocalFileSystem()
            if to_filesystem == FileSystem.HDFS:
                destination_filesystem = fs.HadoopFileSystem.from_uri(to_hdfs_uri)
                fs.copy_files(
                    from_path,
                    os.path.join(to_path, "nmslib_hnsw_index"),
                    source_filesystem=source_filesystem,
                    destination_filesystem=destination_filesystem
                )
            else:
                destination_filesystem = fs.LocalFileSystem()
                fs.copy_files(
                    from_path,
                    os.path.join(to_path, "nmslib_hnsw_index"),
                    source_filesystem=source_filesystem,
                    destination_filesystem=destination_filesystem
                )

            # param use_threads=True (?)