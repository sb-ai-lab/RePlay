import logging
import shutil
import weakref
from abc import ABC
from typing import Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf

from replay.ann.ann_mixin import ANNMixin
from replay.ann.entities.hnswlib_param import HnswlibParam
from replay.ann.index_builders.driver_hnswlib_index_builder import (
    DriverHnswlibIndexBuilder,
)
from replay.ann.index_builders.executor_hnswlib_index_builder import (
    ExecutorHnswlibIndexBuilder,
)
from replay.ann.index_file_managers.hnswlib_index_file_manager import (
    HnswlibIndexFileManager,
)
from replay.session_handler import State
from replay.utils import get_filesystem

logger = logging.getLogger("replay")


class HnswlibMixin(ANNMixin, ABC):
    """Mixin that provides methods to build hnswlib index and infer it.
    Also provides methods to saving and loading index to/from disk.
    """

    _hnswlib_params: Optional[HnswlibParam] = None
    INDEX_FILENAME = "hnswlib_index"

    def _infer_ann_index(  # pylint: disable=too-many-arguments
        self,
        vectors: DataFrame,
        features_col: str,
        params: HnswlibParam,
        k: int,
        filter_seen_items: bool,
    ) -> DataFrame:
        return self._infer_hnsw_index(
            vectors, features_col, params, k, filter_seen_items
        )

    def _build_ann_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: HnswlibParam,
        id_col: Optional[str] = None,
    ) -> None:
        """Builds hnsw index and dump it to hdfs or shared disk,
        or sends it to executors through SparkContext.addFile().

        Args:
            vectors: DataFrame with vectors. Schema: [{id_col}: int, {features_col}: array<float>]
            features_col: the name of the column in the `vectors` dataframe
            that contains features (vectors).
            params: index params
            id_col: the name of the column in the `vectors` dataframe that contains ids (of vectors)
        """

        if params.build_index_on == "executor":
            index_builder = ExecutorHnswlibIndexBuilder()
        else:
            index_builder = DriverHnswlibIndexBuilder(
                index_file_name=f"{self.INDEX_FILENAME}_{self._spark_index_file_uid}"
            )
        temp_index_file_info = index_builder.build_index(
            vectors, features_col, params, id_col
        )
        if temp_index_file_info:
            weakref.finalize(self, shutil.rmtree, temp_index_file_info.path)

    def _infer_hnsw_index(  # pylint: disable=too-many-arguments
        self,
        vectors: DataFrame,
        features_col: str,
        params: HnswlibParam,
        k: int,
        filter_seen_items: bool,
    ):
        if params.build_index_on == "executor":
            index_file = get_filesystem(params.index_path)
        else:
            index_file = f"{self.INDEX_FILENAME}_{self._spark_index_file_uid}"

        _index_file_manager = HnswlibIndexFileManager(
            params,
            index_file=index_file,
        )

        index_file_manager_broadcast = State().session.sparkContext.broadcast(
            _index_file_manager
        )

        return_type = "item_idx array<int>, distance array<double>"

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

                labels, distances = index.knn_query(
                    np.stack(vectors.values),
                    k=k + max_items_to_retrieve,
                    num_threads=1,
                )

                filtered_labels = []
                filtered_distances = []
                for i, item_idxs in enumerate(labels):
                    non_seen_item_indexes = ~np.isin(
                        item_idxs, seen_item_idxs[i], assume_unique=True
                    )
                    filtered_labels.append(
                        (item_idxs[non_seen_item_indexes])[:k]
                    )
                    filtered_distances.append(
                        (distances[i][non_seen_item_indexes])[:k]
                    )

                pd_res = pd.DataFrame(
                    {
                        "item_idx": filtered_labels,
                        "distance": filtered_distances,
                    }
                )

                return pd_res

        else:

            @pandas_udf(return_type)
            def infer_index(vectors: pd.Series) -> pd.DataFrame:
                index_file_manager = index_file_manager_broadcast.value
                index = index_file_manager.index

                labels, distances = index.knn_query(
                    np.stack(vectors.values),
                    k=k,
                    num_threads=1,
                )

                pd_res = pd.DataFrame(
                    {"item_idx": list(labels), "distance": list(distances)}
                )

                return pd_res

        cols = []
        if filter_seen_items:
            cols = ["num_items", "seen_item_idxs"]

        res = vectors.select(
            "user_idx",
            infer_index(features_col, *cols).alias("neighbours"),
        )
        res = self._unpack_infer_struct(res)

        return res

    def _save_hnswlib_index(self, path: str):
        """Method save (copy) index from hdfs (or local) to `path` directory.
        `path` can be a hdfs path or a local path.

        Args:
            path (str): directory where to dump (copy) the index
        """

        self._save_index_files(path, self._hnswlib_params)

    def _load_hnswlib_index(self, path: str):
        """Loads hnsw index from `path` directory to local dir.
        Index file name is 'hnswlib_index'.
        And adds index file to the `SparkFiles`.
        `path` can be an hdfs path or a local path.


        Args:
            path: directory path, where index file is stored
        """

        self._load_index(path, self._hnswlib_params)
