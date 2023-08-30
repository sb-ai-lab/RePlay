import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.pandas.functions import pandas_udf

from replay.models.extensions.ann.index_inferers.base_inferer import IndexInferer
from replay.models.extensions.ann.index_inferers.utils import (
    get_csr_matrix,
)
from replay.models.extensions.ann.utils import create_nmslib_index_instance
from replay.utils.session_handler import State


# pylint: disable=too-few-public-methods
class NmslibFilterIndexInferer(IndexInferer):
    """Nmslib index inferer with filter seen items. Infers nmslib hnsw index."""

    def infer(
        self, vectors: DataFrame, features_col: str, k: int
    ) -> DataFrame:
        _index_store = self.index_store
        index_params = self.index_params

        index_store_broadcast = State().session.sparkContext.broadcast(
            _index_store
        )

        @pandas_udf(self.udf_return_type)
        def infer_index_udf(  # pylint: disable=too-many-locals
            user_idx: pd.Series,
            vector_items: pd.Series,
            vector_relevances: pd.Series,
            num_items: pd.Series,
            seen_item_ids: pd.Series,
        ) -> pd.DataFrame:
            index_store = index_store_broadcast.value
            index = index_store.load_index(
                init_index=lambda: create_nmslib_index_instance(index_params),
                load_index=lambda index, path: index.loadIndex(
                    path, load_data=True
                ),
                configure_index=lambda index: index.setQueryTimeParams(
                    {"efSearch": index_params.ef_s}
                )
                if index_params.ef_s
                else None,
            )

            # max number of items to retrieve per batch
            max_items_to_retrieve = num_items.max()

            user_vectors = get_csr_matrix(
                user_idx, vector_items, vector_relevances
            )

            neighbours = index.knnQueryBatch(
                user_vectors[user_idx.values, :],
                k=k + max_items_to_retrieve,
                num_threads=1,
            )

            neighbours_filtered = []
            for i, (item_idxs, distances) in enumerate(neighbours):
                non_seen_item_indexes = ~np.isin(
                    item_idxs, seen_item_ids[i], assume_unique=True
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

        cols = [
            "user_idx",
            "vector_items",
            "vector_relevances",
            "num_items",
            "seen_item_idxs",
        ]
        # cols = cols + ["num_items", "seen_item_idxs"]

        res = vectors.select(
            "user_idx",
            infer_index_udf(*cols).alias("neighbours"),
        )

        res = self._unpack_infer_struct(res)

        return res
