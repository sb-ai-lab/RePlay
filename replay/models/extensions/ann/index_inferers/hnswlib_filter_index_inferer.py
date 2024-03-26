import numpy as np
import pandas as pd

from replay.models.extensions.ann.utils import create_hnswlib_index_instance
from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame, SparkDataFrame
from replay.utils.session_handler import State

from .base_inferer import IndexInferer

if PYSPARK_AVAILABLE:
    from pyspark.sql.pandas.functions import pandas_udf


class HnswlibFilterIndexInferer(IndexInferer):
    """Hnswlib index inferer with filter seen items. Infers hnswlib index."""

    def infer(self, vectors: SparkDataFrame, features_col: str, k: int) -> SparkDataFrame:
        _index_store = self.index_store
        index_params = self.index_params

        index_store_broadcast = State().session.sparkContext.broadcast(_index_store)

        @pandas_udf(self.udf_return_type)
        def infer_index_udf(
            vectors: pd.Series,
            num_items: pd.Series,
            seen_item_ids: pd.Series,
        ) -> PandasDataFrame:  # pragma: no cover
            index_store = index_store_broadcast.value
            index = index_store.load_index(
                init_index=lambda: create_hnswlib_index_instance(index_params),
                load_index=lambda index, path: index.load_index(path),
                configure_index=lambda index: index.set_ef(index_params.ef_s) if index_params.ef_s else None,
            )

            # max number of items to retrieve per batch
            max_items_to_retrieve = num_items.max()

            labels, distances = index.knn_query(
                np.stack(vectors.values),
                k=k + max_items_to_retrieve,
                num_threads=1,
            )

            filtered_labels = []
            filtered_distances = []
            for i, item_ids in enumerate(labels):
                non_seen_item_indexes = ~np.isin(item_ids, seen_item_ids[i], assume_unique=True)
                filtered_labels.append((item_ids[non_seen_item_indexes])[:k])
                filtered_distances.append((distances[i][non_seen_item_indexes])[:k])

            pd_res = pd.DataFrame(
                {
                    "item_idx": filtered_labels,
                    "distance": filtered_distances,
                }
            )

            return pd_res

        cols = ["num_items", "seen_item_idxs"]

        res = vectors.select(
            "user_idx",
            infer_index_udf(features_col, *cols).alias("neighbours"),
        )
        res = self._unpack_infer_struct(res)

        return res
