import numpy as np
import pandas as pd

from replay.models.extensions.ann.index_inferers.base_inferer import IndexInferer
from replay.models.extensions.ann.utils import create_hnswlib_index_instance
from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame, SparkDataFrame
from replay.utils.session_handler import State

if PYSPARK_AVAILABLE:
    from pyspark.sql.pandas.functions import pandas_udf


# pylint: disable=too-few-public-methods
class HnswlibIndexInferer(IndexInferer):
    """Hnswlib index inferer without filter seen items. Infers hnswlib index."""

    def infer(
        self, vectors: SparkDataFrame, features_col: str, k: int
    ) -> SparkDataFrame:
        _index_store = self.index_store
        index_params = self.index_params

        index_store_broadcast = State().session.sparkContext.broadcast(
            _index_store
        )

        @pandas_udf(self.udf_return_type)
        def infer_index_udf(vectors: pd.Series) -> PandasDataFrame:
            index_store = index_store_broadcast.value
            index = index_store.load_index(
                init_index=lambda: create_hnswlib_index_instance(index_params),
                load_index=lambda index, path: index.load_index(path),
                configure_index=lambda index: index.set_ef(index_params.ef_s)
                if index_params.ef_s
                else None,
            )

            labels, distances = index.knn_query(
                np.stack(vectors.values),
                k=k,
                num_threads=1,
            )

            pd_res = pd.DataFrame(
                {"item_idx": list(labels), "distance": list(distances)}
            )

            return pd_res

        res = vectors.select(
            "user_idx",
            infer_index_udf(features_col).alias("neighbours"),
        )

        res = self._unpack_infer_struct(res)

        return res
