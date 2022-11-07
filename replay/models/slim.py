from typing import Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from scipy.sparse import csc_matrix
from sklearn.linear_model import ElasticNet

from replay.models.base_rec import NeighbourRec
from replay.models.nmslib_hnsw import NmslibHnsw
from replay.session_handler import State


class SLIM(NeighbourRec, NmslibHnsw):
    """`SLIM: Sparse Linear Methods for Top-N Recommender Systems
    <http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf>`_"""

    _search_space = {
        "beta": {"type": "loguniform", "args": [1e-6, 5]},
        "lambda_": {"type": "loguniform", "args": [1e-6, 2]},
    }

    def __init__(
        self,
        beta: float = 0.01,
        lambda_: float = 0.01,
        seed: Optional[int] = None,
        nmslib_hnsw_params: Optional[dict] = None,
    ):
        """
        :param beta: l2 regularization
        :param lambda_: l1 regularization
        :param seed: random seed
        """
        if beta < 0 or lambda_ <= 0:
            raise ValueError("Invalid regularization parameters")
        self.beta = beta
        self.lambda_ = lambda_
        self.seed = seed
        self._nmslib_hnsw_params = nmslib_hnsw_params

    @property
    def _init_args(self):
        return {"beta": self.beta, "lambda_": self.lambda_, "seed": self.seed}

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        pandas_log = log.select("user_idx", "item_idx", "relevance").toPandas()

        interactions_matrix = csc_matrix(
            (pandas_log.relevance, (pandas_log.user_idx, pandas_log.item_idx)),
            shape=(self._user_dim, self._item_dim),
        )
        similarity = (
            State()
            .session.createDataFrame(pandas_log.item_idx, st.IntegerType())
            .withColumnRenamed("value", "item_idx_one")
        )

        alpha = self.beta + self.lambda_
        l1_ratio = self.lambda_ / alpha

        regression = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            max_iter=5000,
            random_state=self.seed,
            selection="random",
            positive=True,
        )

        def slim_column(pandas_df: pd.DataFrame) -> pd.DataFrame:
            """
            fit similarity matrix with ElasticNet
            :param pandas_df: pd.Dataframe
            :return: pd.Dataframe
            """
            idx = int(pandas_df["item_idx_one"][0])
            column = interactions_matrix[:, idx]
            column_arr = column.toarray().ravel()
            interactions_matrix[
                interactions_matrix[:, idx].nonzero()[0], idx
            ] = 0

            regression.fit(interactions_matrix, column_arr)
            interactions_matrix[:, idx] = column
            good_idx = np.argwhere(regression.coef_ > 0).reshape(-1)
            good_values = regression.coef_[good_idx]
            similarity_row = {
                "item_idx_one": good_idx,
                "item_idx_two": idx,
                "similarity": good_values,
            }
            return pd.DataFrame(data=similarity_row)

        self.similarity = similarity.groupby("item_idx_one").applyInPandas(
            slim_column,
            "item_idx_one int, item_idx_two int, similarity double",
        )
        self.similarity.cache().count()

        if self._nmslib_hnsw_params:
            
            similarity_df = self.similarity.select("similarity", 'item_idx_one', 'item_idx_two')
            self._build_hnsw_index(similarity_df, None, self._nmslib_hnsw_params, index_type="sparse")

            self._max_items_to_retrieve, *_ = (
                    log.groupBy('user_idx')
                    .agg(sf.count('item_idx').alias('num_items'))
                    .select(sf.max('num_items'))
                    .first()
            )


    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        
        if self._nmslib_hnsw_params:

            params = self._nmslib_hnsw_params

            # with JobGroup(
            #     "self._get_user_vectors()",
            #     "_predict (inside 1)",
            # ):
            #     user_vectors = self._get_user_vectors(users, log)
            #     user_vectors = user_vectors.cache()
            #     user_vectors.write.mode("overwrite").format("noop").save()
            

            # with JobGroup(
            #     "select vector_to_array",
            #     "_predict (inside 2)",
            # ):
            #     # converts to pandas_udf compatible format
            #     user_vectors = (
            #             user_vectors
            #             .select(
            #                 "user_idx",
            #                 vector_to_array("user_vector").alias("user_vector")
            #             )
            #     )
            #     user_vectors = user_vectors.cache()
            #     user_vectors.write.mode("overwrite").format("noop").save()

            items_count = log.select(sf.max('item_idx')).first()[0] + 1 # .distinct().count()
            max_user_id = log.select(sf.max('user_idx')).first()[0]
            test_unique_user_idx = users.select('user_idx').distinct().rdd.flatMap(list).collect()
            res = self._infer_hnsw_index(log, log.select("user_idx", "item_idx"), "", 
                params, k, filter_seen_items, 
                index_type="sparse", max_user_id=max_user_id, items_count=items_count,
                test_unique_user_idx=test_unique_user_idx)

            return res

            


