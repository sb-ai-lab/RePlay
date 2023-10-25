from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import types as st
from scipy.sparse import csc_matrix
from sklearn.linear_model import ElasticNet

from replay.models.extensions.ann.index_builders.base_index_builder import IndexBuilder
from replay.models.base_neighbour_rec import NeighbourRec
from replay.utils.session_handler import State

from replay.data import Dataset


# pylint: disable=too-many-ancestors
class SLIM(NeighbourRec):
    """`SLIM: Sparse Linear Methods for Top-N Recommender Systems
    <http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf>`_"""

    def _get_ann_infer_params(self) -> Dict[str, Any]:
        return {
            "features_col": None,
        }

    _search_space = {
        "beta": {"type": "loguniform", "args": [1e-6, 5]},
        "lambda_": {"type": "loguniform", "args": [1e-6, 2]},
    }

    def __init__(
        self,
        beta: float = 0.01,
        lambda_: float = 0.01,
        seed: Optional[int] = None,
        index_builder: Optional[IndexBuilder] = None,
    ):
        """
        :param beta: l2 regularization
        :param lambda_: l1 regularization
        :param seed: random seed
        :param index_builder: `IndexBuilder` instance that adds ANN functionality.
            If not set, then ann will not be used.
        """
        if beta < 0 or lambda_ <= 0:
            raise ValueError("Invalid regularization parameters")
        self.beta = beta
        self.lambda_ = lambda_
        self.seed = seed
        if isinstance(index_builder, (IndexBuilder, type(None))):
            self.index_builder = index_builder
        elif isinstance(index_builder, dict):
            self.init_builder_from_dict(index_builder)

    @property
    def _init_args(self):
        return {
            "beta": self.beta,
            "lambda_": self.lambda_,
            "seed": self.seed,
            "index_builder": self.index_builder.init_meta_as_dict() if self.index_builder else None,
        }

    def _save_model(self, path: str):
        if self._use_ann:
            self._save_index(path)

    def _load_model(self, path: str):
        if self._use_ann:
            self._load_index(path)

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        pandas_log = dataset.interactions.select(self.query_col, self.item_col, self.rating_col).toPandas()

        interactions_matrix = csc_matrix(
            (pandas_log[self.rating_col], (pandas_log[self.query_col], pandas_log[self.item_col])),
            shape=(self._user_dim, self._item_dim),
        )
        similarity = (
            State()
            .session.createDataFrame(pandas_log[self.item_col], st.IntegerType())
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

        def slim_column(pandas_df: pd.DataFrame) -> pd.DataFrame:   # pragma: no cover
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
