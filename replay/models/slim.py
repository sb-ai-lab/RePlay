from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.linear_model import ElasticNet

from replay.data import Dataset
from replay.models.base_neighbour_rec import NeighbourRec
from replay.models.extensions.ann.index_builders.base_index_builder import IndexBuilder
from replay.utils import PYSPARK_AVAILABLE
from replay.utils.session_handler import State

if PYSPARK_AVAILABLE:
    from pyspark.sql import types as st


# pylint: disable=too-many-ancestors, too-many-instance-attributes
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

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        pandas_interactions = (
            dataset.interactions
            .select(self.query_column, self.item_column, self.rating_column)
            .toPandas()
        )

        interactions_matrix = csc_matrix(
            (
                pandas_interactions[self.rating_column],
                (
                    pandas_interactions[self.query_column],
                    pandas_interactions[self.item_column],
                ),
            ),
            shape=(self._query_dim, self._item_dim),
        )
        similarity = (
            State()
            .session.createDataFrame(pandas_interactions[self.item_column], st.IntegerType())
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
