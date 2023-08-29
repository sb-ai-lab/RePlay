import numpy as np
import pandas as pd

from typing import Optional
from pyspark.sql import DataFrame
from replay.models.base_rec import HybridRecommender
from replay.utils import convert2spark

class uLinUCB(HybridRecommender):
    """
    A recommender implicitly proposed by 
    `Song et al <https://arxiv.org/abs/2110.09905>`_.
    Is used as the default node recommender in :class:`HierarchicalRecommender`.
    Shares all the logic with classical item-disjoint :class:`LinUCB` but is 
    user-disjoint instead. May be useful in problems with fixed number of users
    and item-oriented data.
    """ 

    def __init__(
        self,
        alpha: float = 0.5,
    ):
        """
        :param alpha: exploration coefficient
        """
        
        self._alpha = alpha
        super().__init__()
    
    @property
    def _init_args(self):
        return {
            "alpha": self._alpha,
        }
    
    def _fit(
        self,
        log: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
    ) -> None:    
        
        # prepare data
        log = log.drop("timestamp").toPandas()
        
        user_features = user_features.orderBy("user_idx").drop("user_idx").toPandas()
        item_features = item_features.orderBy("item_idx").drop("item_idx").toPandas()

        self._num_users, self._num_user_features = user_features.shape
        self._num_items, self._num_item_features = item_features.shape

        self._init_params()

        # main fit loop
        for user_idx, user_batch in log.groupby("user_idx"):    
            self._update_params(user_idx,
                                user_batch["item_idx" ].astype( int ).to_numpy(),
                                user_batch["relevance"].astype(float).to_numpy(),
                                          item_features.astype(float).to_numpy())
    
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        
        K = 10*k 

        user_idx = users.toPandas()["user_idx"].astype(int).to_numpy()

        pd_ucb = pd.DataFrame(self.get_relevance(user_idx), index=user_idx)

        pred_df = pd_ucb.stack().reset_index()
        pred_df.columns = ["user_idx", "item_idx", "relevance"]

        pred_df = pred_df.sort_values(["user_idx", "relevance"], ascending=False
                        ).groupby("user_idx"
                        ).head(K)
        
        return convert2spark(pred_df)
    
    
    def _init_params(self) -> None:
        
        self._th  = np.zeros((self._num_users, self._num_item_features))
        self._b   = np.zeros(self._num_item_features)
        self._A   = np.eye  (self._num_item_features)
        self._ucb = np.zeros((self._num_users, self._num_items))

    def _update_params(
        self,
        user_idx:       int,
        items_idx:      np.ndarray,
        rewards:        np.ndarray,
        item_features:  np.ndarray,
    ) -> None:
        
        self._A = self._A + item_features[items_idx].T @ item_features[items_idx]
        self._b = self._b + item_features[items_idx].T @ rewards
        self._th[user_idx] = np.linalg.inv(self._A) @ self._b

        self._ucb[user_idx] = self._th[user_idx] @ item_features.T \
            + self._alpha \
            * np.sqrt(np.sum(item_features.T * (np.linalg.inv(self._A) @ item_features.T), axis=0))

    def get_relevance(self, user_idx): # counts ucb as relevance
        return self._ucb[user_idx]