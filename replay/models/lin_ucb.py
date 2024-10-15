from typing import Optional, Tuple

import numpy as np
import pandas as pd
import os
import scipy.sparse as scs
from tqdm import tqdm

from replay.data.dataset import Dataset
from replay.utils import SparkDataFrame
from replay.utils.spark_utils import convert2spark

from .base_rec import HybridRecommender


# Object for interactions with a single arm in a UCB disjoint framework
class DisjointArm:
    def __init__(
        self, arm_index, d, eps, alpha
    ):  # in case of lin ucb with disjoint features: d = dimension of user's features solely
        # Track arm index
        self.arm_index = arm_index
        # Exploration parameter
        self.eps = eps
        self.alpha = alpha
        # Inverse of feature matrix for ridge regression
        self.A = self.alpha * np.identity(d)
        self.A_inv = (1.0 / self.alpha) * np.identity(d)
        # right-hand side of the regression
        self.theta = np.zeros(d, dtype=float)

    def feature_update(self, usr_features, relevances):
        """
        function to update featurs or each Lin-UCB hand in the current model
        features:
            usr_features = matrix (np.array of shape (m,d)),
                where m = number of occurences of the current feature in the dataset;
            usr_features[i] = features of i-th user, who rated this particular arm (movie);
            relevances = np.array(d) - rating of i-th user, who rated this particular arm (movie);
        """
        # Update A which is (d * d) matrix.
        self.A += np.dot(usr_features.T, usr_features)
        self.A_inv = np.linalg.inv(self.A)
        # Update the parameter theta by the results linear regression
        self.theta = np.linalg.lstsq(self.A, usr_features.T @ relevances, rcond=1.0)[0]
        self.cond_number = np.linalg.cond(self.A)  # this ome needed for deug only


# Object for interactions with a single arm in a UCB hybrid framework
class HybridArm:
    def __init__(self, arm_index, d, k, eps, alpha):
        # Track arm index
        self.arm_index = arm_index
        # Exploration parameter
        self.eps = eps
        self.alpha = alpha
        # Inverse of feature matrix for ridge regression
        self.A = scs.csr_matrix(self.alpha * np.identity(d))
        self.A_inv = scs.csr_matrix((1.0 / self.alpha) * np.identity(d))
        self.B = scs.csr_matrix(np.zeros((d, k)))
        # right-hand side of the regression
        self.b = np.zeros(d, dtype=float)

    def feature_update(self, usr_features, usr_itm_features, relevances) -> Tuple[np.ndarray, np.ndarray]:
        """
        function to update featurs or each Lin-UCB hand in the current model
        features:
            usr_features = matrix (np.array of shape (m,d)),
                where m = number of occurences of the current feature in the dataset;
            usr_features[i] = features of i-th user, who rated this particular arm (movie);
            relevances = np.array(d) - rating of i-th user, who rated this particular arm (movie);
        """

        self.A += (usr_features.T).dot(usr_features)
        self.A_inv = scs.linalg.inv(self.A)
        self.B += (usr_features.T).dot(usr_itm_features)
        self.b += (usr_features.T).dot(relevances)
        delta_A_0 = np.dot(usr_itm_features.T, usr_itm_features)- self.B.T @ self.A_inv @ self.B  # noqa: N806
        delta_b_0 = (usr_itm_features.T).dot(relevances) - (self.B.T).dot(self.A_inv.dot(self.b))
        return delta_A_0, delta_b_0


class LinUCB(HybridRecommender):
    """
    A recommender algorithm for contextual bandit problems,
    implicitly proposed by `Li et al <https://arxiv.org/pdf/1003.0146>`_.
    The model assumes a linear relationship between user context, item features and action rewards,
    making it efficient for high-dimensional contexts.

    >>> import pandas as pd
    >>> from replay.data.dataset import Dataset, FeatureSchema, FeatureInfo, FeatureHint, FeatureType
    >>> from replay.utils.spark_utils import convert2spark
    >>> data_frame = pd.DataFrame({"user_id": [0, 1, 2, 2], "item_id": [0, 1, 0, 1], "rating": [1, 0, 0, 0]})
    >>> user_features = pd.DataFrame(
    >>>     {"user_id": [0, 1, 2], "usr_feat_1": [1, 2, 3], "usr_feat_2": [4, 5, 6], "usr_feat_3": [7, 8, 9]}
    >>> )
    >>> item_features = pd.DataFrame(
    >>>     {
    >>>         "item_id": [0, 1, 2, 3, 4, 5],
    >>>         "itm_feat_1": [1, 2, 3, 4, 5, 6],
    >>>         "itm_feat_2": [7, 8, 9, 10, 11, 12],
    >>>         "itm_feat_3": [13, 14, 15, 16, 17, 18]
    >>>     }
    >>> )
    >>> interactions = convert2spark(data_frame)
    >>> user_features = convert2spark(user_features)
    >>> item_features = convert2spark(item_features)
    >>> feature_schema = FeatureSchema(
    ...    [
    ...        FeatureInfo(
    ...            column="user_id",
    ...            feature_type=FeatureType.CATEGORICAL,
    ...            feature_hint=FeatureHint.QUERY_ID,
    ...        ),
    ...        FeatureInfo(
    ...            column="item_id",
    ...            feature_type=FeatureType.CATEGORICAL,
    ...            feature_hint=FeatureHint.ITEM_ID,
    ...        ),
    ...        FeatureInfo(
    ...            column="rating",
    ...            feature_type=FeatureType.NUMERICAL,
    ...            feature_hint=FeatureHint.RATING,
    ...        ),
    ...    ]
    ... )
    >>> dataset = Dataset(
    ...     feature_schema=feature_schema,
    ...     interactions=interactions,
    ...     item_features=item_features,
    ...     query_features=user_features,
    ...     categorical_encoded = True
    ... )
    >>> model = LinUCB(eps = -10.0, alpha = 1.0, regr_type = 'disjoint')
    >>> model.fit(dataset)
    >>> model.predict(dataset, k=2, queries=[0,1,2]).toPandas().sort_values(["user_id","rating","item_id"],
    ... ascending=[True,False,True]).reset_index(drop=True)

        user_id   item_id     rating
    0         0         1   -11.073741
    1         0         2   -81.240384
    2         1         0   -6.555529
    3         1         2   -96.436508
    4         2         2   -112.249722
    5         2         3   -112.249722

    """

    def __init__(
        self,
        eps: float,
        alpha: float,
        regr_type: str,
        random_state: Optional[int] = None,
    ):  # pylint: disable=too-many-arguments
        """
        :param eps: exploration coefficient
        :param alpha: ridge parameter
        :param regr_type: type of model "disjoint" or "hybrid"
        :param random_state: random seed. Provides reproducibility if fixed
        """
        np.random.seed(42)
        self.regr_type = regr_type
        self.random_state = random_state
        self.eps = eps
        self.alpha = alpha
        self.linucb_arms = None  # initialize only when working within fit method
        cpu_count = os.cpu_count()
        self.num_threads = cpu_count if cpu_count is not None else 1

        self._study = None  # field required for proper optuna's optimization
        self._search_space = {
            "eps": {"type": "uniform", "args": [-10.0, 10.0]},
            "alpha": {"type": "uniform", "args": [0.001, 10.0]},
        }

    @property
    def _init_args(self):
        return {
            "regression type": self.regr_type,
            "seed": self.random_state,
        }

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        feature_schema = dataset.feature_schema
        # should not work if user features or item features are unavailable
        if dataset.query_features is None:
            msg = "User features are missing for fitting"
            raise ValueError(msg)
        if dataset.item_features is None:
            msg = "Item features are missing for fitting"
            raise ValueError(msg)
        # assuming that user_features and item_features are both dataframes
        # now forget about pyspark until the better times
        log = dataset.interactions.toPandas()
        user_features = dataset.query_features.toPandas()
        item_features = dataset.item_features.toPandas()
        # check that the dataframe contains uer indexes
        if feature_schema.query_id_column not in user_features.columns:
            msg = "User indices are missing in user features dataframe"
            raise ValueError(msg)
        self._num_items = item_features.shape[0]
        self._user_dim_size = user_features.shape[1] - 1
        self._item_dim_size = item_features.shape[1] - 1
        # now initialize an arm object for each potential arm instance
        if self.regr_type == "disjoint":
            self.linucb_arms = [
                DisjointArm(arm_index=i, d=self._user_dim_size, eps=self.eps, alpha=self.alpha)
                for i in range(self._num_items)
            ]
            # now we work with pandas
            for i in range(self._num_items):
                B = log.loc[log[feature_schema.item_id_column] == i]  # noqa: N806
                rel_list = B[feature_schema.interactions_rating_column].values
                if not B.empty:
                    # if we have at least one user interacting with the hand i
                    cur_usrs = user_features.query(
                        f"{feature_schema.query_id_column} in @B[feature_schema.query_id_column].values"
                    ).drop(columns=[feature_schema.query_id_column])
                    self.linucb_arms[i].feature_update(cur_usrs.to_numpy(), rel_list)

        elif self.regr_type == "hybrid":
            k = self._user_dim_size * self._item_dim_size
            self.A_0 = scs.csr_matrix(np.identity(k))
            self.b_0 = np.zeros(k, dtype=float)
            self.linucb_arms = [
                HybridArm(
                    arm_index=i,
                    d=self._user_dim_size,
                    k=k,
                    eps=self.eps,
                    alpha=self.alpha,
                )
                for i in range(self._num_items)
            ]

            # now we work with pandas
            for i in tqdm(range(self._num_items)):
                B = log.loc[log[feature_schema.item_id_column] == i]
                rel_list = B[feature_schema.interactions_rating_column].values
                if not B.empty:
                    # if we have at least one user interacting with the hand i
                    cur_usrs = scs.csr_matrix(
                        user_features.query(
                            f"{feature_schema.query_id_column} in @B[feature_schema.query_id_column].values"
                        )
                        .drop(columns=[feature_schema.query_id_column])
                        .to_numpy()
                    )
                    cur_itm = scs.csr_matrix(
                        item_features.iloc[i].drop(labels=[feature_schema.item_id_column]).to_numpy()
                    )
                    usr_itm_features = scs.kron(cur_usrs, cur_itm)
                    delta_A_0, delta_b_0 = self.linucb_arms[i].feature_update(cur_usrs, usr_itm_features, rel_list)  # noqa: N806

                    self.A_0 += delta_A_0
                    self.b_0 += delta_b_0

            self.beta = scs.linalg.spsolve(self.A_0, self.b_0)
            self.A_0_inv = scs.linalg.inv(self.A_0)

            for i in range(self._num_items):
                self.linucb_arms[i].theta = scs.linalg.spsolve(
                    self.linucb_arms[i].A,
                    self.linucb_arms[i].b - self.linucb_arms[i].B @ self.beta,
                )

        else:
            msg = "model.regr_type must be in ['disjoint', 'hybrid']"
            raise ValueError(msg)
        return

    def _predict(
        self,
        dataset: Dataset,
        k: int,
        users: SparkDataFrame,
        items: SparkDataFrame = None,
        filter_seen_items: bool = True,  # noqa: ARG002
        oversample: int = 20,
    ) -> SparkDataFrame:
        if self.regr_type == "disjoint":
            feature_schema = dataset.feature_schema
            num_user_pred = users.count()  # assuming it is a pyspark dataset
            users = users.toPandas()
            user_features = dataset.query_features.toPandas()
            idxs_list = users[feature_schema.query_id_column].values
            if user_features is None:
                msg = "Can not make predict in the Lin UCB method"
                raise ValueError(msg)
            usrs_feat = (
                user_features.query(f"{feature_schema.query_id_column} in @idxs_list")
                .drop(columns=[feature_schema.query_id_column])
                .to_numpy()
            )
            rel_matrix = np.zeros((num_user_pred, self._num_items), dtype=float)
            # fill in relevance matrix
            for i in range(self._num_items):
                rel_matrix[:, i] = (
                    self.eps
                    * np.sqrt((usrs_feat.dot(self.linucb_arms[i].A_inv) * usrs_feat).sum(axis=1))
                    + usrs_feat @ self.linucb_arms[i].theta
                )
            # select top k predictions from each row (unsorted ones)
            big_k = min(oversample * k, dataset.item_features.count())
            topk_indices = np.argpartition(rel_matrix, -big_k, axis=1)[:, -big_k:]
            rows_inds, _ = np.indices((num_user_pred, big_k))
            # result df
            predict_inds = np.repeat(idxs_list, big_k)
            predict_items = topk_indices.ravel()
            predict_rels = rel_matrix[rows_inds, topk_indices].ravel()
            # return everything in a PySpark template
            res_df = pd.DataFrame(
                {
                    feature_schema.query_id_column: predict_inds,
                    feature_schema.item_id_column: predict_items,
                    feature_schema.interactions_rating_column: predict_rels,
                }
            )
            return convert2spark(res_df)

        if self.regr_type == "hybrid":
            feature_schema = dataset.feature_schema
            num_user_pred = users.count()  # assuming it is a pyspark dataset
            users = users.toPandas()
            items = items.toPandas()
            user_features = dataset.query_features.toPandas()
            item_features = dataset.item_features.toPandas()
            usr_idxs_list = users[feature_schema.query_id_column].values

            if user_features is None:
                msg = "Can not make predict in the Lin UCB method"
                raise ValueError(msg)
            usrs_feat = scs.csr_matrix(
                user_features.query(
                    f"{feature_schema.query_id_column} in @items[feature_schema.item_id_column].values"
                )
                .drop(columns=[feature_schema.query_id_column])
                .to_numpy()
            )

            if item_features is None:
                msg = "Can not make predict in the Lin UCB method"
                raise ValueError(msg)
            itm_feat = scs.csr_matrix(
                item_features.query(
                    f"{feature_schema.item_id_column} in @itm_idxs_list"
                )
                .drop(columns=[feature_schema.item_id_column])
                .to_numpy()
            )
            rel_matrix = np.zeros((num_user_pred, self._num_items), dtype=float)

            # fill in relevance matrix
            for i in tqdm(range(self._num_items)):
                z = scs.kron(usrs_feat, itm_feat[i])
                rel_matrix[:, i] = usrs_feat.dot(self.linucb_arms[i].theta)
                rel_matrix[:, i] += z.dot(self.beta)

                s = (usrs_feat.dot(self.linucb_arms[i].A_inv).multiply(usrs_feat)).sum(axis=1)
                s += (z.dot(self.A_0_inv).multiply(z)).sum(axis=1)
                M = self.A_0_inv @ self.linucb_arms[i].B.T @ self.linucb_arms[i].A_inv  # noqa: N806
                s -= 2 * (z.dot(M).multiply(usrs_feat)).sum(axis=1)
                s += (usrs_feat.dot(M.T @ self.linucb_arms[i].B.T @ self.linucb_arms[i].A_inv).multiply(usrs_feat)).sum(
                    axis=1
                )

                rel_matrix[:, i] += np.array(self.eps * np.sqrt(s))[:, 0]

            # select top k predictions from each row (unsorted ones)
            big_k = min(oversample * k, dataset.item_features.count())
            topk_indices = np.argpartition(rel_matrix, -big_k, axis=1)[:, -big_k:]
            rows_inds, _ = np.indices((num_user_pred, big_k))
            # result df
            predict_inds = np.repeat(usr_idxs_list, big_k)
            predict_items = topk_indices.ravel()
            predict_rels = rel_matrix[rows_inds, topk_indices].ravel()
            # return everything in a PySpark template
            res_df = pd.DataFrame(
                {
                    feature_schema.query_id_column: predict_inds,
                    feature_schema.item_id_column: predict_items,
                    feature_schema.interactions_rating_column: predict_rels,
                }
            )
            return convert2spark(res_df)
        else:
            msg = "model.regr_type must be in ['disjoint', 'hybrid']"
            raise ValueError(msg)
