import warnings
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as scs
from tqdm import tqdm

from replay.data.dataset import Dataset
from replay.utils import SparkDataFrame
from replay.utils.spark_utils import convert2spark

from .base_rec import HybridRecommender


class DisjointArm:
    """
    Object for interactions with a single arm in a disjoint LinUCB framework.

    In disjoint LinUCB features of all arms are disjoint (no common features used).
    """

    def __init__(self, arm_index, d, eps, alpha):
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
        Function to update featurs or each Lin-UCB hand in the current model.

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


class HybridArm:
    """
    Object for interactions with a single arm in a hybrid LinUCB framework.

    Hybrid LinUCB combines shared and arm-specific features.
    Preferrable when there are meaningful relationships between different arms e.g. genres, product categories, etc.
    """

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
        Function to update featurs or each Lin-UCB hand in the current model.

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
        delta_A_0 = np.dot(usr_itm_features.T, usr_itm_features) - self.B.T @ self.A_inv @ self.B  # noqa: N806
        delta_b_0 = (usr_itm_features.T).dot(relevances) - (self.B.T).dot(self.A_inv.dot(self.b))
        return delta_A_0, delta_b_0


class LinUCB(HybridRecommender):
    """
    A recommender algorithm for contextual bandit problems.

    Implicitly proposed by `Li et al <https://arxiv.org/pdf/1003.0146>`_.
    The model assumes a linear relationship between user context, item features and action rewards,
    making it efficient for high-dimensional contexts.

    Note:
        It's recommended to scale features to a similar range (e.g., using StandardScaler or MinMaxScaler)
        to ensure proper convergence and prevent numerical instability (since relationships to learn are linear).

    >>> import pandas as pd
    >>> from replay.data.dataset import (
    ...     Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureSource, FeatureType
    ... )
    >>> interactions = pd.DataFrame({"user_id": [0, 1, 2, 2], "item_id": [0, 1, 0, 1], "rating": [1, 0, 0, 0]})
    >>> user_features = pd.DataFrame(
    ...     {"user_id": [0, 1, 2], "usr_feat_1": [1, 2, 3], "usr_feat_2": [4, 5, 6], "usr_feat_3": [7, 8, 9]}
    ... )
    >>> item_features = pd.DataFrame(
    ...     {
    ...         "item_id": [0, 1, 2, 3, 4, 5],
    ...         "itm_feat_1": [1, 2, 3, 4, 5, 6],
    ...         "itm_feat_2": [7, 8, 9, 10, 11, 12],
    ...         "itm_feat_3": [13, 14, 15, 16, 17, 18]
    ...     }
    ... )
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
    ...        *[
    ...            FeatureInfo(
    ...                column=name, feature_type=FeatureType.NUMERICAL, feature_source=FeatureSource.ITEM_FEATURES,
    ...            )
    ...            for name in ["itm_feat_1", "itm_feat_2", "itm_feat_3"]
    ...        ],
    ...        *[
    ...            FeatureInfo(
    ...                column=name, feature_type=FeatureType.NUMERICAL, feature_source=FeatureSource.QUERY_FEATURES
    ...            )
    ...            for name in ["usr_feat_1", "usr_feat_2", "usr_feat_3"]
    ...        ],
    ...    ]
    ... )
    >>> dataset = Dataset(
    ...     feature_schema=feature_schema,
    ...     interactions=interactions,
    ...     item_features=item_features,
    ...     query_features=user_features,
    ...     categorical_encoded=True,
    ... )
    >>> dataset.to_spark()
    >>> model = LinUCB(eps=-10.0, alpha=1.0, is_hybrid=False)
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

    _search_space = {
        "eps": {"type": "uniform", "args": [-10.0, 10.0]},
        "alpha": {"type": "uniform", "args": [0.001, 10.0]},
    }
    _study = None  # field required for proper optuna's optimization
    linucb_arms: List[Union[DisjointArm, HybridArm]]  # initialize only when working within fit method
    rel_matrix: np.array  # matrix with relevance scores from predict method

    def __init__(
        self,
        eps: float,
        alpha: float = 1.0,
        is_hybrid: bool = False,
    ):
        """
        :param eps: exploration coefficient
        :param alpha: ridge parameter
        :param is_hybrid: flag to choose model type. If True, model is hybrid.
        """
        self.is_hybrid = is_hybrid
        self.eps = eps
        self.alpha = alpha

    @property
    def _init_args(self):
        return {"is_hybrid": self.is_hybrid}

    def _verify_features(self, dataset: Dataset):
        if dataset.query_features is None:
            msg = "User features are missing"
            raise ValueError(msg)
        if dataset.item_features is None:
            msg = "Item features are missing"
            raise ValueError(msg)
        if (
            len(dataset.feature_schema.query_features.categorical_features) > 0
            or len(dataset.feature_schema.item_features.categorical_features) > 0
        ):
            msg = "Categorical features are not supported"
            raise ValueError(msg)

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        self._verify_features(dataset)

        if not dataset.is_pandas:
            warn_msg = "Dataset will be converted to pandas during internal calculations in fit"
            warnings.warn(warn_msg)
            dataset.to_pandas()

        feature_schema = dataset.feature_schema
        log = dataset.interactions
        user_features = dataset.query_features
        item_features = dataset.item_features

        self._num_items = item_features.shape[0]
        self._user_dim_size = user_features.shape[1] - 1
        self._item_dim_size = item_features.shape[1] - 1

        # now initialize an arm object for each potential arm instance
        if self.is_hybrid:
            hybrid_features_k = self._user_dim_size * self._item_dim_size
            self.A_0 = scs.csr_matrix(np.identity(hybrid_features_k))
            self.b_0 = np.zeros(hybrid_features_k, dtype=float)
            self.linucb_arms = [
                HybridArm(
                    arm_index=i,
                    d=self._user_dim_size,
                    k=hybrid_features_k,
                    eps=self.eps,
                    alpha=self.alpha,
                )
                for i in range(self._num_items)
            ]

            for i in tqdm(range(self._num_items)):
                B = log.loc[log[feature_schema.item_id_column] == i]  # noqa: N806
                idxs_list = B[feature_schema.query_id_column].values
                rel_list = B[feature_schema.interactions_rating_column].values
                if not B.empty:
                    # if we have at least one user interacting with the hand i
                    cur_usrs = scs.csr_matrix(
                        user_features.query(f"{feature_schema.query_id_column} in @idxs_list")
                        .drop(columns=[feature_schema.query_id_column])
                        .to_numpy()
                    )
                    cur_itm = scs.csr_matrix(
                        item_features.iloc[i].drop(labels=[feature_schema.item_id_column]).to_numpy()
                    )
                    usr_itm_features = scs.kron(cur_usrs, cur_itm)
                    delta_A_0, delta_b_0 = self.linucb_arms[i].feature_update(  # noqa: N806
                        cur_usrs, usr_itm_features, rel_list
                    )

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
            self.linucb_arms = [
                DisjointArm(arm_index=i, d=self._user_dim_size, eps=self.eps, alpha=self.alpha)
                for i in range(self._num_items)
            ]

            for i in range(self._num_items):
                B = log.loc[log[feature_schema.item_id_column] == i]  # noqa: N806
                idxs_list = B[feature_schema.query_id_column].values  # noqa: F841
                rel_list = B[feature_schema.interactions_rating_column].values
                if not B.empty:
                    # if we have at least one user interacting with the hand i
                    cur_usrs = user_features.query(f"{feature_schema.query_id_column} in @idxs_list").drop(
                        columns=[feature_schema.query_id_column]
                    )
                    self.linucb_arms[i].feature_update(cur_usrs.to_numpy(), rel_list)

        warn_msg = "Dataset will be converted to spark after internal calculations in fit"
        warnings.warn(warn_msg)
        dataset.to_spark()

    def _predict(
        self,
        dataset: Dataset,
        k: int,
        users: SparkDataFrame,
        items: SparkDataFrame = None,
        filter_seen_items: bool = True,  # noqa: ARG002
        oversample: int = 20,
    ) -> SparkDataFrame:
        self._verify_features(dataset)

        if not dataset.is_pandas:
            warn_msg = "Dataset will be converted to pandas during internal calculations in predict"
            warnings.warn(warn_msg)
            dataset.to_pandas()

        feature_schema = dataset.feature_schema
        user_features = dataset.query_features
        item_features = dataset.item_features
        big_k = min(oversample * k, item_features.shape[0])

        users = users.toPandas()
        num_user_pred = users.shape[0]
        rel_matrix = np.zeros((num_user_pred, self._num_items), dtype=float)

        if self.is_hybrid:
            items = items.toPandas()
            usr_idxs_list = users[feature_schema.query_id_column].values
            itm_idxs_list = items[feature_schema.item_id_column].values  # noqa: F841

            usrs_feat = scs.csr_matrix(
                user_features.query(f"{feature_schema.query_id_column} in @usr_idxs_list")
                .drop(columns=[feature_schema.query_id_column])
                .to_numpy()
            )
            itm_feat = scs.csr_matrix(
                item_features.query(f"{feature_schema.item_id_column} in @itm_idxs_list")
                .drop(columns=[feature_schema.item_id_column])
                .to_numpy()
            )

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

        else:
            idxs_list = users[feature_schema.query_id_column].values
            usrs_feat = (
                user_features.query(f"{feature_schema.query_id_column} in @idxs_list")
                .drop(columns=[feature_schema.query_id_column])
                .to_numpy()
            )
            # fill in relevance matrix
            for i in range(self._num_items):
                rel_matrix[:, i] = (
                    self.eps * np.sqrt((usrs_feat.dot(self.linucb_arms[i].A_inv) * usrs_feat).sum(axis=1))
                    + usrs_feat @ self.linucb_arms[i].theta
                )
            # select top k predictions from each row (unsorted ones)
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

        warn_msg = "Dataset will be converted to spark after internal calculations in predict"
        warnings.warn(warn_msg)
        dataset.to_spark()
        return convert2spark(res_df)
