from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from replay.experimental.models.base_rec import HybridRecommender
from replay.experimental.models.u_lin_ucb import ULinUCB
from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame, SparkDataFrame

if PYSPARK_AVAILABLE:
    from replay.utils.spark_utils import convert2spark


class HierarchicalRecommender(HybridRecommender):
    """
    Hierarchical Recommender class is inspired by
    `the article of Song et al <https://arxiv.org/abs/2110.09905>`_ and is a
    generalization of the method. By default it works as HCB proposed there.

    The model sequentially clusterizes the item space constructing a tree of
    given ``depth``. The clusterization is performed according to the
    ``cluster_model`` - any sklearn clusterer instance provided by the user.

    At each node of a tree a node recommender instance is mounted. All of them
    are produced by ``recommender_class`` object (not an instance!) and are
    initialized with ``recommender_params``.

    To predict an item the model goes down the tree each time selecting the
    next node as the one predicted by the parent node recommender. A leaf
    node recommender would give an item itself.

    The log is considered as the history of user-item interactions. To fit
    the model each interaction is counted in all node recommenders on the
    path from the root to the item as if such path would be traversed through
    the prediction process.

    Hierarchical Recommender may be useful to enhance the perforamance of
    simple models not suitable for large item space problems (such as many
    contextual bandits) and to reduce prediction time in models that need to
    iterate through all of the items to make a recommendation.

    In this version Hierarchical Recommender is implemented as
    ``HybridRecommender`` and apart from ``log`` requires both ``item_features``
    and ``user_features`` in ``fit()`` method. By the same reason only
    ``HybridRecommender`` classess may be passed as a ``recommender_class``.
    Need in features at ``predict()`` depends on the ``recommender_class``
    itself.

    Note that current implementation relies mostly on python rather than
    pyspark.
    """

    def __init__(
        self,
        depth,
        cluster_model,
        recommender_class=ULinUCB,
        recommender_params={},
    ):
        """
        :param depth: depth of the item tree
        :param cluster_model: an sklearn.cluster object (or any with similar
            API) that would perform clustering on the item space
        :param recommender_class: a RePlay hybrid recommender class object (not an
            instance!) instances of which would be mounted at each tree node
        :param recommender_params: initialization parameters for the recommenders
        """

        self.depth = depth
        self.cluster_model = cluster_model
        self.recommender_class = recommender_class
        self.recommender_params = recommender_params
        self.root = Node(parent=None, tree=self)

    @property
    def _init_args(self):
        return {
            "depth": self.depth,
            "cluster_model": self.cluster_model,
            "recommender_class": self.recommender_class,
            "recommender_params": self.recommender_params,
        }

    def _fit(
        self,
        log: SparkDataFrame,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
    ) -> None:
        self.logger.debug("Clustering...")
        self.root._procreate(item_features.toPandas())

        self.logger.debug("Fitting...")
        self.root._fit(log.toPandas(), user_features.toPandas(), item_features.toPandas())

    def _predict(
        self,
        log: SparkDataFrame,
        k: int,
        users: SparkDataFrame,
        items: SparkDataFrame,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        self.logger.debug("Predicting...")
        pred = self.root._predict(
            log.toPandas(),
            k,
            users.toPandas(),
            items.toPandas(),
            user_features,
            item_features,
            filter_seen_items,
        )
        return convert2spark(pred)

    def _get_recommender(self):
        new_recommender = self.recommender_class(**self.recommender_params)
        assert isinstance(new_recommender, HybridRecommender)
        return new_recommender

    def _get_clusterer(self, node):
        if node.is_leaf:
            return Clusterer(model=DiscreteClusterer())
        else:
            return Clusterer(model=self.cluster_model)


class Node:
    """
    Node of a Hierarchichal Recommender. The Node receives a clusterer and a
    recommender from the tree and interacts with them at clustering, fitting
    and predicting stages.
    """

    def __init__(self, parent, tree: HierarchicalRecommender = None):
        """
        :param parent: the parent node
        :param tree: the tree which the node belongs to (is None by default
        and is inherited from the parent)
        """
        self.parent = parent
        self.tree = tree
        self.is_leaf = False

        if parent is None:
            self.level = 0
            assert tree is not None
        else:
            self.tree = self.parent.tree
            self.level = self.parent.level + 1

        if self.level == (self.tree.depth - 1):
            self.is_leaf = True
            self.children = None

        self.clusterer = self.tree._get_clusterer(self)
        self.recommender = self.tree._get_recommender()

    def get_num_children(self):
        return len(self.children)

    def _procreate(
        self,
        items: PandasDataFrame,
    ) -> None:
        items["cluster_idx"] = self.clusterer.fit_predict(items)

        if not self.is_leaf:
            self.children = [None] * self.clusterer.get_num_clusters()
            for cl_idx, cl_items in items.groupby("cluster_idx"):
                self.children[cl_idx] = Node(parent=self)
                self.children[cl_idx]._procreate(cl_items)

    def _fit(
        self,
        log: PandasDataFrame,
        user_features: PandasDataFrame,
        item_features: PandasDataFrame,
    ) -> None:
        log["cluster_idx"] = self.clusterer.predict(log[["item_idx"]])

        if not self.is_leaf:
            for cl_idx, cl_log in tqdm(log.groupby("cluster_idx")):
                self.children[cl_idx]._fit(cl_log, user_features, item_features)

        rec_params = {
            "log": convert2spark(log.drop(columns="item_idx").rename(columns={"cluster_idx": "item_idx"})),
            "user_features": convert2spark(user_features),
            "item_features": convert2spark(self.clusterer.get_cluster_centers()),
        }
        self.recommender.fit(**rec_params)

    def _predict(
        self,
        log: PandasDataFrame,
        k: int,
        users: PandasDataFrame,
        items: PandasDataFrame,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
        filter_seen_items: bool = True,
    ) -> PandasDataFrame:
        pred = pd.DataFrame(columns=["user_idx", "item_idx", "relevance"])
        log["cluster_idx"] = self.clusterer.predict(log[["item_idx"]])
        items["cluster_idx"] = self.clusterer.predict(items[["item_idx"]])

        rec_params = {
            "log": convert2spark(log.drop(columns="item_idx").rename(columns={"cluster_idx": "item_idx"})),
            "users": convert2spark(users),
            "items": convert2spark(items.drop(columns="item_idx").rename(columns={"cluster_idx": "item_idx"})),
            "user_features": user_features,
            "item_features": item_features,
        }

        if self.is_leaf:
            rec_params["k"] = k
            rec_params["filter_seen_items"] = filter_seen_items
            pred = self.recommender.predict(**rec_params).toPandas().rename(columns={"item_idx": "cluster_idx"})
            pred["item_idx"] = self.clusterer.predict_items(pred)
            pred = pred.drop(columns=["cluster_idx"])
        else:
            rec_params["k"] = 1
            rec_params["filter_seen_items"] = False
            pred_clusters = (
                self.recommender.predict(**rec_params).toPandas().rename(columns={"item_idx": "cluster_idx"})
            )

            for cl_idx, cluster in pred_clusters.groupby("cluster_idx"):
                child_params = {
                    "log": log[log["cluster_idx"] == cl_idx].drop(columns="cluster_idx"),
                    "k": k,
                    "users": cluster[["user_idx"]],
                    "items": items[items["cluster_idx"] == cl_idx].drop(columns="cluster_idx"),
                    "user_features": user_features,
                    "item_features": item_features,
                    "filter_seen_items": filter_seen_items,
                }
                cl_pred = self.children[cl_idx]._predict(**child_params)
                pred = pd.concat([pred, cl_pred])

        return pred


class Clusterer:
    """
    Wrapper class to provide proper and unified interaction with sklearn
    clusterers.
    """

    def __init__(self, model):
        """
        :param model: sklearn.cluster object or one with similar API
        """
        self._model = model

    def fit_predict(
        self,
        items: PandasDataFrame,
    ):
        self.fit(items)
        return self.predict(items)

    def fit(
        self,
        items: PandasDataFrame,
    ) -> None:
        items = items.sort_values(
            by="item_idx"
        )  # for discrete clusterer to work right, otherwise items would be shuffled

        item_idx = items["item_idx"].to_numpy()
        item_features = items.drop(columns="item_idx").to_numpy()

        self._labels = self._model.fit_predict(item_features)

        self._cluster_map = dict(zip(item_idx, self._labels))
        self._item_map = dict(zip(self._labels, item_idx))

        self._set_cluster_centers(items)

    def predict(
        self,
        items: PandasDataFrame,
    ):
        return items["item_idx"].map(self.get_cluster_map())

    def predict_items(
        self,
        clusters: PandasDataFrame,
    ):
        return clusters["cluster_idx"].map(self.get_item_map())

    def _set_cluster_centers(
        self,
        items: PandasDataFrame,
    ) -> None:
        items["cluster_idx"] = self.predict(items)
        self._cluster_centers = (
            items.drop(columns="item_idx")
            .groupby("cluster_idx")
            .mean()
            .reset_index()
            .rename(columns={"cluster_idx": "item_idx"})
        )

        self._num_clusters = self._cluster_centers.shape[0]

    def get_cluster_map(self) -> dict:
        return self._cluster_map

    def get_item_map(self) -> dict:
        return self._item_map

    def get_cluster_centers(self) -> PandasDataFrame:
        return self._cluster_centers

    def get_num_clusters(self) -> int:
        return self._num_clusters


class DiscreteClusterer:
    """
    Discrete Clusterer - one that counts each item as a cluster already.
    """

    def fit_predict(self, items):
        self.cluster_centers_ = items
        return np.arange(items.shape[0])
