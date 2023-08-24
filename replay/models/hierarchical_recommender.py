#TODO
# docs in .rst
# experiments (02_model_comparison)

# To the Contributor
# - make smart cluster embeddings
# - add support of more than HybridRecommenders
# - translate Pandas to PySpark
# - add predict_proba usage (when ready) and other prediction schemes (sample, and predict proba)
# - something that you come up with yourself!
#

import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Optional
from pyspark.sql import DataFrame
from pandas import DataFrame as PandasDataFrame
from replay.models.base_rec import HybridRecommender
from replay.models.u_lin_ucb import uLinUCB
from replay.utils import convert2spark


class HierarchicalRecommender(HybridRecommender):
    """
    Hierarchical Recommender class inspired by 
    `Song et al <https://arxiv.org/pdf/1102.2490.pdf>`_. The recommender 
    sequentially clusterizes the item space constructing a tree of given depth. 
    At each node of the tree a simple recommeneder operates. If not a leaf the 
    item space of a node recommender is children nodes, otherwise, some cluster 
    of the initial item space. At the fitting stage the hierarchical 
    recommender sequantially fits recommenders at nodes for them to learn which 
    child or item to recommend for each user. To make a prediction for a user 
    the recommender sieves down the tree making a path to the predicted item by 
    selecting recommended child at each node.
    """

    def __init__(
        self,
        depth,
        cluster_model,
        recommender_class = uLinUCB,
        recommender_params = {},
    ):
        """
        :param depth: depth of the item tree
        :param cluster_model: an sklearn.cluseter object (or any with similar 
            API) that would perform clustering on the item space
        :param recommender_class: a RePlay recommeder class object (not an 
            instance!) instances of which would be mounted at each tree node
        :param recommender_params: initialization parameters for the recommenders
        """

        self.depth              = depth
        self.cluster_model      = cluster_model
        self.recommender_class   = recommender_class
        self.recommender_params = recommender_params
        self.root               = Node(parent=None, tree=self)

    @property
    def _init_args(self):
        return {
            "depth":         self.depth,
            "cluster_model": self.cluster_model,
            "recommender_class":   self.recommender_class,
            "recommender_params": self.recommender_params,
        }

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:    
        
        print("Clustering...")
        self.root._procreate(item_features.toPandas())
        
        print("Fitting...")
        self.root._fit(log.toPandas(), user_features.toPandas(), item_features.toPandas())
        
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

        print("Predicting...")
        pred = self.root._predict(  log.toPandas(),
                                    k,
                                    users.toPandas(),
                                    items.toPandas(),
                                    user_features,
                                    item_features,
                                    filter_seen_items
                                  )
        print(pred)
        return convert2spark(pred)
    
    def _get_recommender(self, node):
        new_recommender = self.recommender_class(**self.recommender_params)
        assert(isinstance(new_recommender, HybridRecommender))
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

    def __init__(
            self,
            parent, 
            tree : HierarchicalRecommender = None):
        """
        :param parent: the parent node
        :param tree: the tree which the node belongs to (is None by default
        and is inherited from the parent)
        """
        self.parent   = parent
        self.tree     = tree
        self.is_leaf  = False

        if (parent is None):
            self.level = 0
            assert tree != None
        else:
            self.tree = self.parent.tree
            self.level = self.parent.level + 1
        
        if (self.level == (self.tree.depth - 1)):
            self.is_leaf = True
            self.children = None
        
        self.clusterer = self.tree._get_clusterer(self)
        self.recommender = self.tree._get_recommender(self)

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
        
        rec_params = {"log":           convert2spark(log.drop(columns="item_idx")
                                                        .rename(columns={"cluster_idx": "item_idx"})),
                      "user_features": convert2spark(user_features),
                      "item_features": convert2spark(self.clusterer.get_cluster_centers())
                      }
        self.recommender.fit(**rec_params)


    def _predict(
        self,
        log: PandasDataFrame,
        k: int,
        users: PandasDataFrame,
        items: PandasDataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> PandasDataFrame:
        
        pred = pd.DataFrame(columns=['user_idx', 'item_idx', 'relevance'])
        log  ["cluster_idx"] = self.clusterer.predict(log  [["item_idx"]])
        items["cluster_idx"] = self.clusterer.predict(items[["item_idx"]])

        rec_params = { "log":    convert2spark(log.  drop(columns="item_idx")
                                 .rename(columns={"cluster_idx" : "item_idx"})),
                        "users": convert2spark(users),
                        "items": convert2spark(items.drop(columns="item_idx")
                                 .rename(columns={"cluster_idx" : "item_idx"})),
                        "user_features": user_features,
                        "item_features": item_features,
                       }
        
        if self.is_leaf:
            rec_params["k"] = k
            rec_params["filter_seen_items"] = filter_seen_items
            pred = self.recommender.predict(**rec_params).toPandas().rename(columns={"item_idx" : "cluster_idx"})
            pred["item_idx"] = self.clusterer.predict_items(pred)
            pred = pred.drop(columns=["cluster_idx"])
        else:
            rec_params["k"] = 1
            rec_params["filter_seen_items"] = False
            pred_clusters = self.recommender.predict(**rec_params).toPandas().rename(columns={"item_idx" : "cluster_idx"})

            for cl_idx, cluster in pred_clusters.groupby("cluster_idx"):
                child_params = {"log":   log  [log  ["cluster_idx"] == cl_idx].drop(columns="cluster_idx"),
                                "k": k,
                                "users": cluster[["user_idx"]],
                                "items": items[items["cluster_idx"] == cl_idx].drop(columns="cluster_idx"),
                                "user_features": user_features,
                                "item_features": item_features,
                                "filter_seen_items": filter_seen_items
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
        
        items = items.sort_values(by="item_idx") # for discrete clusterer to work right, otherwise items would be shuffled

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
        items : PandasDataFrame,
    ) -> None:
        items["cluster_idx"] = self.predict(items)
        self._cluster_centers = items.drop(columns="item_idx"
                                    ).groupby("cluster_idx"
                                    ).mean(
                                    ).reset_index().rename(columns={"cluster_idx" : "item_idx"})
        
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