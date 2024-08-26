import math
import numpy as np
import pandas as pd
import os

from typing import Any, Dict, List, Optional

from replay.data.dataset import Dataset
from replay.metrics import NDCG, Metric
from replay.utils import PYSPARK_AVAILABLE, SparkDataFrame

from .base_rec import HybridRecommender

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf

from replay.utils.spark_utils import convert2spark

#import for parameter optimization
from optuna import create_study
from optuna.samplers import TPESampler


#Object for interactions with a single arm in a UCB disjoint framework
class linucb_disjoint_arm():
    def __init__(self, arm_index, d, eps, alpha): #in case of lin ucb with disjoint features: d = dimension of user's features solely
        # Track arm index
        self.arm_index = arm_index
        # Exploration parameter
        self.eps = eps
        self.alpha = alpha
        # Inverse of feature matrix for ridge regression
        self.A = self.alpha*np.identity(d)
        self.A_inv = (1./self.alpha)*np.identity(d)
        # right-hand side of the regression
        self.theta = np.zeros(d, dtype = float)
        self.cond_number = 1.0
    
    def feature_update(self, usr_features, relevances):
        """
        function to update featurs or each Lin-UCB hand in the current model
        features:
            usr_features = matrix (np.array of shape (m,d)), where m = number of occurences of the current feature in the dataset;
            usr_features[i] = features of i-th user, who rated this particular arm (movie);
            relevances = np.array(d) - rating of i-th user, who rated this particular arm (movie);
        """
        # Update A which is (d * d) matrix.
        self.A += np.dot(usr_features.T, usr_features)
        self.A_inv = np.linalg.inv(self.A)
        # Update the parameter theta by the results  linear regression
        self.theta = np.linalg.lstsq(self.A, usr_features.T @ relevances, rcond = 1.0)[0]
        self.cond_number = np.linalg.cond(self.A) #this ome needed for deug only


class LinUCB(HybridRecommender):
    """
    Function implementing the functional of linear UCB 
    """

    def __init__(
        self,
        eps: float, #exploration parameter
        alpha: float, #ridge parameter
        regr_type: str, #put here "disjoint" or "hybrid"
        random_state: Optional[int] = None,
    ):  # pylint: disable=too-many-arguments
        np.random.seed(42)
        self.regr_type = regr_type
        self.random_state = random_state
        self.eps = eps
        self.alpha = alpha
        self.linucb_arms = None #initialize only when working within fit method
        cpu_count = os.cpu_count()
        self.num_threads = cpu_count if cpu_count is not None else 1


        self._study = None #field required for proper optuna's optimization
        self._search_space = {
        "eps": {"type": "uniform", "args": [-10.0, 10.0]},
        "alpha": {"type": "uniform", "args": [0.001, 10.0]},
        }
        #self._objective = MainObjective


    @property
    def _init_args(self):
        return {
            "regression type": self.regr_type,
            "seed": self.random_state,
        }

    # pylint: disable=too-many-arguments
    def optimize(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        param_borders: Optional[Dict[str, List[Any]]] = None,
        criterion: Metric = NDCG,
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Searches best parameters with optuna.

        :param train: train data
        :param test: test data
        :param user_features: user features
        :param item_features: item features
        :param param_borders: a dictionary with search borders, where
            key is the parameter name and value is the range of possible values
            ``{param: [low, high]}``. In case of categorical parameters it is
            all possible values: ``{cat_param: [cat_1, cat_2, cat_3]}``.
        :param criterion: metric to use for optimization
        :param k: recommendation list length
        :param budget: number of points to try
        :param new_study: keep searching with previous study or start a new study
        :return: dictionary with best parameters
        """

        # self.logger.warning(
        #     "The UCB model has only exploration coefficient parameter, which cannot not be directly optimized"
        # )


        self.query_column = train_dataset.feature_schema.query_id_column
        self.item_column = train_dataset.feature_schema.item_id_column
        self.rating_column = train_dataset.feature_schema.interactions_rating_column
        self.timestamp_column = train_dataset.feature_schema.interactions_timestamp_column

        self.criterion = criterion(
            topk=k,
            query_column=self.query_column,
            item_column=self.item_column,
            rating_column=self.rating_column,
        )

        if self._search_space is None:
            self.logger.warning("%s has no hyper parameters to optimize", str(self))
            return None

        if self.study is None or new_study:
            self.study = create_study(direction="maximize", sampler=TPESampler())

        search_space = self._prepare_param_borders(param_borders)
        if self._init_params_in_search_space(search_space) and not self._params_tried():
            self.study.enqueue_trial(self._init_args)

        split_data = self._prepare_split_data(train_dataset, test_dataset)
        objective = self._objective(
            search_space=search_space,
            split_data=split_data,
            recommender=self,
            criterion=self.criterion,
            k=k,
        )

        self.study.optimize(objective, budget)
        best_params = self.study.best_params
        self.set_params(**best_params)
        return best_params
    

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        feature_schema = dataset.feature_schema
        #should not work if user features or item features are unavailable 
        if dataset.query_features is None:
            raise ValueError("User features are missing for fitting")
        if dataset.item_features is None:
            raise ValueError("Item features are missing for fitting")
        #assuming that user_features and item_features are both dataframes
        #now forget about pyspark until the better times
        log = dataset.interactions.toPandas()
        user_features = dataset.query_features.toPandas()
        item_features = dataset.item_features.toPandas()
        #check that the dataframe contains uer indexes
        if not feature_schema.query_id_column in user_features.columns:
            raise ValueError("User indices are missing in user features dataframe")
        self._num_items = item_features.shape[0]
        self._user_dim_size = user_features.shape[1] - 1
        #now initialize an arm object for each potential arm instance
        self.linucb_arms = [linucb_disjoint_arm(arm_index = i, d = self._user_dim_size, eps = self.eps, alpha = self.alpha) for i in range(self._num_items)]
        #now we work with pandas
        for i in range(self._num_items):
            B = log.loc[log[feature_schema.item_id_column] == i]
            idxs_list = B[feature_schema.query_id_column].values
            rel_list = B[feature_schema.interactions_rating_column].values
            if not B.empty:
                #if we have at least one user interacting with the hand i
                cur_usrs = user_features.query(f"{feature_schema.query_id_column} in @idxs_list").drop(columns=[feature_schema.query_id_column])
                self.linucb_arms[i].feature_update(cur_usrs.to_numpy(), rel_list)
        condit_number = [self.linucb_arms[i].cond_number for i in range(self._num_items)]
        #finished
        return
        
    def _predict(
        self,   
        dataset: Dataset,
        k: int,
        users: SparkDataFrame,
        items: SparkDataFrame = None,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        feature_schema = dataset.feature_schema
        num_user_pred = users.count() #assuming it is a pyspark dataset
        users = users.toPandas()
        user_features = dataset.query_features.toPandas()
        item_features = dataset.item_features.toPandas()
        idxs_list = users[feature_schema.query_id_column].values
        if user_features is None:
            raise ValueError("Can not make predict in the Lin UCB method")
        usrs_feat = user_features.query(f"{feature_schema.query_id_column} in @idxs_list").drop(columns=[feature_schema.query_id_column]).to_numpy()
        rel_matrix = np.zeros((num_user_pred,self._num_items),dtype = float)
        #fill in relevance matrix
        for i in range(self._num_items):
            rel_matrix[:,i] = self.eps * np.sqrt((usrs_feat.dot(self.linucb_arms[i].A_inv)*usrs_feat).sum(axis=1)) + usrs_feat @ self.linucb_arms[i].theta  
        #select top k predictions from each row (unsorted ones)
        big_k = 20*k
        topk_indices = np.argpartition(rel_matrix, -big_k, axis=1)[:, -big_k:]
        rows_inds,_ = np.indices((num_user_pred, big_k))
        #result df
        predict_inds = np.repeat(idxs_list, big_k)
        predict_items = topk_indices.ravel()
        predict_rels = rel_matrix[rows_inds,topk_indices].ravel()
        #return everything in a PySpark template
        res_df = pd.DataFrame({feature_schema.query_id_column: predict_inds, feature_schema.item_id_column: predict_items, feature_schema.interactions_rating_column: predict_rels})
        return convert2spark(res_df)