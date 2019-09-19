from abc import ABC, abstractmethod
from typing import Iterable, Dict

from pyspark.sql import DataFrame, SparkSession
from sklearn.preprocessing import LabelEncoder


class BaseRecommender(ABC):
    def __init__(self, spark: SparkSession, **kwargs):
        self.model = None
        self.encoder = LabelEncoder()
        self.spark = spark

    def set_params(self, **params):
        """

        :param params:
        :return:
        """
        if not params:
            return self
        valid_params = self.get_params()

        for param, value in params.items():
            if param not in valid_params:
                raise ValueError(
                    "Неправильный параметр для данного рекоммендера "
                    f"{param}. "
                    "Проверьте список параметров, "
                    f"которые принимает рекоммендер: {valid_params}")

            setattr(self, param, value)
        return self

    @abstractmethod
    def get_params(self) -> Dict[str, object]:
        """

        :return:
        """

    @abstractmethod
    def fit(self, log: DataFrame,
            user_features: DataFrame or None,
            item_features: DataFrame or None) -> None:
        """
        
        :param log: 
        :param user_features: 
        :param item_features: 
        :return: 
        """

    @abstractmethod
    def predict(self,
                k: int,
                users: Iterable or None,
                items: Iterable or None,
                context: str or None,
                log: DataFrame,
                user_features: DataFrame or None,
                item_features: DataFrame or None,
                to_filter_seen_items: bool = True) -> DataFrame:
        """
        
        :param k: 
        :param users: 
        :param items: 
        :param context: 
        :param log: 
        :param user_features: 
        :param item_features: 
        :param to_filter_seen_items: 
        :return: 
        """

    def fit_predict(self,
                    k: int,
                    users: Iterable or None,
                    items: Iterable or None,
                    context: str or None,
                    log: DataFrame,
                    user_features: DataFrame or None,
                    item_features: DataFrame or None,
                    to_filter_seen_items: bool = True) -> DataFrame:
        """

        :param k:
        :param users:
        :param items:
        :param context:
        :param log:
        :param user_features:
        :param item_features:
        :param to_filter_seen_items:
        :return:
        """
        self.fit(log, user_features, item_features)
        return self.predict(k, users, items,
                            context, log,
                            user_features, item_features,
                            to_filter_seen_items)

    @abstractmethod
    def _filter_seen_recs(self, recs: DataFrame, log: DataFrame) -> DataFrame:
        """

        :param recs:
        :param log:
        :return:
        """

    @abstractmethod
    def _leave_top_recs(self, k: int, recs: DataFrame) -> DataFrame:
        """

        :param k:
        :param recs:
        :return:
        """

    @abstractmethod
    def _get_batch_recs(self, users: Iterable,
                        items: Iterable,
                        context: str or None,
                        log: DataFrame,
                        user_features: DataFrame or None,
                        item_features: DataFrame or None,
                        to_filter_seen_items: bool = True) -> DataFrame:
        """

        :param users:
        :param items:
        :param context:
        :param log:
        :param user_features:
        :param item_features:
        :return:
        """

    @abstractmethod
    def _get_single_recs(self,
                         user: str,
                         items: Iterable,
                         context: str or None,
                         log: DataFrame,
                         user_feature: DataFrame or None,
                         item_features: DataFrame or None,
                         to_filter_seen_items: bool = True
                         ) -> DataFrame:
        """

        :param user:
        :param items:
        :param context:
        :param log:
        :param user_feature:
        :param item_features:
        :param to_filter_seen_items:
        :return:
        """
