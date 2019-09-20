from abc import ABC, abstractmethod
from typing import Iterable, Dict, Set

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
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

    @abstractmethod
    def get_params(self) -> Dict[str, object]:
        """

        :return:
        """

    @staticmethod
    def _check_dataframe(df: DataFrame or None,
                         required_columns: Set[str],
                         optional_columns: Set[str]):
        if df is None:
            raise ValueError("Датафрейм есть None")

        # чекаем, что датафрейм не пустой
        if len(df.head(1)) == 0:
            raise ValueError("Датафрейм пустой")

        df_columns = df.columns

        # чекаем на нуллы
        for column in df_columns:
            if df.where(sf.col(column).isNull()).count() > 0:
                raise ValueError(f"В колонке '{column}' есть значения NULL")

        # чекаем колонки
        if not required_columns.issubset(df_columns):
            raise ValueError(
                f"В датафрейме нет обязательных колонок ({required_columns})")

        wrong_columns = set(df_columns) \
            .difference(required_columns) \
            .difference(optional_columns)
        if len(wrong_columns) > 0:
            raise ValueError(
                f"В датафрейме есть лишние колонки: {wrong_columns}")

    @staticmethod
    def _check_feature_dataframe(features: DataFrame or None,
                                 required_columns: Set[str],
                                 optional_columns: Set[str]):
        if features is None:
            return

        columns = set(features.columns).difference({'user_id', 'timestamp'})
        if len(columns) == 0:
            raise ValueError("В датафрейме user_features нет колонок с фичами")
        BaseRecommender._check_dataframe(
            features,
            required_columns=required_columns.union(columns),
            optional_columns=optional_columns
        )

    def fit(self, log: DataFrame,
            user_features: DataFrame or None,
            item_features: DataFrame or None) -> None:
        """

        :param log:
        :param user_features:
        :param item_features:
        :return:
        """
        self._check_dataframe(log,
                              required_columns={'item_id', 'user_id'},
                              optional_columns={'timestamp', 'relevance',
                                                'context'})
        self._check_feature_dataframe(user_features,
                                      required_columns={'user_id'},
                                      optional_columns={'timestamp'})
        self._check_feature_dataframe(item_features,
                                      required_columns={'item_id'},
                                      optional_columns={'timestamp'})

        self._fit(log, user_features, item_features)

    @abstractmethod
    def _fit(self, log: DataFrame,
             user_features: DataFrame or None,
             item_features: DataFrame or None) -> None:
        """

        :param log:
        :param user_features:
        :param item_features:
        :return:
        """

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
        self._check_dataframe(log,
                              required_columns={'item_id', 'user_id'},
                              optional_columns={'timestamp', 'relevance',
                                                'context'}
                              )
        self._check_feature_dataframe(user_features,
                                      required_columns={'user_id'},
                                      optional_columns={'timestamp'})
        self._check_feature_dataframe(item_features,
                                      required_columns={'item_id'},
                                      optional_columns={'timestamp'})

        if users is None:
            users = log.select('user_id').distinct()
        else:
            users = self.spark.createDataFrame(data=[[user] for user in users],
                                               schema=['user_id'])
        if items is None:
            items = log.select('item_id').distinct()['item_id']
        else:
            items = set(items)

        return self._predict(
            k, users, items,
            context, log,
            user_features, item_features,
            to_filter_seen_items
        )

    @abstractmethod
    def _predict(self,
                 k: int,
                 users: Iterable or DataFrame,
                 items: Iterable or DataFrame,
                 context: str,
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

    def _filter_seen_recs(self, recs: DataFrame, log: DataFrame) -> DataFrame:
        """

        :param recs:
        :param log:
        :return:
        """
        raise NotImplementedError("Method is not implemented!")

    def _leave_top_recs(self, k: int, recs: DataFrame) -> DataFrame:
        """

        :param k:
        :param recs:
        :return:
        """
        raise NotImplementedError("Method is not implemented!")

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
        raise NotImplementedError("Method is not implemented!")

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
        raise NotImplementedError("Method is not implemented!")
