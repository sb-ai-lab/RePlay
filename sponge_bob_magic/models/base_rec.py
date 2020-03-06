"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Iterable
from uuid import uuid4

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
import pandas as pd

from sponge_bob_magic import constants
from sponge_bob_magic.session_handler import State
from sponge_bob_magic.utils import write_read_dataframe


class Recommender(ABC):
    """ Базовый класс-рекомендатель. """
    model: Any = None

    def set_params(self, **params: Dict[str, Any]) -> None:
        """
        Устанавливает параметры рекоммендера.

        :param params: словарь, ключ - название параметра,
            значение - значение параметра
        :return:
        """
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
        Возвращает параметры рекоммендера в виде словаря.

        :return: словарь параметров, ключ - название параметра,
            значение - значение параметра
        """

    @staticmethod
    def _check_dataframe(dataframe: DataFrame,
                         required_columns: Set[str],
                         optional_columns: Set[str]) -> None:
        if dataframe is None:
            raise ValueError("Датафрейм есть None")

        # чекаем, что датафрейм не пустой
        if len(dataframe.head(1)) == 0:
            raise ValueError("Датафрейм пустой")

        df_columns = dataframe.columns

        # чекаем на нуллы
        for column in df_columns:
            if dataframe.where(sf.col(column).isNull()).count() > 0:
                raise ValueError(f"В колонке '{column}' есть значения NULL")

        # чекаем колонки
        if not required_columns.issubset(df_columns):
            raise ValueError(
                f"В датафрейме нет обязательных колонок ({required_columns})")

        wrong_columns = (set(df_columns)
                         .difference(required_columns)
                         .difference(optional_columns))
        if len(wrong_columns) > 0:
            raise ValueError(
                f"В датафрейме есть лишние колонки: {wrong_columns}")

    @staticmethod
    def _check_feature_dataframe(features: DataFrame,
                                 required_columns: Set[str],
                                 optional_columns: Set[str]) -> None:
        if features is None:
            return

        columns = set(features.columns).difference(required_columns)
        if len(columns) == 0:
            raise ValueError("В датафрейме features нет колонок с фичами")

        Recommender._check_dataframe(
            features,
            required_columns=required_columns.union(columns),
            optional_columns=optional_columns
        )

    @staticmethod
    def _check_input_dataframes(log: DataFrame,
                                user_features: DataFrame,
                                item_features: DataFrame) -> None:
        Recommender._check_dataframe(
            log,
            required_columns={"item_id", "user_id", "timestamp", "relevance",
                              "context"},
            optional_columns=set()
        )
        Recommender._check_feature_dataframe(
            user_features,
            optional_columns=set(),
            required_columns={"user_id", "timestamp"}
        )
        Recommender._check_feature_dataframe(
            item_features,
            optional_columns=set(),
            required_columns={"item_id", "timestamp"}
        )

    def fit(self, log: DataFrame,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None) -> None:
        """
        Обучает модель на логе и признаках пользователей и объектов.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :return:
        """
        logging.debug("Проверка датафреймов")
        self._check_input_dataframes(log, user_features, item_features)

        logging.debug("Предварительная стадия обучения (pre-fit)")
        self._pre_fit(log, user_features, item_features)

        logging.debug("Основная стадия обучения (fit)")
        self._fit_partial(log, user_features, item_features)

    @abstractmethod
    def _pre_fit(self, log: DataFrame,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None) -> None:
        """
        Метод-helper для обучения модели, в которой параметры не используются.
        Нужен для того, чтобы вынести вычисление трудоемких агрегатов
        в отдельный метод, который по возможности будет вызываться один раз.
        Должен быть имплементирован наследниками.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :return:
        """

    @abstractmethod
    def _fit_partial(self, log: DataFrame,
                     user_features: Optional[DataFrame] = None,
                     item_features: Optional[DataFrame] = None) -> None:
        """
        Метод-helper для обучения модели, в которой используются параметры.
        Должен быть имплементирован наследниками.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :return:
        """

    def predict(self,
                log: DataFrame,
                k: int,
                users: Optional[DataFrame] = None,
                items: Optional[DataFrame] = None,
                context: Optional[str] = None,
                user_features: Optional[DataFrame] = None,
                item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        """
        Выдача рекомендаций для пользователей.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в `items`
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой `[user_id]` или ``array-like``;
            если None, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то вызывается ошибка
        :param items: список объектов, которые необходимо рекомендовать;
            спарк-датафрейм с колонкой `[item_id]` или ``array-like``;
            если None, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в relevance в рекомендациях к ним будет стоять 0
        :param context: контекст, в котором нужно получить рекомендации;
            если None, контекст не будет использоваться
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :param filter_seen_items: если True, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :return: рекомендации, спарк-датафрейм с колонками
            `[user_id , item_id , context , relevance]`
        """
        logging.debug("Проверка датафреймов")
        self._check_input_dataframes(log, user_features, item_features)
        spark = State().session

        users = self._extract_unique_if_needed(log, users, "user_id")
        items = self._extract_unique_if_needed(log, items, "item_id")

        num_items = items.count()
        if num_items < k:
            raise ValueError(
                "Значение k больше, чем множество объектов; "
                f"k = {k}, number of items = {num_items}")

        if context is None:
            context = constants.DEFAULT_CONTEXT

        recs = self._predict(log, k, users, items, context, user_features, item_features, filter_seen_items)
        spark = SparkSession(recs.rdd.context)
        recs = write_read_dataframe(
            recs,
            os.path.join(spark.conf.get("spark.local.dir"),
                         f"recs{uuid4()}.parquet")
        )

        return recs

    @staticmethod
    def _extract_unique_if_needed(log: DataFrame, array: Iterable, column: str):
        """
        Получить уникальные значения из ``array`` и положить в датафрейм с колонкой ``column``.
        Если ``array is None``, то вытащить значение из ``log``.
        """
        spark = State().session
        if array is None:
            logging.debug("Выделение дефолтных юзеров")
            array = log.select(column).distinct()
        elif not isinstance(array, DataFrame):
            if hasattr(array, "__iter__"):
                array = spark.createDataFrame(
                    data=pd.DataFrame(pd.unique(array),
                    columns=[column])
                )
            else:
                raise TypeError(f"Плохой аргумент {array}")
        return array

    @abstractmethod
    def _predict(self,
                 log: DataFrame,
                 k: int,
                 users: Optional[DataFrame] = None,
                 items: Optional[DataFrame] = None,
                 context: Optional[str] = None,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:
        """
        Метод-helper для получения рекомендаций.
        Должен быть имплементирован наследниками.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в `items`
        :param users: список пользователей, для которых необходимо получить
            рекомендации; если None, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то вызывается ошибка
        :param items: список объектов, которые необходимо рекомендовать;
            если None, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в рекомендациях к ним будет стоять 0
        :param context: контекст, в котором нужно получить рекомендоции;
            если None, контекст не будет использоваться
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :param filter_seen_items: если True, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :return: рекомендации, спарк-датафрейм с колонками
            `[user_id , item_id , context , relevance]`
        """

    def fit_predict(self,
                    log: DataFrame,
                    k: int,
                    users: Optional[DataFrame] = None,
                    items: Optional[DataFrame] = None,
                    context: Optional[str] = None,
                    user_features: Optional[DataFrame] = None,
                    item_features: Optional[DataFrame] = None,
                    filter_seen_items: bool = True) -> DataFrame:
        """
        Обучает модель и выдает рекомендации.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в `items`
        :param users: список пользователей, для которых необходимо получить
            рекомендации; если None, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то поднмиается исключение
        :param items: список объектов, которые необходимо рекомендовать;
            если None, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в рекомендациях к ним будет стоять 0
        :param context: контекст, в котором нужно получить рекомендоции;
            если None, контекст не будет использоваться
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :param filter_seen_items: если True, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :return: рекомендации, спарк-датафрейм с колонками
            `[user_id , item_id , context , relevance]`
        """
        self.fit(log, user_features, item_features)
        return self.predict(log, k, users, items, context, user_features, item_features, filter_seen_items)

    @staticmethod
    def _filter_seen_recs(recs: DataFrame, log: DataFrame) -> DataFrame:
        """
        Преобразует рекомендации, заменяя для каждого пользователя
        relevance уже виденных им объекты (на основе лога) на -1.

        :param recs: рекомендации, спарк-датафрейм с колонками
            `[user_id , item_id , context , relevance]`
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :return: измененные рекомендации, спарк-датафрейм с колонками
            `[user_id , item_id , context , relevance]`
        """
        user_item_log = (log
                         .select(sf.col("item_id").alias("item"),
                                 sf.col("user_id").alias("user"))
                         .withColumn("in_log", sf.lit(True)))
        recs = (recs
                .join(user_item_log,
                      (recs.item_id == user_item_log.item) &
                      (recs.user_id == user_item_log.user), how="left"))
        recs = (recs
                .withColumn("relevance",
                            sf.when(recs["in_log"], -1)
                            .otherwise(recs["relevance"]))
                .drop("in_log", "item", "user"))
        return recs
