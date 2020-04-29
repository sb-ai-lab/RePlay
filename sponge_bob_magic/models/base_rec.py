"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Union

import pandas as pd
from pyspark.ml.feature import IndexToString, StringIndexer, StringIndexerModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf

from sponge_bob_magic.converter import convert, get_type
from sponge_bob_magic.session_handler import State
from sponge_bob_magic.utils import get_top_k_recs


class Recommender(ABC):
    """ Базовый класс-рекомендатель. """

    model: Any = None
    user_indexer: StringIndexerModel
    item_indexer: StringIndexerModel
    inv_user_indexer: IndexToString
    inv_item_indexer: IndexToString
    _logger: Optional[logging.Logger] = None
    _spark: Optional[SparkSession] = None
    can_predict_cold_users: bool = False
    can_predict_cold_items: bool = False

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
                    f"которые принимает рекоммендер: {valid_params}"
                )

            setattr(self, param, value)

    @abstractmethod
    def get_params(self) -> Dict[str, object]:
        """
        Возвращает параметры рекоммендера в виде словаря.

        :return: словарь параметров, ключ - название параметра,
            значение - значение параметра
        """

    def __repr__(self):
        return (
            type(self).__name__
            + "("
            + ", ".join(
                [f"{key}={self.get_params()[key]}" for key in self.get_params()]
            )
            + ")"
        )

    def __str__(self):
        return type(self).__name__

    def fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        Обучает модель на логе и признаках пользователей и объектов.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id, timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id, timestamp]`` и колонки с признаками
        :return:
        """
        log = convert(log)
        user_features = convert(user_features)
        item_features = convert(item_features)

        self.logger.debug("Предварительная стадия обучения (pre-fit)")
        self._pre_fit(log, user_features, item_features)
        self.logger.debug("Основная стадия обучения (fit)")
        self._fit(log, user_features, item_features)

    def _pre_fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        Метод-helper для обучения модели, в котором параметры не используются.
        Нужен для того, чтобы вынести вычисление трудоемких агрегатов
        в отдельный метод, который по возможности будет вызываться один раз.
        Может быть имплементирован наследниками.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id, timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id, timestamp]`` и колонки с признаками
        :return:
        """
        self.user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx").fit(
            log
        )
        self.item_indexer = StringIndexer(inputCol="item_id", outputCol="item_idx").fit(
            log
        )
        self.inv_user_indexer = IndexToString(
            inputCol="user_idx", outputCol="user_id", labels=self.user_indexer.labels
        )
        self.inv_item_indexer = IndexToString(
            inputCol="item_idx", outputCol="item_id", labels=self.item_indexer.labels
        )

    @abstractmethod
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        Метод для обучения модели.
        Должен быть имплементирован наследниками.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id, timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id, timestamp]`` и колонки с признаками
        :return:
        """

    def predict(
        self,
        log: DataFrame,
        k: int,
        users: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Выдача рекомендаций для пользователей.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в ``items``
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой ``[user_id]`` или ``array-like``;
            если ``None``, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то вызывается ошибка
        :param items: список объектов, которые необходимо рекомендовать;
            спарк-датафрейм с колонкой ``[item_id]`` или ``array-like``;
            если ``None``, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в ``relevance`` в рекомендациях к ним будет стоять ``0``
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id , timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id , timestamp]`` и колонки с признаками
        :param filter_seen_items: если True, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        type_in = get_type(log)
        log = convert(log)
        user_features = convert(user_features)
        item_features = convert(item_features)

        users = self._extract_unique(log, users, "user_id")
        items = self._extract_unique(log, items, "item_id")
        if ("item_indexer" in self.__dict__ and
                "inv_item_indexer" in self.__dict__):
            self._reindex("item", items)
        if ("user_indexer" in self.__dict__ and
                "inv_user_indexer" in self.__dict__):
            self._reindex("user", users)

        num_items = items.count()
        if num_items < k:
            raise ValueError(
                "Значение k больше, чем множество объектов; "
                f"k = {k}, number of items = {num_items}"
            )
        recs = self._predict(
            log, k, users, items, user_features, item_features, filter_seen_items
        )
        if filter_seen_items:
            recs = self._filter_seen_recs(recs, log)
        recs = get_top_k_recs(recs, k)
        recs = (
            recs.withColumn(
                "relevance",
                sf.when(recs["relevance"] < 0, 0).otherwise(recs["relevance"]),
            )
        ).cache()
        return convert(recs, type_in)

    def _reindex(self,
                 entity: str,
                 objects: DataFrame):
        """
           Переиндексирование пользователей/объектов. В случае если
           рекомендатель может работать с пользователями/объектами не из
           обучения, индексатор дополняется соответствующими элементами.

           :param entity: название сушности item или user
           :param objects: DataFrame со столбцом уникальных
           пользователей/объектов
        """
        indexer = getattr(self, f"{entity}_indexer")
        inv_indexer = getattr(self, f"inv_{entity}_indexer")
        can_reindex = getattr(self, f"can_predict_cold_{entity}s")

        new_objects = set(
            map(str, objects.select(sf.collect_list(indexer.getInputCol()))
                .first()[0])
        ).difference(indexer.labels)
        if new_objects:
            if can_reindex:
                new_labels = indexer.labels + list(new_objects)
                setattr(self, f"{entity}_indexer", indexer.from_labels(
                    new_labels,
                    inputCol=indexer.getInputCol(),
                    outputCol=indexer.getOutputCol(),
                    handleInvalid="error")
                )
                inv_indexer.setLabels(new_labels)
            else:
                self.logger.debug("Список пользователей или объектов содержит "
                                  "элементы, которые отсутствовали при "
                                  "обучении. Результат предсказания будет не "
                                  "полным.")
                indexer.setHandleInvalid("skip")

    def _extract_unique(
        self, log: DataFrame, array: Union[Iterable, DataFrame], column: str
    ) -> DataFrame:
        """
        Получить уникальные значения из ``array`` и положить в датафрейм с колонкой ``column``.
        Если ``array is None``, то вытащить значение из ``log``.
        """
        spark = State().session
        if array is None:
            self.logger.debug("Выделение дефолтных юзеров")
            unique = log.select(column).distinct()
        elif not isinstance(array, DataFrame):
            if isinstance(array, Iterable):
                unique = spark.createDataFrame(
                    data=pd.DataFrame(pd.unique(list(array)), columns=[column])
                )
        else:
            unique = array.select(column).distinct()
        return unique

    @abstractmethod
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
        """
        Метод-helper для получения рекомендаций.
        Должен быть имплементирован наследниками.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в ``items``
        :param users: список пользователей, для которых необходимо получить
            рекомендации; если ``None``, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то вызывается ошибка
        :param items: список объектов, которые необходимо рекомендовать;
            если ``None``, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в рекомендациях к ним будет стоять ``0``
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id , timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id , timestamp]`` и колонки с признаками
        :param filter_seen_items: если ``True``, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """

    def fit_predict(
        self,
        log: DataFrame,
        k: int,
        users: Optional[DataFrame] = None,
        items: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Обучает модель и выдает рекомендации.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в ``items``
        :param users: список пользователей, для которых необходимо получить
            рекомендации; если ``None``, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то поднмиается исключение
        :param items: список объектов, которые необходимо рекомендовать;
            если ``None``, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в рекомендациях к ним будет стоять ``0``
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id , timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id , timestamp]`` и колонки с признаками
        :param filter_seen_items: если ``True``, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        self.fit(log, user_features, item_features)
        return self.predict(
            log, k, users, items, user_features, item_features, filter_seen_items
        )

    @staticmethod
    def _filter_seen_recs(recs: DataFrame, log: DataFrame) -> DataFrame:
        """
        Преобразует рекомендации, заменяя для каждого пользователя
        relevance уже виденных им объекты (на основе лога) на -1.

        :param recs: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :return: измененные рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        user_item_log = log.select(
            sf.col("item_id").alias("item"), sf.col("user_id").alias("user")
        ).withColumn("in_log", sf.lit(True))
        recs = recs.join(
            user_item_log,
            (recs.item_id == user_item_log.item) & (recs.user_id == user_item_log.user),
            how="left",
        )
        recs = recs.withColumn(
            "relevance", sf.when(recs["in_log"], -1).otherwise(recs["relevance"])
        ).drop("in_log", "item", "user")
        return recs

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = logging.getLogger("sponge_bob_magic")
        return self._logger

    @property
    def users_count(self) -> int:
        """
        :returns: количество пользователей в обучающей выборке; выдаёт ошибку, если модель не обучена
        """
        try:
            return len(self.user_indexer.labels)
        except AttributeError:
            raise AttributeError("Перед вызовом этого свойства нужно вызвать метод fit")

    @property
    def spark(self) -> SparkSession:
        if self._spark is None:
            self._spark = State().session
        return self._spark

    @property
    def items_count(self) -> int:
        """
        :returns: количество объектов в обучающей выборке; выдаёт ошибку, если модель не обучена
        """
        try:
            return len(self.item_indexer.labels)
        except AttributeError:
            raise AttributeError("Перед вызовом этого свойства нужно вызвать метод fit")
