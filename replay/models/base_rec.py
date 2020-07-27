"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Union

import pandas as pd
from pyspark.ml.feature import IndexToString, StringIndexer, StringIndexerModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.constants import AnyDataFrame
from replay.session_handler import State
from replay.utils import get_top_k_recs, convert2spark


class BaseRecommender(ABC):
    """ Базовый класс-рекомендатель. """

    model: Any = None
    user_indexer: StringIndexerModel
    item_indexer: StringIndexerModel
    inv_user_indexer: IndexToString
    inv_item_indexer: IndexToString
    _logger: Optional[logging.Logger] = None
    can_predict_cold_users: bool = False
    can_predict_cold_items: bool = False

    def set_params(self, **params: Dict[str, Any]) -> None:
        """
        Устанавливает параметры рекоммендера.

        :param params: словарь, ключ - название параметра,
            значение - значение параметра
        :return:
        """
        for param, value in params.items():
            setattr(self, param, value)

    def __str__(self):
        return type(self).__name__

    def _fit_wrap(
        self,
        log: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        force_reindex: bool = True,
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
        :param force_reindex: обязательно создавать
            индексы, даже если они были созданы ранее
        :return:
        """
        self.logger.debug("Начало обучения %s", type(self).__name__)
        log = convert2spark(log)
        if user_features is not None:
            user_features = convert2spark(user_features)
        if item_features is not None:
            item_features = convert2spark(item_features)
        if "user_indexer" not in self.__dict__ or force_reindex:
            self.logger.debug("Предварительная стадия обучения (pre-fit)")
            self._create_indexers(log, user_features, item_features)
        self.logger.debug("Основная стадия обучения (fit)")
        log = self._convert_index(log)
        if user_features is not None:
            user_features = self._convert_index(user_features)
        if item_features is not None:
            item_features = self._convert_index(item_features)
        self._fit(log, user_features, item_features)

    def _create_indexers(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        Метод для создания индексеров.
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param user_features: свойствва пользователей (обязательно содержат колонку ``user_id``)
        :param item_features: свойствва объектов (обязательно содержат колонку ``item_id``)
        :return:
        """
        if user_features is None:
            users = log.select("user_id")
        else:
            users = log.select("user_id").union(
                user_features.select("user_id")
            )
        if item_features is None:
            items = log.select("item_id")
        else:
            items = log.select("item_id").union(
                item_features.select("item_id")
            )
        self.user_indexer = StringIndexer(
            inputCol="user_id", outputCol="user_idx"
        ).fit(users)
        self.item_indexer = StringIndexer(
            inputCol="item_id", outputCol="item_idx"
        ).fit(items)
        self.inv_user_indexer = IndexToString(
            inputCol="user_idx",
            outputCol="user_id",
            labels=self.user_indexer.labels,
        )
        self.inv_item_indexer = IndexToString(
            inputCol="item_idx",
            outputCol="item_id",
            labels=self.item_indexer.labels,
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

    # pylint: disable=too-many-arguments
    def _predict_wrap(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
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
        self.logger.debug("Начало предикта %s", type(self).__name__)
        log = convert2spark(log)
        if user_features is not None:
            user_features = convert2spark(user_features)
        if item_features is not None:
            item_features = convert2spark(item_features)
        users = self._extract_unique(log, users, "user_id")
        items = self._extract_unique(log, items, "item_id")
        users = self._convert_index(users)
        items = self._convert_index(items)
        if item_features is not None:
            item_features = self._convert_index(item_features)
        if user_features is not None:
            user_features = self._convert_index(user_features)
        log = self._convert_index(log)

        num_items = items.count()
        if num_items < k:
            raise ValueError(
                "Значение k больше, чем множество объектов; "
                f"k = {k}, number of items = {num_items}"
            )
        recs = self._predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )
        if filter_seen_items:
            recs = self._mark_seen_items(recs, self._convert_index(log))
        recs = self._convert_back(recs).select(
            "user_id", "item_id", "relevance"
        )
        recs = get_top_k_recs(recs, k)
        recs = (
            recs.withColumn(
                "relevance",
                sf.when(recs["relevance"] < 0, 0).otherwise(recs["relevance"]),
            )
        ).cache()

        return recs

    def _convert_index(self, data_frame: DataFrame) -> DataFrame:
        """
        Строковые индексы в полях ``user_id``, ``item_id`` заменяются на
        числовые индексы ``user_idx`` и ``item_idx`` соответственно

        :param data_frame: спарк-датафрейм со строковыми индексами
        :return: спарк-датафрейм с числовыми индексами
        """
        if "user_id" in data_frame.columns:
            self._reindex("user", data_frame)
            data_frame = self.user_indexer.transform(data_frame).drop(
                "user_id"
            )
            data_frame = data_frame.withColumn(
                "user_idx", sf.col("user_idx").cast("int")
            )
        if "item_id" in data_frame.columns:
            self._reindex("item", data_frame)
            data_frame = self.item_indexer.transform(data_frame).drop(
                "item_id"
            )
            data_frame = data_frame.withColumn(
                "item_idx", sf.col("item_idx").cast("int")
            )
        return data_frame

    def _convert_back(self, log):
        return self.inv_user_indexer.transform(
            self.inv_item_indexer.transform(log)
        ).drop("user_idx", "item_idx")

    def _reindex(self, entity: str, objects: DataFrame):
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
            map(
                str,
                objects.select(sf.collect_list(indexer.getInputCol())).first()[
                    0
                ],
            )
        ).difference(indexer.labels)
        if new_objects:
            if can_reindex:
                new_labels = indexer.labels + list(new_objects)
                setattr(
                    self,
                    f"{entity}_indexer",
                    indexer.from_labels(
                        new_labels,
                        inputCol=indexer.getInputCol(),
                        outputCol=indexer.getOutputCol(),
                        handleInvalid="error",
                    ),
                )
                inv_indexer.setLabels(new_labels)
            else:
                message = (
                    f"Список {entity} содержит элементы, которые "
                    "отсутствовали при обучении. Результат "
                    "предсказания будет не полным."
                )
                self.logger.warning(message)
                indexer.setHandleInvalid("skip")

    def _extract_unique(
        self,
        log: AnyDataFrame,
        array: Optional[Union[Iterable, AnyDataFrame]],
        column: str,
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

    # pylint: disable=too-many-arguments
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

    @staticmethod
    def _mark_seen_items(recs: DataFrame, log: DataFrame) -> DataFrame:
        """
        Преобразует рекомендации, заменяя для каждого пользователя
        relevance уже увиденных им объектов (на основе лога) на -1.

        :param recs: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :return: измененные рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        user_item_log = log.select(
            sf.col("item_idx").alias("item"), sf.col("user_idx").alias("user")
        ).withColumn("in_log", sf.lit(True))
        recs = recs.join(
            user_item_log,
            (recs.item_idx == user_item_log.item)
            & (recs.user_idx == user_item_log.user),
            how="left",
        )
        recs = recs.withColumn(
            "relevance",
            sf.when(recs["in_log"], -1).otherwise(recs["relevance"]),
        ).drop("in_log", "item", "user")
        return recs

    @property
    def logger(self) -> logging.Logger:
        """
        :returns: стандартный логгер библиотеки
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger

    @property
    def users_count(self) -> int:
        """
        :returns: количество пользователей в обучающей выборке; выдаёт ошибку, если модель не обучена
        """
        try:
            return len(self.user_indexer.labels)
        except AttributeError:
            raise AttributeError(
                "Перед вызовом этого свойства нужно вызвать метод fit"
            )

    @property
    def items_count(self) -> int:
        """
        :returns: количество объектов в обучающей выборке; выдаёт ошибку, если модель не обучена
        """
        try:
            return len(self.item_indexer.labels)
        except AttributeError:
            raise AttributeError(
                "Перед вызовом этого свойства нужно вызвать метод fit"
            )

    def _fit_predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        filter_seen_items: bool = True,
        force_reindex: bool = True,
    ) -> DataFrame:
        self._fit_wrap(log, user_features, item_features, force_reindex)
        return self._predict_wrap(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )


# pylint: disable=abstract-method
class HybridRecommender(BaseRecommender):
    """Рекомендатель, учитывающий фичи"""

    def fit(
        self,
        log: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        force_reindex: bool = True,
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
        :param force_reindex: обязательно создавать
            индексы, даже если они были созданы ранее
        :return:
        """
        self._fit_wrap(
            log=log,
            user_features=user_features,
            item_features=item_features,
            force_reindex=force_reindex,
        )

    # pylint: disable=too-many-arguments
    def predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
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
        return self._predict_wrap(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
        )

    # pylint: disable=too-many-arguments
    def fit_predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        filter_seen_items: bool = True,
        force_reindex: bool = True,
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
        :param force_reindex: обязательно создавать
            индексы, даже если они были созданы ранее
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        return self._fit_predict(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
            force_reindex=force_reindex,
        )


# pylint: disable=abstract-method
class Recommender(BaseRecommender):
    """Обычный рекомендатель"""

    def fit(self, log: AnyDataFrame, force_reindex: bool = True) -> None:
        """
        Обучает модель на логе и признаках пользователей и объектов.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param force_reindex: обязательно создавать
            индексы, даже если они были созданы ранее
        :return:
        """
        self._fit_wrap(
            log=log,
            user_features=None,
            item_features=None,
            force_reindex=force_reindex,
        )

    # pylint: disable=too-many-arguments
    def predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
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
        :param filter_seen_items: если True, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        return self._predict_wrap(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=None,
            item_features=None,
            filter_seen_items=filter_seen_items,
        )

    # pylint: disable=too-many-arguments
    def fit_predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        force_reindex: bool = True,
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
        :param filter_seen_items: если ``True``, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :param force_reindex: обязательно создавать
            индексы, даже если они были созданы ранее
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        return self._fit_predict(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=None,
            item_features=None,
            filter_seen_items=filter_seen_items,
            force_reindex=force_reindex,
        )
