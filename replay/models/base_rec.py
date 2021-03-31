"""
Реализация абстрактных классов рекомендательных моделей:
- BaseRecommender - базовый класс для всех рекомендательных моделей
- Recommender - базовый класс для моделей, обучающихся на логе взаимодействия
- HybridRecommender - базовый класс для моделей, обучающихся на логе взаимодействия и признаках
"""
import collections
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Union, Sequence, List

import pandas as pd
from optuna import create_study
from optuna.samplers import TPESampler
from pyspark.ml.feature import IndexToString, StringIndexer, StringIndexerModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.constants import AnyDataFrame
from replay.metrics import Metric, NDCG
from replay.optuna_objective import SplitData, MainObjective
from replay.session_handler import State
from replay.utils import get_top_k_recs, convert2spark


class BaseRecommender(ABC):
    """ Базовый класс-рекомендатель. """

    model: Any
    user_indexer: StringIndexerModel
    item_indexer: StringIndexerModel
    inv_user_indexer: IndexToString
    inv_item_indexer: IndexToString
    _logger: Optional[logging.Logger] = None
    can_predict_cold_users: bool = False
    can_predict_cold_items: bool = False
    _search_space: Optional[
        Dict[str, Union[str, Sequence[Union[str, int, float]]]]
    ] = None
    _objective = MainObjective

    # pylint: disable=too-many-arguments, too-many-locals
    def optimize(
        self,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        criterion: Metric = NDCG(),
        k: int = 10,
        budget: int = 10,
    ) -> Dict[str, Any]:
        """
        Подбирает лучшие гиперпараметры с помощью optuna.

        :param train: датафрейм для обучения
        :param test: датафрейм для проверки качества
        :param user_features: датафрейм с признаками пользователей
        :param item_features: датафрейм с признаками объектов
        :param param_grid: сетка параметров, задается словарем, где ключ ---
            название параметра, значение --- границы возможных значений.
            ``{param: [low, high]}``.
        :param criterion: метрика, которая будет оптимизироваться
        :param k: количество рекомендаций для каждого пользователя
        :param budget: количество попыток при поиске лучших гиперпараметров
        :return: словарь оптимальных параметров
        """
        if self._search_space is None:
            self.logger.warning(
                "%s has no hyper parameters to optimize", str(self)
            )
            return None
        train = convert2spark(train)
        test = convert2spark(test)

        user_features_train, user_features_test = self._train_test_features(
            train, test, user_features, "user_id"
        )
        item_features_train, item_features_test = self._train_test_features(
            train, test, item_features, "item_id"
        )

        users = test.select("user_id").distinct()
        items = test.select("item_id").distinct()
        split_data = SplitData(
            train,
            test,
            users,
            items,
            user_features_train,
            user_features_test,
            item_features_train,
            item_features_test,
        )
        if param_grid is None:
            params = self._search_space.keys()
            vals = [None] * len(params)
            param_grid = dict(zip(params, vals))
        study = create_study(direction="maximize", sampler=TPESampler())
        objective = self._objective(
            search_space=param_grid,
            split_data=split_data,
            recommender=self,
            criterion=criterion,
            k=k,
        )
        study.optimize(objective, budget)
        return study.best_params

    @staticmethod
    def _train_test_features(train, test, features, column):
        if features is not None:
            features = convert2spark(features)
            features_train = features.join(train.select(column), on=column)
            features_test = features.join(test.select(column), on=column)
        else:
            features_train = None
            features_test = None
        return features_train, features_test

    def set_params(self, **params: Dict[str, Any]) -> None:
        """
        Устанавливает параметры модели.

        :param params: словарь, ключ - название параметра,
            значение - значение параметра
        :return:
        """
        for param, value in params.items():
            setattr(self, param, value)
        self._clear_cache()

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
        user_features = convert2spark(user_features)
        item_features = convert2spark(item_features)

        if "user_indexer" not in self.__dict__ or force_reindex:
            self.logger.debug("Предварительная стадия обучения (pre-fit)")
            self._create_indexers(log, user_features, item_features)
        self.logger.debug("Основная стадия обучения (fit)")

        log = self._convert_index(log)
        user_features = self._convert_index(user_features)
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
        :param user_features: свойства пользователей (обязательно содержат колонку ``user_id``)
        :param item_features: свойства объектов (обязательно содержат колонку ``item_id``)
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
        log: Optional[AnyDataFrame],
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
        user_features = convert2spark(user_features)
        item_features = convert2spark(item_features)

        user_data = users or log or user_features or self.user_indexer.labels
        users = self._get_ids(user_data, "user_id")
        user_type = users.schema["user_id"].dataType

        item_data = items or log or item_features or self.item_indexer.labels
        items = self._get_ids(item_data, "item_id")
        item_type = items.schema["item_id"].dataType

        log = self._convert_index(log)
        users = self._convert_index(users)
        items = self._convert_index(items)
        item_features = self._convert_index(item_features)
        user_features = self._convert_index(user_features)
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
        if filter_seen_items and log:
            recs = recs.join(
                log.withColumnRenamed("item_idx", "item")
                .withColumnRenamed("user_idx", "user")
                .select("user", "item"),
                on=(sf.col("user_idx") == sf.col("user"))
                & (sf.col("item_idx") == sf.col("item")),
                how="anti",
            ).drop("user", "item")
        recs = self._convert_back(recs, user_type, item_type).select(
            "user_id", "item_id", "relevance"
        )
        recs = get_top_k_recs(recs, k)
        return recs

    def _convert_index(self, data_frame: DataFrame) -> DataFrame:
        """
        Строковые индексы в полях ``user_id``, ``item_id`` заменяются на
        числовые индексы ``user_idx`` и ``item_idx`` соответственно

        :param data_frame: спарк-датафрейм со строковыми индексами
        :return: спарк-датафрейм с числовыми индексами
        """
        if data_frame is None:
            return None
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

    def _convert_back(self, log, user_type, item_type):
        res = self.inv_user_indexer.transform(
            self.inv_item_indexer.transform(log)
        ).drop("user_idx", "item_idx")
        res = res.withColumn("user_id", res["user_id"].cast(user_type))
        res = res.withColumn("item_id", res["item_id"].cast(item_type))
        return res

    def _reindex(self, entity: str, objects: DataFrame):
        """
           Переиндексирование пользователей/объектов. В случае если
           рекомендатель может работать с пользователями/объектами не из
           обучения, индексатор дополняется соответствующими элементами.

           :param entity: название сущности item или user
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

    @staticmethod
    def _get_ids(
        log: Union[Iterable, AnyDataFrame], column: str,
    ) -> DataFrame:
        """
        Получить уникальные значения из ``array`` и положить в датафрейм с колонкой ``column``.
        Если ``array is None``, то вытащить значение из ``log``.
        """
        spark = State().session
        if isinstance(log, DataFrame):
            unique = log.select(column).distinct()
        elif isinstance(log, collections.abc.Iterable):
            unique = spark.createDataFrame(
                data=pd.DataFrame(pd.unique(list(log)), columns=[column])
            )
        else:
            raise ValueError("Wrong type %s" % type(log))
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

    def _clear_cache(self):
        """
        Очищает закэшированные данные spark.
        """


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
            не знает, то поднимается исключение
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
            не знает, то поднимается исключение
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


class UserRecommender(BaseRecommender):
    """Использует фичи пользователей, но не использует фичи айтемов. Лог — необязательный параметр."""

    def fit(
        self,
        log: AnyDataFrame,
        user_features: AnyDataFrame,
        force_reindex: bool = True,
    ) -> None:
        """
        Выделить кластеры и посчитать популярность объектов в них.

        :param log: логи пользователей с историей для подсчета популярности объектов
        :param user_features: датафрейм связывающий `user_id` пользователей и их числовые признаки
        :param force_reindex: обязательно создавать
            индексы, даже если они были созданы ранее
        """
        self._fit_wrap(
            log=log, user_features=user_features, force_reindex=force_reindex
        )

    # pylint: disable=too-many-arguments
    def predict(
        self,
        user_features: AnyDataFrame,
        k: int,
        log: Optional[AnyDataFrame] = None,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Получить предсказания для переданных пользователей

        :param user_features: айди пользователей с числовыми фичами
        :param k: длина рекомендаций
        :param log: опциональный датафрейм с логами пользователей.
            Если передан, объекты отсюда удаляются из рекомендаций для соответствующих пользователей.
        :return: датафрейм с рекомендациями
        """
        return self._predict_wrap(
            log=log,
            user_features=user_features,
            k=k,
            filter_seen_items=filter_seen_items,
            users=users,
            items=items,
        )
