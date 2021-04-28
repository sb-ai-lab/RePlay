# pylint: disable=too-many-arguments
# pragma: no cover
from abc import abstractmethod
from typing import Optional, Union, Iterable, Dict, List, Any, Tuple

from pyspark.sql import DataFrame

from replay.constants import AnyDataFrame
from replay.filters import min_entries
from replay.metrics import Metric, NDCG
from replay.models.base_rec import BaseRecommender
from replay.utils import convert2spark


class BaseScenario(BaseRecommender):
    """Базовый класс сценариев для предсказания всем пользователям"""

    can_predict_cold_users: bool = False

    def __init__(self, cold_model, threshold=5):
        self.threshold = threshold
        self.cold_model = cold_model
        self.hot_users = None

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
        hot_data = min_entries(log, self.threshold)
        self.hot_users = hot_data.select("user_id").distinct()
        self._fit_wrap(hot_data, user_features, item_features, force_reindex)
        self.cold_model._fit_wrap(
            log, user_features, item_features, force_reindex
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
        log = convert2spark(log)
        users = users or log or user_features or self.user_indexer.labels
        users = self._get_ids(users, "user_id")
        hot_data = min_entries(log, self.threshold)
        hot_users = hot_data.select("user_id").distinct()
        if not self.can_predict_cold_users:
            hot_users = hot_users.join(self.hot_users)
        hot_users = hot_users.join(users, on="user_id", how="inner")

        hot_pred = self._predict_wrap(
            log=hot_data,
            k=k,
            users=hot_users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
        )
        if log is not None:
            cold_data = log.join(self.hot_users, how="anti", on="user_id")
        else:
            cold_data = None
        cold_users = users.join(self.hot_users, how="anti", on="user_id")
        cold_pred = self.cold_model._predict_wrap(
            log=cold_data,
            k=k,
            users=cold_users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
        )
        return hot_pred.union(cold_pred)

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
        self.fit(log, user_features, item_features, force_reindex)
        return self.predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )

    # pylint: disable=too-many-arguments, too-many-locals
    def optimize(
        self,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_grid: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        criterion: Metric = NDCG(),
        k: int = 10,
        budget: int = 10,
    ) -> Tuple[Dict[str, Any]]:
        """
        Подбирает лучшие гиперпараметры с помощью optuna для обоих моделей
        и инициализирует эти значения.

        :param train: датафрейм для обучения
        :param test: датафрейм для проверки качества
        :param user_features: датафрейм с признаками пользователей
        :param item_features: датафрейм с признаками объектов
        :param param_grid: словарь с ключами main, cold, и значеними в виде сеток параметров.
            Сетка задается словарем, где ключ ---
            название параметра, значение --- границы возможных значений.
            ``{param: [low, high]}``.
        :param criterion: метрика, которая будет оптимизироваться
        :param k: количество рекомендаций для каждого пользователя
        :param budget: количество попыток при поиске лучших гиперпараметров
        :return: словари оптимальных параметров
        """
        if param_grid is None:
            param_grid = {"main": None, "cold": None}
        self.logger.info("Optimizing main model...")
        params = self._optimize(
            train,
            test,
            user_features,
            item_features,
            param_grid["main"],
            criterion,
            k,
            budget,
        )
        if not isinstance(params, tuple):
            self.set_params(**params)
        if self.cold_model._search_space is not None:
            self.logger.info("Optimizing cold model...")
            cold_params = self.cold_model._optimize(
                train,
                test,
                user_features,
                item_features,
                param_grid["cold"],
                criterion,
                k,
                budget,
            )
            if not isinstance(cold_params, tuple):
                self.cold_model.set_params(**cold_params)
        else:
            cold_params = None
        return params, cold_params

    @abstractmethod
    def _optimize(
        self,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_grid: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        criterion: Metric = NDCG(),
        k: int = 10,
        budget: int = 10,
    ):
        pass
