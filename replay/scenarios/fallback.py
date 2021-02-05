# pylint: disable=protected-access
from typing import Optional, Dict, List, Any, Tuple

from pyspark.sql import DataFrame

from replay.constants import AnyDataFrame
from replay.metrics import Metric, NDCG
from replay.models.base_rec import HybridRecommender, BaseRecommender
from replay.utils import fallback


class Fallback(HybridRecommender):
    """Дополняет основную модель рекомендациями с помощью fallback модели.
    Ведет себя точно также, как обычный рекомендатель и имеет такой же интерфейс."""

    def __init__(
        self, main_model: BaseRecommender, fallback_model: BaseRecommender
    ):
        """Для каждого пользователя будем брать рекомендации от `main_model`, а если не хватает,
        то дополним рекомендациями от `fallback_model` снизу. `relevance` побочной модели при этом
        будет изменен, чтобы не оказаться выше, чем у основной модели.

        :param main_model: основная инициализированная модель
        :param fallback_model: дополнительная инициализированная модель
        """
        self.main_model = main_model
        # pylint: disable=invalid-name
        self.fb_model = fallback_model

    def __str__(self):
        return f"Fallback({str(self.main_model)}, {str(self.fb_model)})"

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
        :param param_grid: словарь с ключами main, fallback, и значеними в виде сеток параметров.
            Сетка задается словарем, где ключ ---
            название параметра, значение --- границы возможных значений.
            ``{param: [low, high]}``.
        :param criterion: метрика, которая будет оптимизироваться
        :param k: количество рекомендаций для каждого пользователя
        :param budget: количество попыток при поиске лучших гиперпараметров
        :return: словари оптимальных параметров
        """
        if param_grid is None:
            param_grid = {"main": None, "fallback": None}
        self.logger.info("Optimizing main model...")
        params = self.main_model.optimize(
            train,
            test,
            user_features,
            item_features,
            param_grid["main"],
            criterion,
            k,
            budget,
        )
        self.main_model.set_params(**params)
        if self.fb_model._search_space is not None:
            self.logger.info("Optimizing fallback model...")
            fb_params = self.fb_model.optimize(
                train,
                test,
                user_features,
                item_features,
                param_grid["fallback"],
                criterion,
                k,
                budget,
            )
            self.fb_model.set_params(**fb_params)
        else:
            fb_params = None
        return params, fb_params

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.main_model._fit(log, user_features, item_features)
        self.fb_model._fit(log, user_features, item_features)

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
        pred = self.main_model._predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )
        extra_pred = self.fb_model._predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )
        pred = fallback(pred, extra_pred, k)
        return pred
