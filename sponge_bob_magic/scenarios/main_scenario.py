"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from typing import Any, Dict, Optional, Type

from optuna import create_study
from optuna.samplers import GridSampler
from pyspark.sql import DataFrame

from sponge_bob_magic.constants import IntOrList, NumType
from sponge_bob_magic.experiment import Experiment
from sponge_bob_magic.metrics import HitRate
from sponge_bob_magic.metrics.base_metric import Metric, RecOnlyMetric
from sponge_bob_magic.models import ALSWrap, PopRec, Recommender
from sponge_bob_magic.scenarios.main_objective import MainObjective, SplitData
from sponge_bob_magic.splitters.base_splitter import Splitter
from sponge_bob_magic.splitters.log_splitter import RandomSplitter


class MainScenario:
    """
    Основной сценарий. По умолчанию делает следующее:

    * разбивает лог случайно 70/30 (холодных пользователей и объекты просто выбрасывает)
    * обучает дефолтный рекомендатель (:ref:`ALS <als-rec>`)
    * для тех случаев, когда `KNN` выдаёт слишком мало рекомендаций (мешьше, top-N, которые требуются), добирает рекомендации из fallback-рекомендателя
      (:ref:`PopRec <pop-rec>`)
    * оптимизирует, подбирая гиперпараметры, и включает в отчёт только :ref:`HitRate <hit-rate>`
    """
    experiment: Experiment

    def __init__(
        self,
        splitter: Splitter = RandomSplitter(0.3, True, True),
        recommender: Recommender = ALSWrap(),
        criterion: Type[Metric] = HitRate,
        metrics: Dict[Type[Metric], IntOrList] = dict(),
        fallback_model: Recommender = PopRec()
    ):
        """
        Отдельные блоки сценария можно изменять по своему усмотрению

        :param splitter: как разбивать на train/test
        :param recommender: Бейзлайн; объект класса, который необходимо обучить
        :param criterion: метрика, которая будет оптимизироваться при переборе гипер-параметров
        :param metrics: какие ещё метрики, кроме критерия оптимизации, включить в отчёт об эксперименте
        :param fallback_model: "запасной" рекомендатель, с помощью которого можно дополнять выдачу базового рекомендателя,
                               если вдруг он выдаёт меньшее количество объектов, чем было запрошено
        """
        self.splitter = splitter
        self.recommender = recommender
        self.criterion = criterion
        self.metrics = metrics
        self.fallback_model = fallback_model
        self.logger = logging.getLogger("sponge_bob_magic")

    def _prepare_data(
        self,
        log: DataFrame,
        users: DataFrame,
        items: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame
    ) -> SplitData:
        """ Делит лог и готовит объекти типа ``SplitData``. """
        train, test = self.splitter.split(log)
        self.logger.debug("Длина трейна и теста: %d %d", train.count(), test.count())
        self.logger.debug(
            "Количество пользователей в трейне и тесте: %d, %d",
            train.select("user_id").distinct().count(),
            test.select("user_id").distinct().count(),
        )
        self.logger.debug(
            "Количество объектов в трейне и тесте: %d, %d",
            train.select("item_id").distinct().count(),
            test.select("item_id").distinct().count(),
        )
        users = users if users else test.select("user_id").distinct().cache()
        items = items if items else test.select("item_id").distinct().cache()
        split_data = SplitData(
            train.cache(), test.cache(), users, items, user_features, item_features
        )
        return split_data

    def _run_optimization(
        self,
        n_trials: int,
        params_grid: Dict[str, NumType],
        split_data: SplitData,
        criterion: Metric,
        metrics: Dict[Metric, IntOrList],
        k: int = 10,
        fallback_recs: Optional[DataFrame] = None
    ) -> Dict[str, Any]:
        """ Запускает подбор параметров в ``optuna``. """
        sampler = GridSampler(params_grid)
        study = create_study(direction="maximize", sampler=sampler)
        objective = MainObjective(
            params_grid,
            study,
            split_data,
            self.recommender,
            criterion,
            metrics,
            k,
            fallback_recs,
        )
        study.optimize(objective, n_trials)
        self.experiment = objective.experiment
        self.logger.debug("Лучшее значение метрики: %.2f", study.best_value)
        self.logger.debug("Лучшие параметры: %s", study.best_params)
        return study.best_params

    def research(
        self,
        params_grid: Dict[str, Dict[str, Any]],
        log: DataFrame,
        users: Optional[DataFrame] = None,
        items: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        k: int = 10,
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """
        Обучает и подбирает параметры для модели.

        :param params_grid: сетка параметров, задается словарем, где ключ -
            название параметра (должен совпадать с одним из параметров модели,
            которые возвращает ``get_params()``), значение - словарь с двумя
            ключами "type" и "args", где они должны принимать следующие
            значения в соответствии
            с `optuna.trial.Trial.suggest_* <https://optuna.readthedocs.io/en/stable/reference/trial.html#optuna.trial.Trial.suggest_categorical>`_
            (строковое значение ``type`` и список значений аргументов ``args``): ::

                "uniform" -> [low, high],
                "loguniform" -> [low, high],
                "discrete_uniform" -> [low, high, q],
                "int" -> [low, high],
                "categorical" -> [choices]

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой ``[user_id]``;
            если ``None``, выбираются все пользователи из тестовой выборки
        :param items: список объектов, которые необходимо рекомендовать;
            спарк-датафрейм с колонкой ``[item_id]``;
            если ``None``, выбираются все объекты из тестовой выборки
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id , timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id , timestamp]`` и колонки с признаками
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в ``items``
        :param n_trials: количество уникальных испытаний; должно быть от 1
            до значения параметра ``optuna_max_n_trials``
        :return: словарь оптимальных значений параметров для модели; ключ -
            название параметра (совпадают с параметрами модели,
            которые возвращает ``get_params()``), значение - значение параметра
        """
        self.logger.debug("Деление лога на обучающую и тестовую выборку")
        split_data = self._prepare_data(log, users, items, user_features, item_features)
        self.logger.debug("Инициализация метрик")
        criterion = self.criterion()
        metrics = {criterion: [k]}
        for metric in self.metrics:
            int_or_list = self.metrics[metric]
            if isinstance(int_or_list, list):
                k_list = int_or_list
            elif isinstance(int_or_list, int):
                k_list = [int_or_list]
            if issubclass(metric, RecOnlyMetric):
                metrics[metric(split_data.train)] = k_list
            else:
                metrics[metric()] = k_list
        self.logger.debug("Обучение и предсказание дополнительной модели")
        fallback_recs = self._fit_predict_fallback_recs(split_data, k)
        self.logger.debug("Пре-фит модели")
        self.recommender._pre_fit(split_data.train, split_data.user_features,
                                  split_data.item_features)
        self.logger.debug("Оптимизация параметров")
        self.logger.debug("Количество попыток: %d", n_trials)
        best_params = self._run_optimization(n_trials, params_grid, split_data,
                                             criterion, metrics, k,
                                             fallback_recs)
        return best_params

    def _fit_predict_fallback_recs(
            self,
            split_data: SplitData,
            k: int
    ) -> Optional[DataFrame]:
        """ Обучает fallback модель и возвращает ее рекомендации. """
        fallback_recs = None
        if self.fallback_model is not None:
            fallback_recs = self.fallback_model.fit_predict(
                split_data.train,
                k,
                split_data.users,
                split_data.items,
                split_data.user_features,
                split_data.item_features,
            )
        return fallback_recs

    def production(
        self,
        params: Dict[str, Any],
        log: DataFrame,
        users: Optional[DataFrame] = None,
        items: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        k: int = 10,
    ) -> DataFrame:
        """
        Обучает модель с нуля при заданных параметрах ``params`` и формирует
        рекомендации для ``users`` и ``items``.
        В качестве выборки для обучения используется весь лог, без деления.

        :param params: словарь значений параметров для модели; ключ -
            название параметра (должен совпадать с одним из параметров модели,
            которые возвращает ``get_params()``), значение - значение параметра
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой ``[user_id]``;
            если ``None``, выбираются все пользователи из тестовой выборки
        :param items: список объектов, которые необходимо рекомендовать;
            спарк-датафрейм с колонкой ``[item_id]``;
            если ``None``, выбираются все объекты из тестовой выборки
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id , timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id , timestamp]`` и колонки с признаками
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в ``items``
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        self.recommender.set_params(**params)
        return self.recommender.fit_predict(
            log, k, users, items, user_features, item_features
        )
