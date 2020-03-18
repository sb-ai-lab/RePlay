"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from typing import Any, Dict, Optional

from optuna import Study, create_study, samplers
from pyspark.sql import DataFrame

from sponge_bob_magic.constants import DEFAULT_CONTEXT, IterOrList
from sponge_bob_magic.metrics import HitRate, Metric, Surprisal, Unexpectedness
from sponge_bob_magic.models import KNNRec, PopRec, Recommender
from sponge_bob_magic.scenarios.main_objective import MainObjective, SplitData
from sponge_bob_magic.session_handler import State
from sponge_bob_magic.splitters.base_splitter import Splitter
from sponge_bob_magic.splitters.log_splitter import RandomSplitter


class MainScenario:
    """
    Основной сценарий. По умолчанию делает следующее:

    * разбивает лог случайно 70/30 (холодных пользователей и объекты просто выбрасывает)
    * обучает дефолтный рекомендатель (:ref:`KNN <knn-model>`)
    * для тех случаев, когда `KNN` выдаёт слишком мало рекомендаций (мешьше, top-N, которые требуются), добирает рекомендации из fallback-рекомендателя
      (:ref:`PopRec <pop-rec>`)
    * оптимизирует, подбирая гиперпараметры, и включает в отчёт только :ref:`HitRate <hit-rate>`
    """
    def __init__(
            self,
            splitter: Splitter = RandomSplitter(0.3, True, True),
            recommender: Recommender = KNNRec(),
            criterion: type = HitRate,
            metrics: Dict[type, IterOrList] = dict(),
            fallback_rec: Recommender = PopRec()
    ):
        """
        Отдельные блоки сценария можно изменять по своему усмотрению

        :param splitter: как разбивать на train/test
        :param recommender: Бейзлайн; объект класса, который необходимо обучить
        :param criterion: метрика, которая будет оптимизироваться при переборе гипер-параметров
        :param metrics: какие ещё метрики, кроме критерия оптимизации, включить в отчёт об эксперименте
        :param fallback_rec: "запасной" рекомендатель, с помощью которого можно дополнять выдачу базового рекомендателя,
                             если вдруг он выдаёт меньшее количество объектов, чем было запрошено
        """
        self.splitter = splitter
        self.recommender = recommender
        self.criterion = criterion
        self.metrics = metrics
        self.fallback_rec = fallback_rec

    optuna_study: Optional[Study]
    optuna_max_n_trials: int = 100
    optuna_n_jobs: int = 1
    filter_seen_items: bool = True
    study: Study
    _optuna_seed: Optional[int] = None

    def _prepare_data(
            self,
            log: DataFrame,
            users: Optional[DataFrame] = None,
            items: Optional[DataFrame] = None,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
    ) -> SplitData:
        """ Делит лог и готовит объекти типа `SplitData`. """
        train, test = self.splitter.split(log)
        logging.debug(
            "Длина трейна и теста: %d %d", train.count(), test.count()
        )
        logging.debug(
            "Количество пользователей в трейне и тесте: %d, %d",
            train.select('user_id').distinct().count(),
            test.select('user_id').distinct().count()
        )
        logging.debug(
            "Количество объектов в трейне и тесте: %d, %d",
            train.select('item_id').distinct().count(),
            test.select('item_id').distinct().count()
        )
        # если users или items нет, возьмем всех из теста,
        # чтобы не делать на каждый trial их заново
        users = users if users else test.select("user_id").distinct().cache()
        items = items if items else test.select("item_id").distinct().cache()
        split_data = SplitData(train, test,
                               users, items,
                               user_features, item_features)
        return split_data

    def _run_optimization(
            self,
            n_trials: int,
            params_grid: Dict[str, Dict[str, Any]],
            split_data: SplitData,
            criterion: Metric,
            metrics: Dict[Metric, IterOrList],
            k: int = 10,
            context: Optional[str] = None,
            fallback_recs: Optional[DataFrame] = None
    ) -> Dict[str, Any]:
        """ Запускает подбор параметров в optuna. """
        sampler = samplers.RandomSampler(seed=self._optuna_seed)
        self.study = create_study(direction="maximize", sampler=sampler)
        # делаем триалы до тех пор, пока не засемплим уникальных n_trials или
        # не используем максимально попыток
        count = 1
        n_unique_trials = 0
        while n_trials > n_unique_trials and count <= self.optuna_max_n_trials:
            spark = State().session
            self.study.optimize(
                MainObjective(
                    params_grid, self.study, split_data,
                    self.recommender, criterion, metrics,
                    k, context,
                    fallback_recs,
                    self.filter_seen_items,
                    spark.conf.get("spark.local.dir")),
                n_trials=1,
                n_jobs=self.optuna_n_jobs
            )
            count += 1
            n_unique_trials = len(
                set(str(t.params)
                    for t in self.study.trials)
            )
        logging.debug("Лучшие значения метрики: %d", self.study.best_value)
        logging.debug("Лучшие параметры: %s", self.study.best_params)
        return self.study.best_params

    def research(
            self,
            params_grid: Dict[str, Dict[str, Any]],
            log: DataFrame,
            users: Optional[DataFrame] = None,
            items: Optional[DataFrame] = None,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            k: int = 10,
            context: Optional[str] = None,
            n_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Обучает и подбирает параметры для модели.

        :param params_grid: сетка параметров, задается словарем, где ключ -
            название параметра (должен совпадать с одним из параметров модели,
            которые возвращает ``get_params()``), значение - словарь с двумя
            ключами "type" и "args", где они должны принимать следующие
            значения в соответствии
            с `optuna.trial.Trial.suggest_* <https://optuna.readthedocs.io/en/stable/reference/trial.html#optuna.trial.Trial.suggest_categorical>`_
            (строковое значение "type" и список значений аргументов "args"):
            "uniform" -> [low, high],
            "loguniform" -> [low, high],
            "discrete_uniform" -> [low, high, q],
            "int" -> [low, high],
            "categorical" -> [choices]
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_id , item_id , timestamp , context , relevance]``
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой ``[user_id]``;
            если None, выбираются все пользователи из тестовой выборки
        :param items: список объектов, которые необходимо рекомендовать;
            спарк-датафрейм с колонкой ``[item_id]``;
            если None, выбираются все объекты из тестовой выборки
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id , timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id , timestamp]`` и колонки с признаками
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в ``items``
        :param context: контекст, в котором нужно получить рекомендации
        :param n_trials: количество уникальных испытаний; должно быть от 1
            до значения параметра ``optuna_max_n_trials``
        :return: словарь оптимальных значений параметров для модели; ключ -
            название параметра (совпадают с параметрами модели,
            которые возвращает ``get_params()``), значение - значение параметра
        """
        context = context if context else DEFAULT_CONTEXT
        logging.debug("Деление лога на обучающую и тестовую выборку")
        split_data = self._prepare_data(log,
                                        users, items,
                                        user_features, item_features)
        logging.debug("Инициализация метрик")
        metrics = {}
        for metric in self.metrics:
            if metric in {Surprisal, Unexpectedness}:
                metrics[metric(split_data.train)] = self.metrics[metric]
            else:
                metrics[metric()] = self.metrics[metric]
        criterion = self.criterion()
        logging.debug("Обучение и предсказание дополнительной модели")
        fallback_recs = self._predict_fallback_recs(self.fallback_rec,
                                                    split_data, k, context)
        logging.debug("Пре-фит модели")
        self.recommender._pre_fit(split_data.train, split_data.user_features,
                                  split_data.item_features)
        logging.debug("-------------")
        logging.debug("Оптимизация параметров")
        logging.debug(
            "Максимальное количество попыток: %d %s", self.optuna_max_n_trials,
            "(чтобы поменять его, задайте параметр 'optuna_max_n_trials')"
        )
        best_params = self._run_optimization(n_trials, params_grid, split_data,
                                             criterion, metrics, k, context,
                                             fallback_recs)
        return best_params

    def _predict_fallback_recs(
            self,
            fallback_rec: Recommender,
            split_data: SplitData,
            k: int,
            context: Optional[str] = None
    ) -> Optional[DataFrame]:
        """ Обучает fallback модель и возвращает ее рекомендации. """
        fallback_recs = None
        if fallback_rec is not None:
            fallback_recs = fallback_rec.fit_predict(
                split_data.train,
                k,
                split_data.users, split_data.items,
                context,
                split_data.user_features,
                split_data.item_features,
                self.filter_seen_items
            )
        return fallback_recs

    def production(
            self,
            params: Dict[str, Any],
            log: DataFrame,
            users: Optional[DataFrame],
            items: Optional[DataFrame],
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            k: int = 10,
            context: Optional[str] = None
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
            ``[user_id , item_id , timestamp , context , relevance]``
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой ``[user_id]``;
            если None, выбираются все пользователи из тестовой выборки
        :param items: список объектов, которые необходимо рекомендовать;
            спарк-датафрейм с колонкой ``[item_id]``;
            если None, выбираются все объекты из тестовой выборки
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id , timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id , timestamp]`` и колонки с признаками
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в ``items``
        :param context: контекст, в котором нужно получить рекомендации
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id , item_id , context , relevance]``
        """
        self.recommender.set_params(**params)
        return self.recommender.fit_predict(
            log, k, users, items, context, user_features, item_features,
            self.filter_seen_items)
