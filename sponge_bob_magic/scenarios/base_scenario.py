"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

from optuna import Study
from pyspark.sql import DataFrame, SparkSession


class Scenario(ABC):
    """ Базовый класс сценария. """

    optuna_study: Optional[Study]
    optuna_max_n_trials: int = 100
    optuna_n_jobs: int = 1
    filter_seen_items: bool = True

    def __init__(self, spark: SparkSession, **kwargs):
        """
        Инициализирует сценарий и сохраняет параметры, если они есть.

        :param spark: инициализированная спарк-сессия
        :param kwargs: дополнительные параметры классов-наследников
        """
        self.spark = spark

    @abstractmethod
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
            n_trials: int = 10,
            path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Обучает и подбирает параметры для модели.

        :param params_grid: сетка параметров, задается словарем, где ключ -
            название параметра (должен совпадать с одним из параметров модели,
            которые возвращает `get_params()`), значение - словарь с двумя
            ключами "type" и "args", где они должны принимать следующие
            значения в соответствии с optuna.trial.Trial.suggest_*
            (строковое значение "type" и список значений аргументов "args"):
            "uniform" -> [low, high],
            "loguniform" -> [low, high],
            "discrete_uniform" -> [low, high, q],
            "int" -> [low, high],
            "categorical" -> [choices]
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой `[user_id]`;
            если None, выбираются все пользователи из тестовой выборки
        :param items: список объектов, которые необходимо рекомендовать;
            спарк-датафрейм с колонкой `[item_id]`;
            если None, выбираются все объекты из тестовой выборки
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в `items`
        :param context: контекст, в котором нужно получить рекомендации
        :param n_trials: количество уникальных испытаний; должно быть от 1
            до значения параметра `optuna_max_n_trials`
        :param path: путь к директории, в которой сохраняются временные файлы
        :return: словарь оптимальных значений параметров для модели; ключ -
            название параметра (совпадают с параметрами модели,
            которые возвращает `get_params()`), значение - значение параметра
        """

    @abstractmethod
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
        Обучает модель с нуля при заданных параметрах `params` и формирует
        рекомендации для `users` и `items`.
        В качестве выборки для обучения используется весь лог, без деления.

        :param params: словарь значений параметров для модели; ключ -
            название параметра (должен совпадать с одним из параметров модели,
            которые возвращает `get_params()`), значение - значение параметра
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой `[user_id]`;
            если None, выбираются все пользователи из тестовой выборки
        :param items: список объектов, которые необходимо рекомендовать;
            спарк-датафрейм с колонкой `[item_id]`;
            если None, выбираются все объекты из тестовой выборки
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в `items`
        :param context: контекст, в котором нужно получить рекомендации
        :return: рекомендации, спарк-датафрейм с колонками
            `[user_id , item_id , context , relevance]`
        """
