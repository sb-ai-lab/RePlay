"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import collections
import logging
from typing import Any, Dict, List, Optional, Callable, Union

from optuna import Trial
from pyspark.sql import DataFrame
from functools import partial

from replay.experiment import Experiment
from replay.metrics.base_metric import Metric
from replay.models.base_rec import Recommender
from replay.utils import fallback

SplitData = collections.namedtuple(
    "SplitData", "train test users items user_features item_features"
)


# pylint: disable=too-few-public-methods
class ObjectiveWrapper:
    """
    Данный класс реализован в соответствии с
    `инструкцией <https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments>`_
    по интеграции произвольной библиотеки машинного обучения с ``optuna``.
    По факту представляет собой обёртку вокруг некоторой целевой функции (процедуры обучения модели),
    параметры которой (гипер-параметры модели) ``optuna`` подбирает.

    Вызов подсчета критерия происходит через ``__call__``,
    а все остальные аргументы передаются через ``__init__``.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes

    def __init__(
        self, objective_calculator: Callable[..., float], **kwargs: Any
    ):
        self.objective_calculator = objective_calculator
        self.kwargs = kwargs

    def __call__(self, trial: Trial) -> float:
        """
        Эта функция вызывается при вычислении критерия в переборе параметров с помощью ``optuna``.
        Сигнатура функции совапдает с той, что описана в документации ``optuna``.

        :param trial: текущее испытание
        :return: значение критерия, который оптимизируется
        """
        return self.objective_calculator(trial=trial, **self.kwargs)


def suggest_param_value(
    trial: Trial,
    param_name: str,
    param_bounds: List[Optional[Any]],
    default_params_data: Dict[str, Dict[str, Union[str, List[Any]]]],
) -> Union[str, float, int]:
    """
    Функция принимает границы поиска значения гиперпараметра, заданные пользователем, и
    список гиперпараметров модели, их типов и границ модели. Вызывает метод trial-а,
    соотвествующий типу параметра и возвращает сэмплированное значение.

    :param trial: optuna trial, текущий запуск поиска гиперпараметров
    :param param_name: имя гиперпараметра
    :param param_bounds: нижняя и верхняя граница поиска, список значений для категориального
    или пустой список, если нужно использовать границы поиска, определенные в модели
    :param default_params_data: список гиперпараметров, их типы и дефолтные границы/значения,
    определенные для модели
    :return: значение гиперпараметра
    """
    to_optuna_types_dict = {
        "uniform": trial.suggest_uniform,
        "int": trial.suggest_int,
        "loguniform": trial.suggest_loguniform,
        "loguniform_int": partial(trial.suggest_int, log=True),
    }

    if param_name not in default_params_data:
        raise ValueError(
            "Гиперпараметр {} не определен для выбранной модели".format(
                param_name
            )
        )
    param_type = default_params_data[param_name]["type"]
    param_args = (
        param_bounds
        if param_bounds
        else default_params_data[param_name]["args"]
    )
    if param_type == "categorical":
        return trial.suggest_categorical(param_name, param_args)

    if len(param_args) != 2:
        raise ValueError(
            """
        Гиперпараметр {} является числовым. Передайте верхнюю
        и нижнюю границы поиска в формате [lower, upper]""".format(
                param_name
            )
        )
    lower, upper = param_args

    return to_optuna_types_dict[param_type](param_name, low=lower, high=upper)


# pylint: disable=too-many-arguments
def scenario_objective_calculator(
    trial: Trial,
    search_space: Dict[str, List[Optional[Any]]],
    split_data: SplitData,
    recommender: Recommender,
    criterion: Metric,
    k: int,
    experiment: Optional[Experiment] = None,
    fallback_recs: Optional[DataFrame] = None,
) -> float:
    """
    Функция для вычисления значения критерия при выбранных гиперпараметрах.
    :param trial: optuna trial, текущий запуск поиска гиперпараметров
    :param search_space: пространство поиска гиперпарамтеров, определенное пользователем
    :param split_data: данные для обучения
    :param recommender: модель replay
    :param criterion: критерий оптимизации (метрика)
    :param k: число рекомендаций
    :param experiment: объект Experiment для логирования результатов
    :param fallback_recs: рекомендации, полученные с помощью fallback_model
    :return: значение оптимизируемого критерия
    """
    logger = logging.getLogger("replay")

    params_for_trial = dict()
    for param_name, param_data in search_space.items():
        params_for_trial[param_name] = suggest_param_value(
            # pylint: disable=protected-access
            trial,
            param_name,
            param_data,
            recommender._search_space,
        )

    recommender.set_params(**params_for_trial)
    logger.debug("-- Второй фит модели в оптимизации")
    # pylint: disable=protected-access
    recommender._fit_wrap(
        split_data.train,
        split_data.user_features,
        split_data.item_features,
        False,
    )
    logger.debug("-- Предикт модели в оптимизации")
    recs = recommender._predict_wrap(
        log=split_data.train,
        k=k,
        users=split_data.users,
        items=split_data.items,
        user_features=split_data.user_features,
        item_features=split_data.item_features,
    ).cache()
    if fallback_recs is not None:
        recs = fallback(recs, fallback_recs, k)
    logger.debug("-- Подсчет метрики в оптимизации")
    criterion_value = criterion(recs, split_data.test, k)
    if experiment is not None:
        experiment.add_result(f"{str(recommender)}{params_for_trial}", recs)
    logger.debug("%s=%.2f", criterion, criterion_value)
    return criterion_value
