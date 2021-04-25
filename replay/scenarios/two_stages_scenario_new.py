# pylint: skip-file
from collections.abc import Iterable
from typing import Dict, Optional, Tuple, List, Union, Callable

import pyspark.sql.functions as sf
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, isnull, lit, when

from lightautoml.automl.presets.tabular_presets import TabularAutoML

# from lightautoml.dataset.roles import DatetimeRole, TextRole
from lightautoml.tasks import Task

# from lightautoml.utils.profiler import Profiler

from replay.constants import AnyDataFrame
from replay.experiment import Experiment

# from replay.metrics import HitRate, Metric
from replay.models import ALSWrap, RandomRec
from replay.models.base_rec import Recommender

# from replay.models.classifier_rec import ClassifierRec
from replay.scenarios import BaseScenario
from replay.session_handler import State
from replay.splitters import Splitter, UserSplitter
from replay.utils import get_log_info, horizontal_explode, join_or_return


def _create_cols_list(log: DataFrame, agg_col: str = "user_id") -> List:
    """
    Создание списка статистических признаков в зависимости от типа значений relevance (унарные или нет)
    и наличия времени оценки (timestamp).
    :param log: лог взаимодействий пользователей и объектов, спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
    :param agg_col: столбец, по которому будут строиться статистические признаки, user_id или item_id
    :return: список столбцов для передачи в pyspark agg
    """
    if agg_col != "user_id" and agg_col != "item_id":
        raise ValueError(
            "некорректный столбец для агрегации: {} ".format(agg_col)
        )
    prefix = agg_col[:1]

    aggregates = [
        # The log of the number of ratings
        sf.log(sf.count(sf.col("relevance"))).alias(
            "{}_log_ratings_count".format(prefix)
        )
    ]

    # timestamp presents
    if log.select(sf.countDistinct(sf.col("timestamp"))).collect()[0][0] > 1:
        aggregates.extend(
            [
                # The log of the distinct dates of ratings
                sf.log(sf.countDistinct(sf.col("timestamp"))).alias(
                    "{}_log_rating_dates_count".format(prefix)
                ),
                # First date of rating
                sf.min(sf.col("timestamp")).alias(
                    "{}_min_rating_date".format(prefix)
                ),
                # Last date of rating
                sf.max(sf.col("timestamp")).alias(
                    "{}_max_rating_date".format(prefix)
                ),
            ]
        )

    # non-unary interactions
    if log.select(sf.countDistinct(sf.col("relevance"))).collect()[0][0] > 1:
        # mean/std of ratings
        aggregates.extend(
            [
                sf.stddev(sf.col("relevance")).alias("{}_std".format(prefix)),
                sf.mean(sf.col("relevance")).alias("{}_mean".format(prefix)),
            ]
        )
        # median and min/max cleared from outliers
        for percentile in [0.05, 0.5, 0.95]:
            aggregates.append(
                sf.expr(
                    "percentile_approx({}, {})".format("relevance", percentile)
                ).alias("{}_quantile_{}".format(prefix, percentile))
            )

    return aggregates


def _add_cond_distr_features(
    cat_cols: List[str], log: DataFrame, features_df: DataFrame
) -> Dict[str, DataFrame]:
    """
    Подсчет популярности объектов в зависимости от значения категориальных признаков пользователей
    или, наоборот, популярности у пользователя объектов с теми или иными значениями категориальных признаков.
    Например, популярность фильма у пользователей данной возрастной группы. Если переданы признаки пользователей,
    результат будет содержать признаки объектов и наоборот.
    :param cat_cols: список категориальных признаков для подсчета популярности
    :param log: лог взаимодействий пользователей и объектов, спарк-датафрейм с колонками
        ``[user_id, item_id, timestamp, relevance]``
    :param features_df: спарк-датафрейм с признаками пользователей или объектов
    :return: словарь "имя категориального признака - датафрейм с вычисленными значениями популярности
        по id и значениям категориального признака"
    """
    if "item_id" in features_df.columns:
        join_col, agg_col = "item_id", "user_id"
    else:
        join_col, agg_col = "user_id", "item_id"

    join_col = "item_id" if agg_col == "user_id" else "user_id"
    conditional_dist = dict()
    log_with_features = log.join(features_df, on=join_col, how="left")
    count_by_agg_col_name = "count_by_{}".format(agg_col)
    count_by_agg_col = log_with_features.groupBy(agg_col).agg(
        sf.count("relevance").alias(count_by_agg_col_name)
    )
    for cat_col in cat_cols:
        col_name = "{}_pop_by_{}".format(agg_col[:4], cat_col)
        intermediate_df = log_with_features.groupBy(agg_col, cat_col).agg(
            sf.count("relevance").alias(col_name)
        )
        intermediate_df = intermediate_df.join(
            count_by_agg_col, on=agg_col, how="left"
        )
        conditional_dist[cat_col] = intermediate_df.withColumn(
            col_name, sf.col(col_name) / sf.col(count_by_agg_col_name)
        ).drop(count_by_agg_col_name)
        conditional_dist[cat_col].cache()
    return conditional_dist


class TwoStagesFeaturesProcessor:
    user_log_features = None
    item_log_features = None
    user_cond_dist_cat_features = None
    items_cond_dist_cat_features = None

    def __init__(
        self,
        log: DataFrame,
        first_level_train: DataFrame,
        second_level_train: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        user_cat_features_list: Optional[List] = None,
        item_cat_features_list: Optional[List] = None,
    ) -> None:
        """
        Подсчет признаков пользователей и объектов, основанные на логе.
        Признаки выбираются таким образом, чтобы корректно рассчитываться и для implicit,
        и для explicit feedback.
        :param log: лог взаимодействий пользователей и объектов, спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param first_level_train: лог взаимодействий пользователей и объектов для обучения моделей первого уровня
        Чтобы избежать переобучения статистические признаки рассчитываются на основании данного лога.
        :param second_level_train: лог взаимодействий пользователей и объектов для обучения модели второго уровня
        :param user_features: признаки пользователей, лог с обязательным столбцом ``user_id`` и столбцами с признаками
        :param item_features: признаки объектов, лог с обязательным столбцом `` item_id`` и столбцами с признаками
        :param user_cat_features_list: категориальные признаки пользователей, которые нужно использовать для построения признаков
            популярности объекта у пользователей в зависимости от значения категориального признака
            (например, популярность фильма у пользователей данной возрастной группы)
        :param item_cat_features_list: категориальные признаки объектов, которые нужно использовать для построения признаков
            популярности у пользователя объектов в зависимости от значения категориального признака
        """

        self.all_users = log.select("user_id").distinct().cache()
        self.all_items = log.select("item_id").distinct().cache()

        base_df = first_level_train

        user_aggs = _create_cols_list(base_df, agg_col="user_id")
        self.user_log_features = base_df.groupBy("user_id").agg(*user_aggs)

        item_aggs = _create_cols_list(base_df, agg_col="item_id")
        self.item_log_features = base_df.groupBy("item_id").agg(*item_aggs)

        # Average log number of ratings for the user items
        mean_log_rating_of_user_items = base_df.join(
            self.item_log_features.select("item_id", "i_log_ratings_count"),
            on="item_id",
            how="left",
        )
        mean_log_rating_of_user_items = mean_log_rating_of_user_items.groupBy(
            "user_id"
        ).agg(
            sf.mean("i_log_ratings_count").alias(
                "u_mean_log_items_ratings_count"
            )
        )
        self.user_log_features = self.user_log_features.join(
            mean_log_rating_of_user_items, on="user_id", how="left"
        )

        # Average log number of ratings for the item users
        mean_log_rating_of_item_users = base_df.join(
            self.user_log_features.select("user_id", "u_log_ratings_count"),
            on="user_id",
            how="left",
        )
        mean_log_rating_of_item_users = mean_log_rating_of_item_users.groupBy(
            "item_id"
        ).agg(
            sf.mean("u_log_ratings_count").alias(
                "i_mean_log_users_ratings_count"
            )
        )
        self.item_log_features = self.item_log_features.join(
            mean_log_rating_of_item_users, on="item_id", how="left"
        ).cache()

        # Abnormality: https://hal.inria.fr/hal-01254172/document
        abnormality_df = base_df.join(
            self.item_log_features.select("item_id", "i_mean", "i_std"),
            on="item_id",
            how="left",
        )
        abnormality_df = abnormality_df.withColumn(
            "abnormality", sf.abs(sf.col("relevance") - sf.col("i_mean"))
        )

        abnormality_aggs = [
            sf.mean(sf.col("abnormality")).alias("abnormality")
        ]

        # Abnormality CR: https://hal.inria.fr/hal-01254172/document
        max_std = self.item_log_features.select(sf.max("i_std")).collect()[0][
            0
        ]
        min_std = self.item_log_features.select(sf.min("i_std")).collect()[0][
            0
        ]

        if max_std - min_std != 0:
            abnormality_df = abnormality_df.withColumn(
                "controversy",
                1
                - (sf.col("i_std") - sf.lit(min_std))
                / (sf.lit(max_std - min_std)),
            )
            abnormality_df = abnormality_df.withColumn(
                "abnormalityCR",
                (sf.col("abnormality") * sf.col("controversy")) ** 2,
            )
            abnormality_aggs.append(
                sf.mean(sf.col("abnormalityCR")).alias("abnormalityCR")
            )

        abnormality_res = abnormality_df.groupBy("user_id").agg(
            *abnormality_aggs
        )
        self.user_log_features = self.user_log_features.join(
            abnormality_res, on="user_id", how="left"
        ).cache()

        # Mean rating distribution by the users' cat features
        if user_features is not None and user_cat_features_list is not None:
            self.item_cond_dist_cat_features = _add_cond_distr_features(
                user_cat_features_list, base_df, user_features
            )

        # Mean rating distribution by the items' cat features
        if item_features is not None and item_cat_features_list is not None:
            self.user_cond_dist_cat_features = _add_cond_distr_features(
                item_cat_features_list, base_df, item_features
            )

        self.fitted = True

    def __call__(self, log: DataFrame, step="train"):
        joined = (
            log.join(self.user_log_features, on="user_id", how="left")
            .join(self.item_log_features, on="item_id", how="left")
            .fillna(
                {
                    col_name: 0
                    for col_name in self.user_log_features.columns
                    + self.item_log_features.columns
                }
            )
        )

        if self.user_cond_dist_cat_features is not None:
            for key, value in self.user_cond_dist_cat_features.items():
                joined = joined.join(value, on=["user_id", key], how="left")
                joined = joined.fillna(
                    {col_name: 0 for col_name in value.columns}
                )

        if self.item_cond_dist_cat_features is not None:
            for key, value in self.item_cond_dist_cat_features.items():
                joined = joined.join(value, on=["item_id", key], how="left")
                joined = joined.fillna(
                    {col_name: 0 for col_name in value.columns}
                )

        joined = joined.withColumn(
            "u_log_ratings_count_diff",
            sf.col("u_log_ratings_count")
            - sf.col("i_mean_log_users_ratings_count"),
        ).withColumn(
            "i_log_ratings_count_diff",
            sf.col("i_log_ratings_count")
            - sf.col("u_mean_log_items_ratings_count"),
        )

        return joined

    def __del__(self):
        self.user_log_features.unpersist()
        self.item_log_features.unpersist()

        if self.user_cond_dist_cat_features is not None:
            for value in self.user_cond_dist_cat_features.values():
                value.unpersist()

        if self.item_cond_dist_cat_features is not None:
            for value in self.item_cond_dist_cat_features.values():
                value.unpersist()


# pylint: disable=too-many-instance-attributes
class TwoStagesScenario(BaseScenario):
    """
    Двухуровневый сценарий состоит из следующих этапов:
    train:
    * получить ``log`` взаимодействия и разбить его на first_level_train и second_level_train
    с помощью переданного splitter-а или дефолтного splitter, разбивающего лог для каждого пользователя 50/50
    * на ``first_stage_train`` обучить ``first_stage_models`` - модели первого уровня, которые могут предсказывать
    релевантность и генерировать дополнительные признаки пользователей и объектов (эмбеддинги)
    * сгенерировать негативные примеры для обучения модели второго уровня
        - как предсказания основной модели первого уровня, не релевантные для пользователя
        - случайным образом
    количество негативных примеров на 1 пользователя определяется параметром ``num_negatives``
    * дополнить датасет признаками:
        - получить предсказания моделей 1 уровня для позитивных взаимодействий из second_level_train и сгенерированных
    негативных примеров
        - дополнить полученный датасет признаками пользователей и объектов,
        - сгенерировать признаки взаимодействия для пар пользователь-айтем и статистические признаки
    * обучить ``TabularAutoML`` из библиотеки LightAutoML на полученном датасете с признаками

    inference:
    * получить ``log`` взаимодействия
    * сгенерировать объекты-кандидаты с помощью модели первого уровня для оценки моделью второго уровня
    количество кандидатов по дефолту равно числу негативных примеров при обучении и определяется параметром
     ``num_candidates``
    * дополнить полученный датасет признаками аналогично дополнению в train
    * получить top-k взаимодействий для каждого пользователя
    """

    can_predict_cold_users: bool = True
    can_predict_cold_items: bool = True

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        train_splitter: Splitter = UserSplitter(
            item_test_size=0.5, shuffle=True, seed=42
        ),
        first_level_models: Union[List[Recommender], Recommender] = ALSWrap(
            rank=128
        ),
        use_first_level_features: Union[List[bool], bool] = False,
        second_model_params: Optional[Union[Dict, str]] = None,
        second_model_config_path: Optional[str] = None,
        num_negatives: int = 100,
        negatives_type: str = "first_level",
        use_generated_features: bool = False,
        user_cat_features_list: Optional[List] = None,
        item_cat_features_list: Optional[List] = None,
        custom_features_processor: Callable = None,
        seed: int = 123,
    ) -> None:
        """
        Сборка двухуровневой рекомендательной архитектуры из блоков

        :param train_splitter: splitter для разбиения лога на ``first_level_train`` и ``second_level_train``.
            По умолчанию для каждого пользователя 50% объектов из лога, выбранные случайно (не по времени),
            используются для обучения модели первого уровня (first_level_train),
            а остальное - для обучения модели второго уровня (second_level_train).
        :param first_level_models: Модель или список инициализированных моделей RePlay, использующихся
            на первом этапе обучения. Для генерации кандидатов для переранжирования моделью второго уровня
            используется первая модель из списка. По умолчанию используется модель :ref:`ALS<als-rec>`.
        :param use_first_level_features: Флаг или список флагов, определяющих использование признаков,
            полученных моделью первого уровня (например, вектора пользователей и объектов из ALS,
            эмбеддинги пользователей из multVAE), для обучения модели второго уровня.
            Если bool, флаг применяется ко всем моделям, в случае передачи списка
            для каждой модели должно быть указано свое значение флага.
        :param second_model_params: Параметры TabularAutoML в виде многоуровневого dict
        :param second_model_config_path: Путь к конфиг-файлу для настройки TabularAutoML
        :param num_negatives: сколько объектов класса 0 будем генерировать для обучения
        :param negatives_type: каким образом генерировать негативные примеры для обучения модели второго уровня,
            случайно ``random`` или как наиболее релевантные предсказанные моделью первого уровня ``first-level``
        :param use_generated_features: нужно ли использовать автоматически сгенерированные
            по логу признаки для обучения модели второго уровня
        :param user_cat_features_list: категориальные признаки пользователей, которые нужно использовать для построения признаков
            популярности объекта у пользователей в зависимости от значения категориального признака
            (например, популярность фильма у пользователей данной возрастной группы)
        :param item_cat_features_list: категориальные признаки объектов, которые нужно использовать для построения признаков
            популярности у пользователя объектов в зависимости от значения категориального признака
        :param custom_features_processor: в двухуровневый сценарий можно передать свой callable-объект для
            генерации признаков для выбранных пар пользователь-объект во время обучения и inference
            на базе лога и признаков пользователей и объектов.
            Пример реализации - TwoLevelFeaturesProcessor.
        :param seed: random seed для обеспечения воспроизводимости результатов.
        """

        super().__init__(cold_model=None, threshold=0)
        self.train_splitter = train_splitter

        self.first_level_models = (
            first_level_models
            if isinstance(first_level_models, Iterable)
            else [first_level_models]
        )

        self.random_model = RandomRec(seed=seed)

        if isinstance(use_first_level_features, bool):
            self.use_first_level_models_feat = [
                use_first_level_features
            ] * len(self.first_level_models)
        else:
            if len(self.first_level_models) != len(use_first_level_features):
                raise ValueError(
                    "Для каждой модели из first_level_models укажите,"
                    "нужно ли использовать фичи, полученные моделью. Длина списков не совпадает."
                    "Количество моделей (first_level_models) равно {}, "
                    "количество флагов использования признаков (use_first_level_features) равно {}".format(
                        len(first_level_models), len(use_first_level_features)
                    )
                )

            self.use_first_level_models_feat = use_first_level_features

        if (
            second_model_config_path is not None
            or second_model_params is not None
        ):
            second_model_params = (
                dict() if second_model_params is None else second_model_params
            )
            self.second_stage_model = TabularAutoML(
                config_path=second_model_config_path, **second_model_params
            )
        else:
            self.second_stage_model = TabularAutoML(
                task=Task("binary"),
                reader_params={"cv": 5, "random_state": seed},
            )

        self.num_negatives = num_negatives
        if negatives_type not in ["random", "first_level"]:
            raise ValueError(
                "incorrect negatives_type, select random or first_level"
            )
        self.negatives_type = negatives_type

        self.use_generated_features = use_generated_features
        self.user_cat_features_list = user_cat_features_list
        self.item_cat_features_list = item_cat_features_list
        self.features_processor = (
            custom_features_processor if custom_features_processor else None
        )
        self.seed = seed

    def add_features(self, log, user_features, item_features, step="train"):
        self.logger.info("Feature enrichment: first-level features")
        # first-level pred and features
        full_second_level_train = log
        for idx, model in enumerate(self.first_level_models):
            current_pred = model.predict_for_pairs(
                full_second_level_train.select("user_id", "item_id"),
                user_features,
                item_features,
            ).withColumnRenamed("relevance", "{}_{}_rel".format(idx, model))
            full_second_level_train = full_second_level_train.join(
                current_pred, on=["user_id", "item_id"], how="left"
            )
            if self.use_first_level_models_feat[idx]:
                prefix = "{}_{}".format(idx, model)
                features = model.add_features(
                    full_second_level_train.select("user_id", "item_id"),
                    user_features,
                    item_features,
                    prefix,
                )
                full_second_level_train = full_second_level_train.join(
                    features, on=["user_id", "item_id"], how="left"
                )
        full_second_level_train = full_second_level_train.fillna(0).cache()

        self.logger.info(full_second_level_train.columns)

        # dataset features
        self.logger.info("Feature enrichment: dataset features")
        full_second_level_train = join_or_return(
            full_second_level_train,
            user_features,
            on_col="user_id",
            how="left",
        )
        full_second_level_train = join_or_return(
            full_second_level_train,
            item_features,
            on_col="item_id",
            how="left",
        )
        full_second_level_train.cache()

        self.logger.info(full_second_level_train.columns)

        # generated features
        self.logger.info("Feature enrichment: generated features")
        if self.use_generated_features:
            full_second_level_train = self.features_processor(
                full_second_level_train, step=step
            )
            self.logger.info(full_second_level_train.columns)

        return full_second_level_train

    def _split_data(self, log: DataFrame) -> Tuple[DataFrame, DataFrame]:
        first_level_train, second_level_train = self.train_splitter.split(log)
        State().logger.debug("Log info: %s", get_log_info(log))
        State().logger.debug(
            "first_level_train info: %s", get_log_info(first_level_train)
        )
        State().logger.debug(
            "second_level_train info: %s", get_log_info(second_level_train)
        )
        return first_level_train, second_level_train

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
        """

        # split
        # на каком уровне логирования пишем?
        self.logger.info("Data split")
        first_level_train, second_level_train = self._split_data(log)

        # нужны ли одинаковые индексы для всех моделей? место для оптимизации
        self.logger.info("First level models train")
        for base_model in self.first_level_models:
            base_model._fit_wrap(
                first_level_train, user_features, item_features, force_reindex
            )

        self.random_model.fit(log=log, force_reindex=force_reindex)

        self.logger.info("Negatives generation")
        negatives_source = (
            self.first_level_models[0]
            if self.negatives_type == "first_level"
            else self.random_model
        )
        negatives = negatives_source.predict(
            log,
            k=self.num_negatives,
            users=log.select("user_id").distinct(),
            items=log.select("item_id").distinct(),
            filter_seen_items=True,
        ).withColumn("relevance", sf.lit(0.0))

        full_second_level_train = (
            second_level_train.select("user_id", "item_id", "relevance")
            .withColumn("relevance", sf.lit(1))
            .unionByName(negatives)
            .cache()
        )
        full_second_level_train.groupBy("relevance").agg(
            sf.count(sf.col("relevance"))
        ).show()

        self.logger.info("Feature enrichment")
        if self.features_processor is None:
            self.features_processor = TwoStagesFeaturesProcessor(
                log,
                first_level_train=first_level_train,
                second_level_train=second_level_train,
                user_features=user_features,
                item_features=item_features,
                user_cat_features_list=self.user_cat_features_list,
                item_cat_features_list=self.item_cat_features_list,
            )

        full_second_level_train = self.add_features(
            log=full_second_level_train,
            user_features=user_features,
            item_features=item_features,
            step="train",
        )
        self.logger.info("Convert to pandas")
        full_second_level_train_pd = full_second_level_train.toPandas()
        full_second_level_train.unpersist()

        oof_pred = self.second_stage_model.fit_predict(
            full_second_level_train_pd, roles={"target": "relevance"}
        )
        print(
            self.second_stage_model.levels[0][0]
            .ml_algos[0]
            .get_features_score()
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
        if self.can_predict_cold_users:
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
