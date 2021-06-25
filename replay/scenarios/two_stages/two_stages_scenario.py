import copy
from collections.abc import Iterable
from typing import Dict, Optional, Tuple, List, Union, Any

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

from replay.constants import AnyDataFrame
from replay.metrics import Metric, Precision
from replay.models import ALSWrap, RandomRec, PopRec
from replay.models.base_rec import BaseRecommender, HybridRecommender
from replay.scenarios.two_stages.feature_processor import (
    TwoStagesFeaturesProcessor,
)

from replay.session_handler import State
from replay.splitters import Splitter, UserSplitter
from replay.utils import (
    fallback,
    get_log_info,
    get_top_k_recs,
    get_first_level_model_features,
    join_or_return,
    convert2spark,
)


# pylint: disable=too-many-instance-attributes
class TwoStagesScenario(HybridRecommender):
    """
    Двухуровневый сценарий состоит из следующих этапов:

    *train*:

    1) получить ``log`` взаимодействия и разбить его на first_level_train и second_level_train
       с помощью переданного splitter-а или дефолтного splitter, разбивающего лог для каждого пользователя 50/50
    2) на ``first_stage_train`` обучить ``first_stage_models`` - модели первого уровня, которые могут предсказывать
       релевантность и генерировать дополнительные признаки пользователей и объектов (эмбеддинги)
    3) сгенерировать негативные примеры для обучения модели второго уровня:

       - как предсказания основной модели первого уровня, не релевантные для пользователя
       - случайным образом

       количество негативных примеров на 1 пользователя определяется параметром ``num_negatives``
    4) дополнить датасет признаками:

       - получить предсказания моделей 1 уровня для позитивных взаимодействий из second_level_train и сгенерированных
         негативных примеров
       - дополнить полученный датасет признаками пользователей и объектов,
       - сгенерировать признаки взаимодействия для пар пользователь-айтем и статистические признаки

    5) обучить ``TabularAutoML`` из библиотеки LightAutoML на полученном датасете с признаками

    *inference*:

    1) получить ``log`` взаимодействия
    2) сгенерировать объекты-кандидаты с помощью модели первого уровня для оценки моделью второго уровня
       количество кандидатов по дефолту равно числу негативных примеров при обучении и определяется параметром
       ``num_candidates``
    3) дополнить полученный датасет признаками аналогично дополнению в train
    4) получить top-k взаимодействий для каждого пользователя

    """

    can_predict_cold_users: bool = True
    can_predict_cold_items: bool = True

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        train_splitter: Splitter = UserSplitter(
            item_test_size=0.5, shuffle=True, seed=42
        ),
        first_level_models: Union[
            List[BaseRecommender], BaseRecommender
        ] = ALSWrap(rank=128),
        fallback_model: Optional[BaseRecommender] = PopRec(),
        use_first_level_features: Union[List[bool], bool] = False,
        second_model_params: Optional[Union[Dict, str]] = None,
        second_model_config_path: Optional[str] = None,
        num_negatives: int = 100,
        negatives_type: str = "first_level",
        use_generated_features: bool = False,
        user_cat_features_list: Optional[List] = None,
        item_cat_features_list: Optional[List] = None,
        custom_features_processor: TwoStagesFeaturesProcessor = None,
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
        :param fallback_model: Модель для дополнения списка рекомендаций, полученных моделью первого уровня,
            в случае, если моделью первого уровня получены рекомендации не для всех пользователей или
            получено недостаточное число объектов
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
        :param user_cat_features_list: категориальные признаки пользователей,
            которые нужно использовать для построения признаков
            популярности объекта у пользователей в зависимости от значения категориального признака
            (например, популярность фильма у пользователей данной возрастной группы)
        :param item_cat_features_list: категориальные признаки объектов,
            которые нужно использовать для построения признаков
            популярности у пользователя объектов в зависимости от значения категориального признака
        :param custom_features_processor: в двухуровневый сценарий можно передать свой объект,
            наследующийся от TwoStagesFeaturesProcessor для
            генерации признаков для выбранных пар пользователь-объект во время обучения и inference
            на базе лога и признаков пользователей и объектов.
            Пример реализации - TwoLevelFeaturesProcessor.
        :param seed: random seed для обеспечения воспроизводимости результатов.
        """
        # разбиение данных
        super().__init__(cold_model=None, threshold=0)
        self.train_splitter = train_splitter
        self.cached_list = []

        # модели первого уровня
        self.first_level_models = (
            first_level_models
            if isinstance(first_level_models, Iterable)
            else [first_level_models]
        )

        self.first_level_item_indexer_len = 0
        self.first_level_user_indexer_len = 0

        self.random_model = RandomRec(seed=seed)
        self.fallback_model = fallback_model

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

        # модель второго уровня
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
            # CHECK! спросить про параметры у Антона или Саши
            self.second_stage_model = TabularAutoML(
                task=Task("binary"),
                reader_params={"cv": 5, "random_state": seed},
            )

        # генерация отрицательных примеров
        self.num_negatives = num_negatives
        if negatives_type not in ["random", "first_level"]:
            raise ValueError(
                "incorrect negatives_type, select random or first_level"
            )
        self.negatives_type = negatives_type

        # добавление признаков
        self.use_generated_features = use_generated_features
        self.user_cat_features_list = user_cat_features_list
        self.item_cat_features_list = item_cat_features_list
        self.features_processor = (
            custom_features_processor
            if custom_features_processor
            else TwoStagesFeaturesProcessor()
        )
        self.seed = seed

    def add_features(
        self,
        log_to_add_features: DataFrame,
        log_used_in_predict: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
    ) -> DataFrame:
        """
        Дополнение признаками для передачи в модель второго уровня.
        Датафрейм дополняется признаками:
            - релевантность из моделей первого уровня
            - признаки пользователей и объектов из моделей первого уровня
            - признаки датасета
            - признаки, сгенерированные FeatureProcessor на основе лога
                и выбранных категориальных признаков

        :param log_to_add_features: лог взаимодействий пользователей и объектов,
            для которого нужно получить признаки
            спарк-датафрейм с колонками
            ``[user_idx, item_idx, timestamp, relevance]``
        :param log_used_in_predict: лог взаимодействий пользователей и объектов,
            использующийся в predict моделей первого уровня
            спарк-датафрейм с колонками
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонкой
            ``[user_idx]`` и колонками с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонкой
            ``[item_idx]`` и колонками с признаками
        :return: лог, обогащенный признаками
        """
        self.logger.info(
            "Генерация признаков: релевантность и признаки из моделей первого уровня"
        )
        # first-level pred and features
        full_second_level_train = log_to_add_features
        for idx, model in enumerate(self.first_level_models):
            current_pred = self._predict_pairs_with_first_level_model(
                model=model,
                log=log_used_in_predict,
                pairs=full_second_level_train.select("user_idx", "item_idx"),
                user_features=user_features,
                item_features=item_features,
            ).withColumnRenamed("relevance", "rel_{}_{}".format(idx, model))
            full_second_level_train = full_second_level_train.join(
                current_pred, on=["user_idx", "item_idx"], how="left"
            )

            if self.use_first_level_models_feat[idx]:
                # prefix = "{}_{}".format(idx, model)
                # TO DO: после merge пулреквеста с новой сигнатурой get_first_level_model_features обновить код
                features = get_first_level_model_features(
                    model=model,
                    pairs=full_second_level_train.select(
                        "user_idx", "item_idx"
                    ),
                )
                full_second_level_train = full_second_level_train.join(
                    features, on=["user_idx", "item_idx"], how="left"
                )
        full_second_level_train = full_second_level_train.fillna(0).cache()
        self.logger.info(
            "Колонки после добавления признаков первого уровня: %s",
            " ".join(full_second_level_train.columns),
        )

        self.logger.info(
            "Генерация признаков: добавление признаков из датасета"
        )
        full_second_level_train = join_or_return(
            full_second_level_train, user_features, on="user_idx", how="left",
        )
        full_second_level_train = join_or_return(
            full_second_level_train, item_features, on="item_idx", how="left",
        )
        full_second_level_train.cache()
        self.cached_list.append(full_second_level_train)

        if self.use_generated_features:
            if not self.features_processor.fitted:
                self.features_processor.fit(
                    log=log_used_in_predict,
                    user_features=user_features,
                    item_features=item_features,
                    user_cat_features_list=self.user_cat_features_list,
                    item_cat_features_list=self.item_cat_features_list,
                )
            self.logger.info(
                "Генерация признаков: добавление сгенерированных признаков"
            )
            full_second_level_train = self.features_processor.transform(
                log=full_second_level_train,
                user_features=user_features,
                item_features=item_features,
            )
        self.logger.info(
            "Колонки после добавления признаков из датасета: %s",
            " ".join(full_second_level_train.columns),
        )

        return full_second_level_train

    def _split_data(self, log: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Печать статистик разбиения лога"""
        first_level_train, second_level_train = self.train_splitter.split(log)
        State().logger.debug("Log info: %s", get_log_info(log))
        State().logger.debug(
            "first_level_train info: %s", get_log_info(first_level_train)
        )
        State().logger.debug(
            "second_level_train info: %s", get_log_info(second_level_train)
        )
        return first_level_train, second_level_train

    def _fit_wrap(
        self,
        log: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        force_reindex: bool = True,
    ) -> None:
        # разбиение данных
        log, user_features, item_features = [
            convert2spark(df) for df in [log, user_features, item_features]
        ]
        self._fit(log, user_features, item_features)

    @staticmethod
    def _filter_or_return(dataframe, condition):
        if dataframe is None:
            return dataframe
        return dataframe.filter(condition)

    def _predict_with_first_level_model(
        self,
        model,
        log,
        k,
        users,
        items,
        user_features,
        item_features,
        log_to_filter,
    ):
        """
        Фильтрация объектов и пользователей зависимости от флагов can_predict_cold_items,
        can_predict_cold_users, predict моделью и фильтрация top-k наиболее релевантных результатов.
        """
        if not model.can_predict_cold_items:
            log, items, item_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("item_idx")
                    < self.first_level_item_indexer_len,
                )
                for df in [log, items, item_features]
            ]
        if not model.can_predict_cold_users:
            log, users, user_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("user_idx")
                    < self.first_level_user_indexer_len,
                )
                for df in [log, users, user_features]
            ]

        max_positives_to_filter = min(
            [
                log_to_filter.groupBy("user_idx")
                .agg(sf.count("item_idx").alias("num_positives"))
                .select(sf.max("num_positives"))
                .collect()[0][0],
                log.select("item_idx").distinct().count() - k,
                items.select("item_idx").distinct().count() - k,
            ]
        )

        print("max_positives_second_level", max_positives_to_filter)
        negatives = model._predict(
            log,
            k=k + max_positives_to_filter,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=False,
        )

        # TO DO: это неоптимально, можно попробовать для каждого пользователя определять
        # свое k до фильтрации просмотренных,
        # фильтровать top-k, а потом исключать просмотренных,
        # чтобы сделать anti-join не таким объемным
        negatives = negatives.join(
            log_to_filter.select("user_idx", "item_idx"),
            on=["user_idx", "item_idx"],
            how="anti",
        ).drop("user", "item")

        return get_top_k_recs(negatives, k)

    def _predict_pairs_with_first_level_model(
        self, model, log, pairs, user_features, item_features
    ):
        if not model.can_predict_cold_items:
            log, pairs, item_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("item_idx")
                    < self.first_level_item_indexer_len,
                )
                for df in [log, pairs, item_features]
            ]
        if not model.can_predict_cold_users:
            log, pairs, user_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("user_idx")
                    < self.first_level_user_indexer_len,
                )
                for df in [log, pairs, user_features]
            ]

        return model._predict_pairs(
            pairs=pairs,
            log=log,
            user_features=user_features,
            item_features=item_features,
        )

    # pylint: disable=too-many-locals
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        Обучает модель на логе и признаках пользователей и объектов.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_idx, timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id, timestamp]`` и колонки с признаками
        :param force_reindex: обязательно создавать
            индексы, даже если они были созданы ранее
        """

        self.cached_list = []

        self.logger.info("Разбиение данных")
        first_level_train, second_level_train = self._split_data(log)
        self.logger.info("Индексирование пользователей и объектов")
        self._create_indexers(first_level_train, None, None)

        # индексы для фильтрации при передачи в модели первого уровня
        self.first_level_item_indexer_len = len(self.item_indexer.labels)
        self.first_level_user_indexer_len = len(self.user_indexer.labels)

        # для моделей 1 уровня используются копии индексеров, обученных на логе для 1 уровня
        # TO DO! сейчас наличие индексеров обязательно для корректной работы некоторых моделей, например, lightFM,
        # но, возможно, надо убрать эту привязку
        for model in self.first_level_models:
            model.user_indexer = copy.deepcopy(self.user_indexer)
            model.item_indexer = copy.deepcopy(self.item_indexer)
            model.inv_user_indexer = copy.deepcopy(self.inv_user_indexer)
            model.inv_item_indexer = copy.deepcopy(self.inv_item_indexer)

        # конвертация с обновлением индексеров
        log, first_level_train, second_level_train = [
            self._convert_index(df).cache()
            for df in [log, first_level_train, second_level_train]
        ]
        self.cached_list.extend([log, first_level_train, second_level_train])

        if user_features is not None:
            user_features = self._convert_index(user_features).cache()
            self.cached_list.append(user_features)

        if item_features is not None:
            item_features = self._convert_index(item_features).cache()
            self.cached_list.append(item_features)

        # TO DO: добавить отдельную предобработку признаков для моделей первого уровня, например:
        # one-hot для категориальных признаков с небольшим числом значений,
        # удаление остальных категориальных признаков,
        # обработка листов тегов (например, жанров фильма)
        # для модели второго уровня признаки используются в исходном виде
        for base_model in [
            *self.first_level_models,
            self.random_model,
            self.fallback_model,
        ]:
            base_model._fit(
                log=first_level_train,
                user_features=user_features.filter(
                    sf.col("user_idx") < self.first_level_user_indexer_len
                ),
                item_features=item_features.filter(
                    sf.col("item_idx") < self.first_level_item_indexer_len
                ),
            )

        self.logger.info(
            "Генерация негативных примеров для обучения модели второго уровня"
        )
        negatives_source = (
            self.first_level_models[0]
            if self.negatives_type == "first_level"
            else self.random_model
        )

        negatives = self._predict_with_first_level_model(
            model=negatives_source,
            log=first_level_train,
            k=self.num_negatives,
            users=log.select("user_idx").distinct(),
            items=log.select("item_idx").distinct(),
            user_features=user_features,
            item_features=item_features,
            log_to_filter=log,
        ).withColumn("relevance", sf.lit(0.0))

        if self.fallback_model is not None:
            negatives_fallback = self._predict_with_first_level_model(
                model=self.fallback_model,
                log=first_level_train,
                k=self.num_negatives,
                users=log.select("user_idx").distinct(),
                items=log.select("item_idx").distinct(),
                user_features=user_features,
                item_features=item_features,
                log_to_filter=log,
            ).withColumn("relevance", sf.lit(0.0))

            negatives = fallback(
                base=negatives,
                fill=negatives_fallback,
                k=self.num_negatives,
                id_type="idx",
            )

        self.logger.info(
            "Формирование датасета для обучения модели второго уровня"
        )
        full_second_level_train = (
            second_level_train.select("user_idx", "item_idx", "relevance")
            .withColumn("relevance", sf.lit(1))
            .unionByName(negatives)
            .cache()
        )

        self.cached_list.append(full_second_level_train)

        dataset_class_sizes = (
            full_second_level_train.groupBy("relevance")
            .agg(sf.count(sf.col("relevance")).alias("count_for_class"))
            .toPandas()
        )

        self.logger.info("В train для модели второго уровня:")
        for row_num in range(1):
            self.logger.info(
                "\t%s объектов класса %s",
                dataset_class_sizes.loc[row_num, "count_for_class"],
                dataset_class_sizes.loc[row_num, "relevance"],
            )

        self.features_processor.fit(
            log=first_level_train,
            user_features=user_features,
            item_features=item_features,
            user_cat_features_list=self.user_cat_features_list,
            item_cat_features_list=self.item_cat_features_list,
        )

        self.logger.info("Дополнение train модели второго уровня признаками")
        full_second_level_train = self.add_features(
            log_to_add_features=full_second_level_train,
            log_used_in_predict=first_level_train,
            user_features=user_features,
            item_features=item_features,
        )

        full_second_level_train = full_second_level_train.drop(
            "user_idx", "item_idx"
        )
        self.logger.info("Конвертация в pandas")
        full_second_level_train.cache()
        full_second_level_train_pd = full_second_level_train.toPandas()
        full_second_level_train.unpersist()
        for dataframe in self.cached_list:
            dataframe.unpersist()

        self.second_stage_model.fit_predict(
            full_second_level_train_pd, roles={"target": "relevance"}
        )

        self.logger.info("Завершено обучение модели второго уровня")
        print(
            self.second_stage_model.levels[0][0]
            .ml_algos[0]
            .get_features_score()[:20]
        )

    # pylint: disable=too-many-arguments
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
        Выдача рекомендаций для пользователей.

        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в ``items``
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой ``[user_idx]`` или ``array-like``;
            если ``None``, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то вызывается ошибка
        :param items: список объектов, которые необходимо рекомендовать;
            спарк-датафрейм с колонкой ``[item_idx]`` или ``array-like``;
            если ``None``, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в ``relevance`` в рекомендациях к ним будет стоять ``0``
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонкой
            ``[user_idx]`` и колонками с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонкой
            ``[item_idx]`` и колонками с признаками
        :param filter_seen_items: если True, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_idx, item_idx, relevance]``
        """
        State().logger.debug(msg="Генерация кандидатов для переранжирования")
        candidates = self._predict_with_first_level_model(
            model=self.first_level_models[0],
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            log_to_filter=log,
        )

        if self.fallback_model is not None:
            fallback_candidates = self._predict_with_first_level_model(
                model=self.fallback_model,
                log=log,
                k=k,
                users=users,
                items=items,
                user_features=user_features,
                item_features=item_features,
                log_to_filter=log,
            )

            candidates = fallback(
                base=candidates,
                fill=fallback_candidates,
                k=self.num_negatives,
                id_type="idx",
            )

        self.logger.info("Дополнение датасета кандидатов признаками")
        candidates_features = self.add_features(
            log_to_add_features=candidates,
            log_used_in_predict=log,
            user_features=user_features,
            item_features=item_features,
        )
        candidates_features.cache()
        self.logger.info(
            "Сгенерировано %s кандидатов для %s пользователей",
            candidates_features.count(),
            candidates_features.select("user_id").distinct().count(),
        )
        candidates_features_pd = candidates_features.toPandas()
        candidates_features.unpersist()
        candidates_ids = candidates_features_pd[
            ["user_idx", "item_idx", "relevance"]
        ]
        candidates_features_pd.drop(
            columns=["user_idx", "item_idx"], inplace=True
        )

        self.logger.info("Начато переранжирование моделью второго уровня")
        candidates_pred = self.second_stage_model.predict(
            candidates_features_pd
        )
        candidates_ids["relevance"] = candidates_pred.data[:, 0]
        self.logger.info(
            "%s candidates rated for %s users",
            candidates_ids.shape[0],
            candidates.select("user_idx").distinct().count(),
        )

        second_level_res = convert2spark(candidates_ids)

        self.logger.info("Выбор top-k")
        return get_top_k_recs(recs=second_level_res, k=k, id_type="idx")

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
            спарк-датафрейм с колонкой
            ``[user_id]`` и колонками с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонкой
            ``[item_id]`` и колонками с признаками
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

    @staticmethod
    def _optimize_one_model(
        model: BaseRecommender,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        criterion: Metric = Precision(),
        k: int = 10,
        budget: int = 10,
    ):
        """
        Поиск оптимальных параметров для одной модели
        """
        params = model.optimize(
            train,
            test,
            user_features,
            item_features,
            param_grid,
            criterion,
            k,
            budget,
        )
        model.set_params(**params)
        return params

    # pylint: disable=too-many-arguments, too-many-locals
    def optimize(
        self,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_grid: Optional[List[Dict[str, List[Any]]]] = None,
        criterion: Metric = Precision(),
        k: int = 10,
        budget: int = 10,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Подбирает лучшие гиперпараметры  моделей первого уровня с помощью optuna.

        :param train: лог взаимодействий пользователей и объектов для обучения,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
        :param test: лог взаимодействий пользователей и объектов для проверки качества,
            спарк-датафрейм с колонками
            ``[user_id, item_id, timestamp, relevance]``
            :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id , timestamp]`` и колонки с признаками
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонкой
            ``[user_id]`` и колонками с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонкой
            ``[item_id]`` и колонками с признаками
        :param param_grid: лист, содержащий сетки параметров для каждой из моделей первого уровня.
            Для использования дефолтной сетки передайте None в соответствующем элементе листа.
            Чтобы не искать параметры для какой-то из моделей,
            передайте пустой словарь в соответствующем элементе листа.
            Сетка для модели задается словарем, где ключ ---
            название параметра, значение --- границы возможных значений.
            ``{param: [low, high]}``.
        :param criterion: метрика, которая будет оптимизироваться
        :param k: количество рекомендаций для каждого пользователя
        :param budget: количество попыток поиска лучших гиперпараметров для каждой из моделей
        :return: лист словарей оптимальных параметров для моделей первого уровня и словарь для fallback-модели.
            В случае, если параметры для какой-то из моделей не подбирались, для нее возвращается None.
        """
        number_of_models = len(self.first_level_models)
        if self.fallback_model is not None:
            number_of_models += 1
        if number_of_models != len(param_grid):
            raise ValueError(
                "Передайте сетку параметров или None для каждой из моделей первого уровня и fallback-модели"
            )

        params_found = []
        for i, model in enumerate(self.first_level_models):
            if param_grid[i] is None or (
                isinstance(param_grid[i], dict) and param_grid[i]
            ):
                self.logger.info(
                    "Оптимизируем модель первого уровня номер %s, %s",
                    i,
                    model.__str__(),
                )
                params_found.append(
                    self._optimize_one_model(
                        model=model,
                        train=train,
                        test=test,
                        user_features=user_features,
                        item_features=item_features,
                        param_grid=param_grid[i],
                        criterion=criterion,
                    )
                )
            else:
                params_found.append(None)

        if self.fallback_model is None or (
            isinstance(param_grid[-1], dict) and not param_grid[-1]
        ):
            return params_found, None

        self.logger.info("Оптимизируем fallback-модель")
        fallback_params = self._optimize_one_model(
            model=self.fallback_model,
            train=train,
            test=test,
            user_features=user_features,
            item_features=item_features,
            param_grid=param_grid[-1],
            criterion=criterion,
        )
        return params_found, fallback_params
