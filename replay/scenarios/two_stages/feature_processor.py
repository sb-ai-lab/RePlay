from typing import Dict, Optional, Tuple, List

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType

from replay.data_preparator import CatFeaturesTransformer
from replay.session_handler import State
from replay.utils import join_or_return, ugly_join, unpersist_if_exists


class FirstLevelFeaturesProcessor:
    """Трансформация признаков для моделей первого уровня в двухуровневом сценарии"""

    cat_feat_transformer: Optional[CatFeaturesTransformer]
    cols_to_one_hot: Optional[List]
    cols_to_del: Optional[List]
    all_columns: Optional[List]

    def __init__(self, threshold: Optional[int] = 100):
        self.threshold = threshold
        self.fitted = False

    def fit(self, spark_df: Optional[DataFrame]) -> None:
        """
        Определение списка категориальных колонок для one-hot encoding.
        Нечисловые колонки с большим числом уникальных значений, чем threshold, будут удалены.
        Сохранение списка категорий для каждой из колонок.
        :param spark_df: spark-датафрейм, содержащий признаки пользователей / объектов
        """
        self.cat_feat_transformer = None
        if spark_df is None:
            return

        self.all_columns = sorted(spark_df.columns)
        self.cols_to_del = []
        idx_cols_set = {"user_idx", "item_idx", "user_id", "item_id"}

        spark_df_non_numeric = spark_df.select(
            *[
                col
                for col in spark_df.columns
                if (not isinstance(spark_df.schema[col].dataType, NumericType))
                and (col not in idx_cols_set)
            ]
        )
        if self.threshold is None:
            self.cols_to_one_hot = spark_df_non_numeric.columns
        else:
            counts_pd = (
                spark_df.agg(
                    *[
                        sf.approx_count_distinct(sf.col(c)).alias(c)
                        for c in spark_df_non_numeric.columns
                    ]
                )
                .toPandas()
                .T
            )
            self.cols_to_one_hot = (
                counts_pd[counts_pd[0] <= self.threshold]
            ).index.values
            self.cols_to_del = [
                col
                for col in spark_df_non_numeric.columns
                if col not in set(self.cols_to_one_hot)
            ]
            if self.cols_to_del:
                State().logger.warning(
                    "Колонки %s содержат более threshold уникальных "
                    "категориальных значений и будут удалены",
                    self.cols_to_del,
                )

        self.cat_feat_transformer = CatFeaturesTransformer(
            cat_cols_list=self.cols_to_one_hot, threshold=None
        )
        self.cat_feat_transformer.fit(spark_df.drop(*self.cols_to_del))

    def transform(self, spark_df: Optional[DataFrame]) -> Optional[DataFrame]:
        """
        Трансформация нечисловых признаков.
        Нечисловые колонки с большим числом уникальных значений, чем threshold,
        будут удалены иначе - преобразованы с использованием one-hot encoding.
        :param spark_df: spark-датафрейм, содержащий признаки пользователей / объектов
        :return: spark-датафрейм, содержащий числовые признаки пользователей / объектов
        """
        if spark_df is None or self.cat_feat_transformer is None:
            return None

        if sorted(spark_df.columns) != self.all_columns:
            raise ValueError(
                "Колонки датафрейма, переданного в fit, не совпадают с колонками, "
                "датафрейма, переданного в transform. "
                "Колонки в fit: %s,"
                "колонки в transform: %s"
                % (self.all_columns, sorted(spark_df.columns)),
            )

        return self.cat_feat_transformer.transform(
            spark_df.drop(*self.cols_to_del)
        )

    def fit_transform(self, spark_df: DataFrame) -> DataFrame:
        """
        Последовательное обучение и применение FirstLevelFeaturesProcessor.
        :param spark_df: исходный spark-датафрейм для one-hot encoding
        :return: результирующий spark-датафрейм
        """
        self.fit(spark_df)
        return self.transform(spark_df)


# pylint: disable=too-many-instance-attributes, too-many-arguments
class SecondLevelFeaturesProcessor:
    """
    Подсчет дополнительных признаков для модели второго уровня в двухуровневом сценарии
    """

    def __init__(
        self,
        use_log_features: bool = True,
        use_conditional_popularity: bool = True,
        use_cooccurrence: bool = False,
        user_id: str = "user_idx",
        item_id: str = "item_idx",
    ):
        self.use_log_features = use_log_features
        self.use_conditional_popularity = use_conditional_popularity
        self.use_cooccurrence = use_cooccurrence
        self.user_id = user_id
        self.item_id = item_id
        self.fitted: bool = False
        self.user_log_features_cached: Optional[DataFrame] = None
        self.item_log_features_cached: Optional[DataFrame] = None
        self.item_cond_dist_cat_feat_c: Optional[Dict[str, DataFrame]] = None
        self.user_cond_dist_cat_feat_c: Optional[Dict[str, DataFrame]] = None

    @staticmethod
    def _create_cols_list(log: DataFrame, agg_col: str = "user_idx") -> List:
        """
        Создание списка статистических признаков в зависимости от типа значений relevance
        (только единицы или различные значения) и наличия времени оценки (timestamp).
        :param log: лог взаимодействий пользователей и объектов, спарк-датафрейм с колонками
                ``[user_id(x), item_id(x), timestamp, relevance]``
        :param agg_col: столбец, по которому будут строиться статистические признаки,
            user_id(x) или item_id(x)
        :return: список столбцов для передачи в pyspark agg
        """
        prefix = agg_col[:1]

        aggregates = [
            # Логарифм числа взаимодействий
            sf.log(sf.count(sf.col("relevance"))).alias(
                "{}_log_ratings_count".format(prefix)
            )
        ]

        # В случае присутствия различных timestamp
        if (
            log.select(sf.countDistinct(sf.col("timestamp"))).collect()[0][0]
            > 1
        ):
            aggregates.extend(
                [
                    # Количество различных дат взаимодействия
                    sf.log(sf.countDistinct(sf.col("timestamp"))).alias(
                        "{}_log_rating_dates_count".format(prefix)
                    ),
                    # Минимальная дата взаимодействия
                    sf.min(sf.col("timestamp")).alias(
                        "{}_min_rating_date".format(prefix)
                    ),
                    # Максимальная дата взаимодействия
                    sf.max(sf.col("timestamp")).alias(
                        "{}_max_rating_date".format(prefix)
                    ),
                ]
            )

        # Для взаимодействий, характеризующихся различными значениями релевантности
        if (
            log.select(sf.countDistinct(sf.col("relevance"))).collect()[0][0]
            > 1
        ):
            # mean и std релевантности
            aggregates.extend(
                [
                    (
                        sf.when(
                            sf.stddev(sf.col("relevance")).isNull()
                            | sf.isnan(sf.stddev(sf.col("relevance"))),
                            0,
                        )
                        .otherwise(sf.stddev(sf.col("relevance")))
                        .alias("{}_std".format(prefix))
                    ),
                    sf.mean(sf.col("relevance")).alias(
                        "{}_mean".format(prefix)
                    ),
                ]
            )
            # медиана и 5-, 95-й перцентили релевантности
            for percentile in [0.05, 0.5, 0.95]:
                aggregates.append(
                    sf.expr(
                        "percentile_approx({}, {})".format(
                            "relevance", percentile
                        )
                    ).alias(
                        "{}_quantile_{}".format(prefix, str(percentile)[2:])
                    )
                )

        return aggregates

    def _calc_log_features(
        self, log: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        user_aggs = self._create_cols_list(log, agg_col=self.user_id)
        user_log_features = log.groupBy(self.user_id).agg(*user_aggs)

        item_aggs = self._create_cols_list(log, agg_col=self.item_id)
        item_log_features = log.groupBy(self.item_id).agg(*item_aggs)

        # Среднее лог-число взаимодействий у объектов, с которыми взаимодействовал пользователь
        mean_log_rating_of_user_items = log.join(
            item_log_features.select(self.item_id, "i_log_ratings_count"),
            on=self.item_id,
            how="left",
        )
        mean_log_rating_of_user_items = mean_log_rating_of_user_items.groupBy(
            self.user_id
        ).agg(
            sf.mean("i_log_ratings_count").alias(
                "u_mean_log_items_ratings_count"
            )
        )

        user_log_features = ugly_join(
            left=user_log_features,
            right=mean_log_rating_of_user_items,
            on_col_name=self.user_id,
            how="left",
        )

        # Среднее лог-число взаимодействий у пользователей, взаимодействовавших с объектом
        mean_log_rating_of_item_users = log.join(
            user_log_features.select(self.user_id, "u_log_ratings_count"),
            on=self.user_id,
            how="left",
        )
        mean_log_rating_of_item_users = mean_log_rating_of_item_users.groupBy(
            self.item_id
        ).agg(
            sf.mean("u_log_ratings_count").alias(
                "i_mean_log_users_ratings_count"
            )
        )

        item_log_features = ugly_join(
            left=item_log_features,
            right=mean_log_rating_of_item_users,
            on_col_name=self.item_id,
            how="left",
        ).cache()

        if "i_mean" in item_log_features.columns:
            # Abnormality: https://hal.inria.fr/hal-01254172/document
            abnormality_df = ugly_join(
                left=log,
                right=sf.broadcast(
                    item_log_features.select(self.item_id, "i_mean", "i_std")
                ),
                on_col_name=self.item_id,
                how="left",
            )
            abnormality_df = abnormality_df.withColumn(
                "abnormality", sf.abs(sf.col("relevance") - sf.col("i_mean"))
            )

            abnormality_aggs = [
                sf.mean(sf.col("abnormality")).alias("abnormality")
            ]

            # Abnormality CR: https://hal.inria.fr/hal-01254172/document
            max_std = item_log_features.select(sf.max("i_std")).collect()[0][0]
            min_std = item_log_features.select(sf.min("i_std")).collect()[0][0]
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

            abnormality_res = abnormality_df.groupBy(self.user_id).agg(
                *abnormality_aggs
            )
            user_log_features = user_log_features.join(
                sf.broadcast(abnormality_res), on=self.user_id, how="left"
            ).cache()

        return user_log_features, item_log_features

    def _add_cond_distr_feat(
        self, cat_cols: List[str], log: DataFrame, features_df: DataFrame
    ) -> Dict[str, DataFrame]:
        """
        Подсчет популярности объектов в зависимости от значения категориальных признаков пользователей
        или, наоборот, популярности у пользователя объектов с теми или иными значениями категориальных признаков.
        Например, популярность фильма у пользователей данной возрастной группы. Если переданы признаки пользователей,
        результат будет содержать признаки объектов и наоборот.
        :param cat_cols: список категориальных признаков для подсчета популярности
        :param log: лог взаимодействий пользователей и объектов, спарк-датафрейм с колонками
            ``[user_id(x), item_id(x), timestamp, relevance]``
        :param features_df: спарк-датафрейм с признаками пользователей или объектов
        :return: словарь "имя категориального признака - датафрейм с вычисленными значениями популярности
            по id и значениям категориального признака"
        """
        if self.item_id in features_df.columns:
            join_col, agg_col = self.item_id, self.user_id
        else:
            join_col, agg_col = self.user_id, self.item_id

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
                sf.broadcast(count_by_agg_col), on=agg_col, how="left"
            )
            conditional_dist[cat_col] = intermediate_df.withColumn(
                col_name, sf.col(col_name) / sf.col(count_by_agg_col_name)
            ).drop(count_by_agg_col_name)
            conditional_dist[cat_col].cache()

        return conditional_dist

    def fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        user_cat_features_list: Optional[List] = None,
        item_cat_features_list: Optional[List] = None,
    ) -> None:
        """
        Подсчет признаков пользователей и объектов, основанные на логе.
        Подсчет популярности в зависимости от значения категориальных признаков.
        Признаки выбираются таким образом, чтобы корректно рассчитываться и для implicit,
        и для explicit feedback.
        :param log: лог взаимодействий пользователей и объектов, спарк-датафрейм с колонками
            ``[user_id(x), item_id(x), timestamp, relevance]``
        :param user_features: признаки пользователей, лог с обязательным столбцом ``user_id(x)`` и столбцами с признаками
        :param item_features: признаки объектов, лог с обязательным столбцом ``item_id(x)`` и столбцами с признаками
        :param user_cat_features_list: категориальные признаки пользователей, которые нужно использовать для построения
            признаков популярности объекта у пользователей в зависимости от значения категориального признака
            (например, популярность фильма у пользователей данной возрастной группы)
        :param item_cat_features_list: категориальные признаки объектов, которые нужно использовать для построения признаков
            популярности у пользователя объектов в зависимости от значения категориального признака
        """
        log = log.cache()
        if self.use_cooccurrence:
            raise NotImplementedError("co-occurrence will be implemented soon")

        if self.use_log_features:
            (
                self.user_log_features_cached,
                self.item_log_features_cached,
            ) = self._calc_log_features(log)

        if self.use_conditional_popularity:
            # Популярность объектов для различных категорий пользователей
            if (
                user_features is not None
                and user_cat_features_list is not None
            ):
                self.item_cond_dist_cat_feat_c = self._add_cond_distr_feat(
                    user_cat_features_list, log, user_features
                )

            # Популярность у пользователей различных категорий объектов
            if (
                item_features is not None
                and item_cat_features_list is not None
            ):
                self.user_cond_dist_cat_feat_c = self._add_cond_distr_feat(
                    item_cat_features_list, log, item_features
                )

        self.fitted = True
        log.unpersist()

    def transform(
        self, log: DataFrame,
    ):
        """
        Обогащение лога сгенерированными признаками.
        :param log: пары пользователей и объектов и их признаки из датасета, спарк-датафрейм с колонками
            ``[user_id(x), item_id(x), ...]``, для которого нужно сгенерировать признаки
        :return: датафрейм, содержащий взаимодействия из лога и сгенерированные признаки
        """
        if not self.fitted:
            raise AttributeError("Вызовите fit перед использованием transform")
        joined = log

        if self.use_log_features:
            joined = (
                joined.join(
                    sf.broadcast(self.user_log_features_cached),
                    on=self.user_id,
                    how="left",
                )
                .join(
                    sf.broadcast(self.item_log_features_cached),
                    on=self.item_id,
                    how="left",
                )
                .fillna(
                    {
                        col_name: 0
                        for col_name in self.user_log_features_cached.columns
                        + self.item_log_features_cached.columns
                    }
                )
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

        if self.use_conditional_popularity:
            if self.user_cond_dist_cat_feat_c is not None:
                for (key, value,) in self.user_cond_dist_cat_feat_c.items():
                    joined = join_or_return(
                        joined,
                        sf.broadcast(value),
                        on=[self.user_id, key],
                        how="left",
                    )
                    joined = joined.fillna({"user_pop_by_" + key: 0})

            if self.item_cond_dist_cat_feat_c is not None:
                for (key, value,) in self.item_cond_dist_cat_feat_c.items():
                    joined = join_or_return(
                        joined,
                        sf.broadcast(value),
                        on=[self.item_id, key],
                        how="left",
                    )
                    joined = joined.fillna({"item_pop_by_" + key: 0})
        return joined

    def __del__(self):
        for log_feature in [
            self.user_log_features_cached,
            self.item_log_features_cached,
        ]:
            unpersist_if_exists(log_feature)

        for conditional_feature in [
            self.user_cond_dist_cat_feat_c,
            self.item_cond_dist_cat_feat_c,
        ]:
            if conditional_feature is not None:
                for value in conditional_feature.values():
                    unpersist_if_exists(value)
