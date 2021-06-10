from typing import Iterable, Optional, Union

import pyspark
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit, udf, when
from pyspark.sql.types import DoubleType

from replay.constants import AnyDataFrame
from replay.models.base_rec import HybridRecommender
from replay.utils import func_get, get_top_k_recs, convert2spark

# pylint: disable=ungrouped-imports
if pyspark.__version__.startswith("3.1"):
    from pyspark.ml.classification import (
        ClassificationModel,
        Classifier,
    )
elif pyspark.__version__.startswith("3.0"):
    from pyspark.ml.classification import (
        JavaClassificationModel as ClassificationModel,
        JavaClassifier as Classifier,
    )
else:
    from pyspark.ml.classification import (
        JavaClassificationModel as ClassificationModel,
        JavaEstimator as Classifier,
    )


class ClassifierRec(HybridRecommender):
    """
    Рекомендатель на основе классификатора.

    Получает на вход лог, в котором ``relevance`` принимает значения ``0`` и ``1``.
    Обучение строится следующим образом:

    * к логу присоединяются свойства пользователей и объектов (если есть)
    * свойства считаются фичами классификатора, а ``relevance`` --- таргетом
    * обучается классификатор, который умеет предсказывать ``relevance``

    В выдачу рекомендаций попадает top K объектов с наивысшим предсказанным скором от классификатора.
    """

    model: ClassificationModel
    augmented_data: DataFrame

    def __init__(
        self,
        spark_classifier: Optional[Classifier] = None,
        use_recs_value: Optional[bool] = False,
    ):
        """
        Инициализирует параметры модели.

        :param use_recs_value: использовать ли поле recs для рекомендаций
        :param spark_classifier: объект модели-классификатора на Spark
        """
        if spark_classifier is None:
            self.spark_classifier = RandomForestClassifier()
        else:
            self.spark_classifier = spark_classifier
        self.use_recs_value = use_recs_value

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        relevances = {
            row[0] for row in log.select("relevance").distinct().collect()
        }
        if relevances != {0, 1}:
            raise ValueError(
                "в логе должны быть relevance только 0 или 1"
                " и присутствовать значения обоих классов"
            )
        self.augmented_data = (
            self._augment_data(log, user_features, item_features)
            .withColumnRenamed("relevance", "label")
            .select("label", "features", "user_idx", "item_idx")
        )
        self.model = self.spark_classifier.fit(self.augmented_data)

    def _augment_data(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        """
        Обогащает лог фичами пользователей и объектов.

        :param log: лог в стандартном формате
        :param user_features: свойства пользователей в стандартном формате
        :param item_features: свойства объектов в стандартном формате
        :return: новый спарк-датафрейм, в котором к каждой строчке лога
            добавлены фичи соответствующих пользователя и объекта
        """
        feature_cols = ["recs"] if self.use_recs_value else []
        raw_join = log
        if user_features is not None:
            user_vectors = VectorAssembler(
                inputCols=user_features.drop("user_idx").columns,
                outputCol="user_features",
            ).transform(user_features)
            raw_join = raw_join.join(
                user_vectors.select("user_idx", "user_features"),
                on="user_idx",
                how="inner",
            )
            feature_cols += ["user_features"]
        if item_features is not None:
            item_vectors = VectorAssembler(
                inputCols=item_features.drop("item_idx").columns,
                outputCol="item_features",
            ).transform(item_features)
            raw_join = raw_join.join(
                item_vectors.select("item_idx", "item_features"),
                on="item_idx",
                how="inner",
            )
            feature_cols += ["item_features"]
        if feature_cols:
            return VectorAssembler(
                inputCols=feature_cols, outputCol="features",
            ).transform(raw_join)
        raise ValueError(
            "модель должна использовать хотя бы одно из: "
            "свойства пользователей, свойства объектов, "
            "рекомендации предыдущего шага при стекинге"
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
        data = self._augment_data(
            users.crossJoin(items), user_features, item_features,
        ).select("features", "item_idx", "user_idx")
        recs = self.model.transform(data).select(
            "user_idx",
            "item_idx",
            # pylint: disable=redundant-keyword-arg
            udf(func_get, DoubleType())("probability", lit(1)).alias(
                "relevance"
            ),
        )
        return recs

    def _rerank(
        self,
        log: DataFrame,
        users: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        data = self._augment_data(
            log.join(users, on="user_idx", how="inner").select(
                *(
                    ["item_idx", "user_idx"]
                    + (["recs"] if self.use_recs_value else [])
                )
            ),
            user_features,
            item_features,
        ).select("features", "item_idx", "user_idx")
        recs = self.model.transform(data).select(
            "user_idx",
            "item_idx",
            # pylint: disable=redundant-keyword-arg
            udf(func_get, DoubleType())("probability", lit(1)).alias(
                "relevance"
            ),
        )
        return recs

    def rerank(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[DataFrame, Iterable]] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
    ) -> DataFrame:
        """
        Выдача рекомендаций для пользователей.

        :param log: лог рекомендаций пользователей и объектов,
            спарк-датафрейм с колонками, который нужно переранжировать
            ``[user_id, item_id, timestamp, relevance]``
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в ``items``
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой ``[user_id]`` или
            ``array-like``;
            если ``None``, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то вызывается ошибка
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id , timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id , timestamp]`` и колонки с признаками
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]``
        """
        log = convert2spark(log)
        user_type = log.schema["user_id"].dataType
        item_type = log.schema["item_id"].dataType
        user_features = convert2spark(user_features)
        user_features = self._convert_index(user_features)
        item_features = convert2spark(item_features)
        item_features = self._convert_index(item_features)
        users = self._convert_index(users)
        log = self._convert_index(log)
        users = self._get_ids(users or log, "user_idx")

        recs = self._rerank(log, users, user_features, item_features)
        recs = self._convert_back(recs, user_type, item_type).select(
            "user_id", "item_id", "relevance"
        )
        recs = get_top_k_recs(recs, k)
        recs = recs.withColumn(
            "relevance",
            when(recs["relevance"] < 0, 0).otherwise(recs["relevance"]),
        )
        return recs
