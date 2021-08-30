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
    Treats recommendation task as a binary classification problem.
    """

    model: ClassificationModel
    augmented_data: DataFrame

    def __init__(
        self,
        spark_classifier: Optional[Classifier] = None,
        use_recs_value: Optional[bool] = False,
    ):
        """
        :param spark_classifier: Spark classification model object
        :param use_recs_value: flag to use ``recs`` value to produce recommendations
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
                "relevance values must be strictly 0 and 1 with both classes present"
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
        Add features to log.

        :param log: usual log dataframe
        :param user_features: user features dataframe
        :param item_features: item features dataframe
        :return: augmented log
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
                inputCols=feature_cols,
                outputCol="features",
            ).transform(raw_join)
        raise ValueError(
            "model must use at least one of: "
            "user features, item features, "
            "recommendations from the previous step"
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
            users.crossJoin(items),
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
        Get recommendations

        :param log: historical log of interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param k: length of recommendation lists, should be less that the total number of ``items``
        :param users: users to create recommendations for
            dataframe containing ``[user_id]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param user_features: user features
            ``[user_id , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_id , timestamp]`` + feature columns
        :return: recommendation dataframe
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
