"""
Класс, реализующий стэккинг моделей.
"""
import logging
from typing import List, Optional

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.wrapper import JavaEstimator
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql import functions as sf

from replay.constants import BASE_FIELDS, SCHEMA, PRED_SCHEMA
from replay.models.base_rec import Recommender
from replay.session_handler import State
from replay.splitters import k_folds
from replay.utils import get_top_k_recs


class Stack(Recommender):
    """Стэк базовых моделей возвращает свои скоры, которые используются регрессором как фичи."""

    def __init__(
        self,
        models: List[Recommender],
        top_model: Optional[JavaEstimator] = None,
        n_folds: Optional[int] = 5,
    ):
        """
        :param models: список инициализированных моделей
        :param top_model: инициализированный регрессор pyspark
        :param n_folds: количество фолдов для обучения регрессора
        """
        self.models = models
        State()
        if top_model is None:
            top_model = LinearRegression()
        self.top_model = top_model
        self.n_folds = n_folds
        self._logger = logging.getLogger("replay")

    # pylint: disable=too-many-locals
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        schema = StructType(
            BASE_FIELDS
            + [StructField(str(model), DoubleType()) for model in self.models]
        )
        top_train = State().session.createDataFrame(data=[], schema=schema)
        # pylint: disable=invalid-name
        df = log.withColumnRenamed("user_idx", "user_id").withColumnRenamed(
            "item_idx", "item_id"
        )
        for i, (train, test) in enumerate(k_folds(df, self.n_folds)):
            self._logger.info("Processing fold #%d", i)
            fold_train = State().session.createDataFrame(
                data=[], schema=SCHEMA
            )
            test_items = test.select("item_id").distinct()
            train_items = train.select("item_id").distinct()
            items_pos = test_items.join(train_items, on="item_id", how="inner")
            if items_pos.count() == 0:
                self._logger.info(
                    "Bad split, no positive examples, skipping..."
                )
                continue
            n_pos = (
                test.groupBy("user_id")
                .count()
                .agg({"count": "max"})
                .collect()[0][0]
            )
            items_neg = train_items.join(
                test_items, on="item_id", how="left_anti"
            )
            n_pos = min(
                n_pos,
                items_pos.select("item_id").distinct().count(),
                items_neg.select("item_id").distinct().count(),
            )
            for model in self.models:
                pos = model.fit_predict(
                    train,
                    k=n_pos,
                    items=items_pos.select("item_id").distinct(),
                )
                pos = pos.withColumn("label", lit(1.0))
                pos = pos.join(
                    test.select("user_id", "item_id"),
                    on=["user_id", "item_id"],
                    how="inner",
                )
                if pos.count() == 0:
                    self._logger.info(
                        "Couldn't produce positive examples for %s, skipping...",
                        str(model),
                    )
                    scores = State().session.createDataFrame(
                        data=[], schema=PRED_SCHEMA
                    )
                    fold_train = fold_train.join(
                        scores, on=["user_id", "item_id", "label"], how="outer"
                    )
                    continue
                neg = model.predict(
                    train,
                    k=n_pos,
                    items=items_neg.select("item_id").distinct(),
                )
                neg = neg.withColumn("label", lit(0.0))
                neg = get_top_k_recs(
                    neg,
                    pos.count() // pos.select("user_id").distinct().count(),
                )

                scores = pos.union(neg)
                scores = scores.withColumnRenamed("relevance", str(model))
                fold_train = fold_train.join(
                    scores, on=["user_id", "item_id", "label"], how="outer"
                )
            top_train = top_train.union(fold_train)

        top_train = top_train.na.drop()
        if top_train.count() == 0:
            raise ValueError("Couldn't produce training set")
        feature_cols = [str(model) for model in self.models]
        top_train = VectorAssembler(
            inputCols=feature_cols, outputCol="features",
        ).transform(top_train)
        # pylint: disable=attribute-defined-outside-init
        self.top_train = top_train
        self.model = self.top_model.fit(top_train)
        for model in self.models:
            model.fit(df)

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
        top = (
            State()
            .session.createDataFrame(data=[], schema=SCHEMA)
            .drop("label")
        )
        top = top.withColumnRenamed("user_id", "user_idx").withColumnRenamed(
            "item_id", "item_idx"
        )
        top = top.withColumn("user_idx", top["user_idx"].cast("integer"))
        top = top.withColumn("item_idx", top["item_idx"].cast("integer"))
        for model in self.models:
            # pylint: disable=protected-access
            scores = model._predict(
                log,
                k,
                users,
                items,
                user_features,
                item_features,
                filter_seen_items,
            )
            if filter_seen_items:
                scores = self._mark_seen_items(
                    scores, self._convert_index(log)
                )
            scores = scores.withColumn(
                "relevance",
                sf.when(scores["relevance"] < 0, None).otherwise(
                    scores["relevance"]
                ),
            )
            scores = scores.withColumnRenamed("relevance", str(model))
            top = top.join(scores, on=["user_idx", "item_idx"], how="outer")
        top = top.na.drop()
        feature_cols = [str(model) for model in self.models]
        top = VectorAssembler(
            inputCols=feature_cols, outputCol="features",
        ).transform(top)
        pred = self.model.transform(top).withColumnRenamed(
            "prediction", "relevance"
        )
        pred = pred.drop(*feature_cols, "features")
        return pred
