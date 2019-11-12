"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

import numpy as np
from pyspark.sql.types import (ArrayType, FloatType, IntegerType, StringType,
                               StructField, StructType, TimestampType)
from sponge_bob_magic.constants import DEFAULT_CONTEXT, LOG_SCHEMA
from sponge_bob_magic.models.linear_recomennder import LinearRecommender
from tests.pyspark_testcase import PySparkTest


class LinearRecommenderTestCase(PySparkTest):
    def setUp(self):
        self.model = LinearRecommender(self.spark)
        self.user_features = self.spark.createDataFrame(
            [("1", datetime(2019, 1, 1), 1)],
            schema=StructType([
                StructField("user_id", StringType()),
                StructField("timestamp", TimestampType()),
                StructField("feature1", IntegerType())
            ])
        )
        self.item_features = self.spark.createDataFrame(
            [("1", datetime(2019, 1, 1), 1), (2, datetime(2019, 1, 1), 0)],
            schema=StructType([
                StructField("item_id", StringType()),
                StructField("timestamp", TimestampType()),
                StructField("feature2", IntegerType())
            ])
        )
        self.log = self.spark.createDataFrame(
            [
                ("1", "1", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("1", "2", datetime(2019, 1, 1), DEFAULT_CONTEXT, 0.0)
            ],
            schema=LOG_SCHEMA
        )

    def test_get_params(self):
        self.assertEqual(self.model.get_params(), dict())

    def test_fit_partial(self):
        self.model._fit_partial(
            log=self.log,
            user_features=self.user_features,
            item_features=self.item_features
        )
        self.assertEqual(-17.559285977683665, self.model._model.intercept)
        self.assertTrue(np.allclose(
            [0.0, 36.16632625479383],
            self.model._model.coefficients
        ))

    def test_predict(self):
        self.model.fit(
            log=self.log,
            user_features=self.user_features,
            item_features=self.item_features
        )
        prediction = self.model._predict(
            k=2,
            user_features=self.user_features,
            item_features=self.item_features,
            log=self.log,
            context=DEFAULT_CONTEXT,
            users=self.user_features.select("user_id"),
            items=self.item_features.select("item_id"),
            to_filter_seen_items=False
        )
        self.assertSparkDataFrameEqual(self.log.drop("timestamp"), prediction)
        empty_prediction = self.model._predict(
            k=2,
            user_features=self.user_features,
            item_features=self.item_features,
            log=self.log,
            context=DEFAULT_CONTEXT,
            users=self.user_features.select("user_id"),
            items=self.item_features.select("item_id"),
            to_filter_seen_items=True
        )
        self.assertEqual(
            sorted([(field.name, field.dataType) for field in
                    self.log.drop("timestamp").schema.fields],
                   key=lambda pair: pair[0]),
            sorted([(field.name, field.dataType) for field in
                    empty_prediction.schema.fields],
                   key=lambda pair: pair[0])
        )
        self.assertEqual(empty_prediction.count(), 0)

    def test_augment_data(self):
        augmented_data = self.model._augment_data(
            self.log, self.user_features, self.item_features
        )
        true_value = self.spark.createDataFrame(
            [
                ("1", "1", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0, 1, 1,
                 [1, 1]),
                ("1", "2", datetime(2019, 1, 1), DEFAULT_CONTEXT, 0.0, 1, 0,
                 [1, 0])
            ],
            schema=StructType([
                StructField("user_id", StringType()),
                StructField("item_id", StringType()),
                StructField("timestamp", TimestampType()),
                StructField("context", StringType()),
                StructField("relevance", FloatType()),
                StructField("feature1", IntegerType()),
                StructField("feature2", IntegerType()),
                StructField("features", ArrayType(IntegerType())),
            ])
        )
        self.assertSparkDataFrameEqual(
            true_value,
            augmented_data
        )
