"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

import numpy as np
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA
from sponge_bob_magic.models.classifier_rec import ClassifierRec


class ClassifierRecTestCase(PySparkTest):
    def setUp(self):
        self.model = ClassifierRec(seed=47)
        self.user_features = self.spark.createDataFrame(
            [("1", Vectors.dense([1.0, 2.0]))]
        ).toDF("user_id", "user_features")
        self.item_features = self.spark.createDataFrame(
            [("1", Vectors.dense([3.0, 4.0])), ("2", Vectors.dense([5.0, 6.0]))]
        ).toDF("item_id", "item_features")
        self.log = self.spark.createDataFrame([("1", "1", 1.0), ("1", "2", 0.0)],).toDF(
            "user_id", "item_id", "relevance"
        )

    def test_get_params(self):
        model = ClassifierRec(seed=42)
        self.assertEqual(model.get_params(), {"seed": 42})

    def test_fit(self):
        self.model.fit(
            log=self.log,
            user_features=self.user_features,
            item_features=self.item_features,
        )
        self.assertEqual(self.model.model.treeWeights, 20 * [1.0])

    def test_predict(self):
        self.model.fit(
            log=self.log,
            user_features=self.user_features,
            item_features=self.item_features,
        )
        empty_prediction = self.model.predict(
            log=self.log,
            k=2,
            users=self.user_features.select("user_id"),
            items=self.item_features.select("item_id"),
            user_features=self.user_features,
            item_features=self.item_features,
            filter_seen_items=True,
        )
        self.assertEqual(empty_prediction.count(), 2)

    def test_augment_data(self):
        augmented_data = self.model._augment_data(
            self.log, self.user_features, self.item_features
        )
        true_value = self.spark.createDataFrame(
            [
                (
                    "1",
                    "1",
                    1.0,
                    Vectors.dense([1.0, 2.0]),
                    Vectors.dense([3.0, 4.0]),
                    Vectors.dense([1.0, 2.0, 3.0, 4.0, 3.0, 8.0, 11.0]),
                ),
                (
                    "1",
                    "2",
                    0.0,
                    Vectors.dense([1.0, 2.0]),
                    Vectors.dense([5.0, 6.0]),
                    Vectors.dense([1.0, 2.0, 5.0, 6.0, 5.0, 12.0, 17.0]),
                ),
            ]
        ).toDF(
            "user_id",
            "item_id",
            "relevance",
            "user_features",
            "item_features",
            "features",
        )
        self.assertSparkDataFrameEqual(true_value, augmented_data)

    def test_pre_fit_raises(self):
        with self.assertRaises(ValueError):
            self.model._pre_fit(self.spark.createDataFrame([(1,)]).toDF("relevance"))
