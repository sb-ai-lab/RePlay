"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

from pyspark.sql.types import (ArrayType, FloatType, IntegerType, StringType,
                               StructField, StructType, TimestampType)
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA
from sponge_bob_magic.models.classifier_rec import ClassifierRec


class ClassifierRecTestCase(PySparkTest):
    def setUp(self):
        self.model = ClassifierRec(seed=47)
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
                ("1", "1", datetime(2019, 1, 1), 1.0),
                ("1", "2", datetime(2019, 1, 1), 0.0)
            ],
            schema=LOG_SCHEMA
        )

    def test_get_params(self):
        model = ClassifierRec(seed=42)
        self.assertEqual(model.get_params(), {"seed": 42})

    def test_fit(self):
        self.model.fit(
            log=self.log,
            user_features=self.user_features,
            item_features=self.item_features
        )
        self.assertEqual(self.model.model.treeWeights, 20 * [1.0])

    def test_predict(self):
        self.model.fit(
            log=self.log,
            user_features=self.user_features,
            item_features=self.item_features
        )
        empty_prediction = self.model._predict(
            log=self.log,
            k=2,
            users=self.user_features.select("user_id"),
            items=self.item_features.select("item_id"),
            user_features=self.user_features,
            item_features=self.item_features,
            filter_seen_items=True
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
                ("1", "1", datetime(2019, 1, 1), 1.0, 1, 1, [1, 1]),
                ("1", "2", datetime(2019, 1, 1), 0.0, 1, 0, [1, 0])
            ],
            schema=StructType([
                StructField("user_id", StringType()),
                StructField("item_id", StringType()),
                StructField("timestamp", TimestampType()),
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
