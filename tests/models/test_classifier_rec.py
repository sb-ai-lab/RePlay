from pyspark.ml.classification import RandomForestClassifier

# pylint: disable-all
from pyspark.ml.linalg import Vectors
from tests.pyspark_testcase import PySparkTest

from replay.models.classifier_rec import ClassifierRec


class ClassifierRecTestCase(PySparkTest):
    def setUp(self):
        self.model = ClassifierRec(RandomForestClassifier(seed=47))
        self.user_features = self.spark.createDataFrame(
            [("1", 1.0, 2.0)]
        ).toDF("user_id", "user_feature_1", "user_feature_2")
        self.item_features = self.spark.createDataFrame(
            [("1", 3.0, 4.0), ("2", 5.0, 6.0)]
        ).toDF("item_id", "item_feature_1", "item_feature_2")
        self.log = self.spark.createDataFrame(
            [("1", "1", 1.0), ("1", "2", 0.0)],
        ).toDF("user_id", "item_id", "relevance")

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
        self.assertEqual(empty_prediction.count(), 0)
