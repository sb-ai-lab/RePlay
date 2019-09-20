import unittest

from sponge_bob_magic.models.popular_recomennder import PopularRecommender
from pyspark_testcase import PySparkTest


class PopularRecommenderTestCase(PySparkTest):
    def setUp(self):
        self.model = PopularRecommender(self.spark)

    def test_popularity_recs(self):
        data = [
            ["u1", "i1", 1.0, 'context1'],
            ["u2", "i3", 2.0, 'context1'],
            ["u1", "i2", 1.0, 'context2'],
            ["u3", "i3", 2.0, 'context1'],
        ]
        schema = ['user_id', 'item_id', 'relevance', 'context']
        log = self.spark.createDataFrame(data=data,
                                         schema=schema)

        users = ["u1", "u2", "u3"]
        items = ["i1", "i2", "i3"]

        true_recs_data = [
            ["u1", "i1", 0.25, "no_context"],
            ["u2", "i1", 0.25, "no_context"],
            ["u3", "i1", 0.25, "no_context"],
            ["u1", "i2", 0.25, "no_context"],
            ["u2", "i2", 0.25, "no_context"],
            ["u3", "i2", 0.25, "no_context"],
            ["u1", "i3", 0.50, "no_context"],
            ["u2", "i3", 0.50, "no_context"],
            ["u3", "i3", 0.50, "no_context"],
        ]
        true_recs_schema = ['user_id', 'item_id', 'relevance', 'context']
        true_recs = self.spark.createDataFrame(data=true_recs_data,
                                               schema=true_recs_schema)

        self.model.set_params(**{'alpha': 0, 'beta': 0})
        test_recs = self.model.fit_predict(k=10, users=users, items=items,
                                           context='no_context',
                                           log=log,
                                           user_features=None,
                                           item_features=None,
                                           to_filter_seen_items=False)

        self.assertSparkDataFrameEqual(true_recs, test_recs)


if __name__ == '__main__':
    unittest.main()
