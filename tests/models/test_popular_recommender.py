import unittest

from parameterized import parameterized
from pyspark.sql import functions as sf

from pyspark_testcase import PySparkTest
from sponge_bob_magic.models.popular_recomennder import PopularRecommender


class PopularRecommenderTestCase(PySparkTest):
    def setUp(self):
        self.model = PopularRecommender(self.spark)

    @parameterized.expand([
        # users, context, to_filter_seen_items, k, items
        # проверяем выделение айтемов
        (["u1", "u2", "u3"], "no_context", 10, [["i1", 3 / 14],
                                                ["i2", 2 / 14],
                                                ["i3", 4 / 14],
                                                ["i4", 5 / 14],
                                                ["i999", 0.0]],),
        (["u1", "u2", "u3"], "c1", 10, [["i1", 2 / 7],
                                        ["i999", 0.0],
                                        ["i998", 0.0],
                                        ["i3", 3 / 7],
                                        ["i4", 2 / 7]],),
        (["u1", "u2", "u3"], "c2", 10, [["i1", 1 / 7],
                                        ["i2", 2 / 7],
                                        ["i3", 1 / 7],
                                        ["i998", 0.0],
                                        ["i999", 0.0],
                                        ["i4", 3 / 7]],),

        # проверяем выделение юзеров
        (["u1", "u2"], "no_context", 10, [["i1", 3 / 14],
                                          ["i2", 2 / 14],
                                          ["i3", 4 / 14],
                                          ["i4", 5 / 14],
                                          ["i999", 0.0]],),

        # проверяем выделение топ-к
        (["u1", "u2"], "c1", 1, [["i3", 3 / 7]]),
        (["u1", "u3"], "c2", 2, [["i2", 2 / 7],
                                 ["i4", 3 / 7]],),
        (["u3", "u2"], "no_context", 3, [["i1", 3 / 14],
                                         ["i3", 4 / 14],
                                         ["i4", 5 / 14]]),
    ])
    def test_popularity_recs_no_params(self,
                                       users, context, k,
                                       items_relevance
                                       ):
        log_data = [
            ["u1", "i1", 1.0, "c1"],
            ["u2", "i1", 1.0, "c1"],
            ["u3", "i3", 2.0, "c1"],
            ["u3", "i3", 2.0, "c1"],
            ["u2", "i3", 2.0, "c1"],
            ["u3", "i4", 2.0, "c1"],
            ["u1", "i4", 2.0, "c1"],

            ["u2", "i1", 3.0, "c2"],
            ["u3", "i2", 1.0, "c2"],
            ["u2", "i2", 1.0, "c2"],
            ["u2", "i3", 2.0, "c2"],
            ["u3", "i4", 3.0, "c2"],
            ["u2", "i4", 2.0, "c2"],
            ["u1", "i4", 1.0, "c2"],
        ]
        log_schema = ['user_id', 'item_id', 'relevance', 'context']
        log = self.spark.createDataFrame(data=log_data,
                                         schema=log_schema)

        true_recs = (
            self.spark.createDataFrame(data=[[user] for user in users],
                                       schema=['user_id'])
                .crossJoin(
                self.spark.createDataFrame(items_relevance,
                                           schema=['item_id', 'relevance']))
        )
        true_recs = true_recs \
            .withColumn('context', sf.lit(context))

        self.model.set_params(**{'alpha': 0, 'beta': 0})

        # два вызова нужны, чтобы проверить, что они возващают одно и то же
        test_recs_first = self.model.fit_predict(
            k=k, users=users,
            items=set([elem[0]
                       for elem in items_relevance]),
            context=context,
            log=log,
            user_features=None,
            item_features=None,
            to_filter_seen_items=False)
        test_recs_second = self.model.fit_predict(
            k=k, users=users,
            items=set([elem[0]
                       for elem in items_relevance]),
            context=context,
            log=log,
            user_features=None,
            item_features=None,
            to_filter_seen_items=False)

        self.assertSparkDataFrameEqual(true_recs, test_recs_second)
        self.assertSparkDataFrameEqual(true_recs, test_recs_first)
        self.assertSparkDataFrameEqual(test_recs_first, test_recs_second)


if __name__ == '__main__':
    unittest.main()
