import unittest

from parameterized import parameterized
from pyspark.sql import functions as sf

from sponge_bob_magic import constants

from pyspark_testcase import PySparkTest
from sponge_bob_magic.models.popular_recomennder import PopularRecommender


class PopularRecommenderTestCase(PySparkTest):
    def setUp(self):
        self.model = PopularRecommender(self.spark)

    @parameterized.expand([
        # users, context, k, items_relevance
        # проверяем выделение айтемов
        (["u1", "u2", "u3"], constants.DEFAULT_CONTEXT, 5, [["i1", 3 / 14],
                                               ["i2", 2 / 14],
                                               ["i3", 4 / 14],
                                               ["i4", 5 / 14],
                                               ["i999", 0.0]],),
        (["u1", "u2", "u3"], "c1", 5, [["i1", 2 / 7],
                                       ["i999", 0.0],
                                       ["i998", 0.0],
                                       ["i3", 3 / 7],
                                       ["i4", 2 / 7]],),
        (["u1", "u2", "u3"], "c2", 6, [["i1", 1 / 7],
                                       ["i2", 2 / 7],
                                       ["i3", 1 / 7],
                                       ["i998", 0.0],
                                       ["i999", 0.0],
                                       ["i4", 3 / 7]],),

        # проверяем выделение юзеров
        (["u1", "u2"], constants.DEFAULT_CONTEXT, 5, [["i1", 3 / 14],
                                         ["i2", 2 / 14],
                                         ["i3", 4 / 14],
                                         ["i4", 5 / 14],
                                         ["i999", 0.0]],),

        # проверяем выделение топ-к
        (["u1", "u2"], "c1", 1, [["i3", 3 / 7]]),
        (["u1", "u3"], "c2", 2, [["i2", 2 / 7],
                                 ["i4", 3 / 7]],),
        (["u3", "u2"], constants.DEFAULT_CONTEXT, 3, [["i1", 3 / 14],
                                         ["i3", 4 / 14],
                                         ["i4", 5 / 14]]),
    ])
    def test_popularity_recs_no_params(self,
                                       users, context, k,
                                       items_relevance):
        log_data = [
            ["u1", "i1", 1.0, "c1", "2019-01-01"],
            ["u2", "i1", 1.0, "c1", "2019-01-01"],
            ["u3", "i3", 2.0, "c1", "2019-01-01"],
            ["u3", "i3", 2.0, "c1", "2019-01-01"],
            ["u2", "i3", 2.0, "c1", "2019-01-01"],
            ["u3", "i4", 2.0, "c1", "2019-01-01"],
            ["u1", "i4", 2.0, "c1", "2019-01-01"],

            ["u2", "i1", 3.0, "c2", "2019-01-01"],
            ["u3", "i2", 1.0, "c2", "2019-01-01"],
            ["u2", "i2", 1.0, "c2", "2019-01-01"],
            ["u2", "i3", 2.0, "c2", "2019-01-01"],
            ["u3", "i4", 3.0, "c2", "2019-01-01"],
            ["u2", "i4", 2.0, "c2", "2019-01-01"],
            ["u1", "i4", 1.0, "c2", "2019-01-01"],
        ]
        log_schema = ['user_id', 'item_id', 'relevance',
                      'context', 'timestamp']
        log = self.spark.createDataFrame(data=log_data,
                                         schema=log_schema)

        items_relevance = self.spark.createDataFrame(items_relevance,
                                                     schema=['item_id',
                                                             'relevance'])
        users = self.spark.createDataFrame(data=[[user] for user in users],
                                           schema=['user_id'])

        true_recs = users.crossJoin(items_relevance)
        true_recs = (true_recs
                     .withColumn('context', sf.lit(context)))

        self.model.set_params(**{'alpha': 0, 'beta': 0})

        # два вызова нужны, чтобы проверить, что они возващают одно и то же
        test_recs_first = self.model.fit_predict(
            k=k, users=users,
            items=items_relevance.select('item_id'),
            context=context,
            log=log,
            user_features=None,
            item_features=None,
            to_filter_seen_items=False)
        test_recs_second = self.model.fit_predict(
            k=k, users=users,
            items=items_relevance.select('item_id'),
            context=context,
            log=log,
            user_features=None,
            item_features=None,
            to_filter_seen_items=False)

        self.assertSparkDataFrameEqual(true_recs, test_recs_second)
        self.assertSparkDataFrameEqual(true_recs, test_recs_first)
        self.assertSparkDataFrameEqual(test_recs_first, test_recs_second)

    def test_popularity_recs_no_params_to_filter_seen_items(self):
        log_data = [
            ["u1", "i1", 1.0, "c1", "2019-01-01"],
            ["u1", "i4", 2.0, "c1", "2019-01-01"],
            ["u2", "i1", 1.0, "c1", "2019-01-01"],
            ["u2", "i3", 2.0, "c1", "2019-01-01"],
            ["u3", "i3", 2.0, "c1", "2019-01-01"],

            ["u3", "i4", 1.0, "c2", "2019-01-01"],
            ["u3", "i3", 2.0, "c2", "2019-01-01"],
        ]
        log_schema = ['user_id', 'item_id', 'relevance',
                      'context', 'timestamp']
        log = self.spark.createDataFrame(data=log_data,
                                         schema=log_schema)
        context = 'c1'  # вычищение лога не зависит от контекста

        true_recs_data = [
            ["u1", "i2", 0 / 5, context],
            ["u1", "i3", 2 / 5, context],
            ["u2", "i2", 0 / 5, context],
            ["u2", "i4", 1 / 5, context],
            ["u3", "i1", 2 / 5, context],
            ["u3", "i2", 0 / 5, context],
        ]
        true_recs_schema = ['user_id', 'item_id', 'relevance', 'context']
        true_recs = self.spark.createDataFrame(data=true_recs_data,
                                               schema=true_recs_schema)

        self.model.set_params(**{'alpha': 0, 'beta': 0})

        users = self.spark.createDataFrame(
            data=[[user] for user in ["u1", "u2", "u3"]],
            schema=['user_id'])
        items = self.spark.createDataFrame(
            data=[[item] for item in ["i1", "i2", "i3", "i4"]],
            schema=['item_id'])

        test_recs = self.model.fit_predict(
            k=2, users=users,
            items=items,
            context=context,
            log=log,
            user_features=None,
            item_features=None,
            to_filter_seen_items=True)

        self.assertSparkDataFrameEqual(true_recs, test_recs)

    @parameterized.expand([
        # alpha, beta
        (1, 1),
        (10, 100),
        (999, 0),
        (0, 19999),
        (0.0009, 0.4),
    ])
    def test_popularity_recs_with_params(self, alpha, beta):
        log_data = [
            ["u2", "i1", 1.0, "c1", "2019-01-01"],
            ["u3", "i3", 2.0, "c1", "2019-01-01"],
            ["u1", "i4", 2.0, "c1", "2019-01-01"],

            ["u1", "i1", 1.0, "c2", "2019-01-01"],
            ["u3", "i1", 2.0, "c2", "2019-01-01"],
            ["u2", "i2", 1.0, "c2", "2019-01-01"],
            ["u2", "i3", 3.0, "c2", "2019-01-01"],
            ["u3", "i4", 2.0, "c2", "2019-01-01"],
            ["u1", "i4", 2.0, "c2", "2019-01-01"],
            ["u3", "i4", 4.0, "c2", "2019-01-01"],
        ]
        log_schema = ['user_id', 'item_id', 'relevance',
                      'context', 'timestamp']
        log = self.spark.createDataFrame(data=log_data,
                                         schema=log_schema)
        context = 'c2'

        true_recs_data = [
            ["u1", "i1", (2 + alpha) / (beta + 7), context],
            ["u1", "i2", (1 + alpha) / (beta + 7), context],
            ["u1", "i3", (1 + alpha) / (beta + 7), context],
            ["u1", "i4", (3 + alpha) / (beta + 7), context],
            ["u2", "i1", (2 + alpha) / (beta + 7), context],
            ["u2", "i2", (1 + alpha) / (beta + 7), context],
            ["u2", "i3", (1 + alpha) / (beta + 7), context],
            ["u2", "i4", (3 + alpha) / (beta + 7), context],
            ["u3", "i1", (2 + alpha) / (beta + 7), context],
            ["u3", "i2", (1 + alpha) / (beta + 7), context],
            ["u3", "i3", (1 + alpha) / (beta + 7), context],
            ["u3", "i4", (3 + alpha) / (beta + 7), context],
        ]
        true_recs_schema = ['user_id', 'item_id', 'relevance', 'context']
        true_recs = self.spark.createDataFrame(data=true_recs_data,
                                               schema=true_recs_schema)

        self.model.set_params(**{'alpha': alpha, 'beta': beta})

        test_recs = self.model.fit_predict(
            k=4, users=None,
            items=None,
            context=context,
            log=log,
            user_features=None,
            item_features=None,
            to_filter_seen_items=False)

        self.assertSparkDataFrameEqual(true_recs, test_recs)

    def test_popularity_recs_default_items_users(self):
        log_data = [
            ["u2", "i1", 1.0, "c1", "2019-01-01"],
            ["u3", "i3", 2.0, "c1", "2019-01-01"],
            ["u1", "i4", 2.0, "c1", "2019-01-01"],

            ["u1", "i1", 1.0, "c2", "2019-01-01"],
            ["u3", "i1", 2.0, "c2", "2019-01-01"],
            ["u2", "i2", 1.0, "c2", "2019-01-01"],
            ["u2", "i3", 3.0, "c2", "2019-01-01"],
            ["u3", "i4", 2.0, "c2", "2019-01-01"],
            ["u1", "i4", 2.0, "c2", "2019-01-01"],
            ["u3", "i4", 4.0, "c2", "2019-01-01"],
        ]
        log_schema = ['user_id', 'item_id', 'relevance',
                      'context', 'timestamp']
        log = self.spark.createDataFrame(data=log_data,
                                         schema=log_schema)
        context = 'c2'

        true_recs_data = [
            ["u1", "i1", 2 / 7, context],
            ["u1", "i2", 1 / 7, context],
            ["u1", "i3", 1 / 7, context],
            ["u1", "i4", 3 / 7, context],
            ["u2", "i1", 2 / 7, context],
            ["u2", "i2", 1 / 7, context],
            ["u2", "i3", 1 / 7, context],
            ["u2", "i4", 3 / 7, context],
            ["u3", "i1", 2 / 7, context],
            ["u3", "i2", 1 / 7, context],
            ["u3", "i3", 1 / 7, context],
            ["u3", "i4", 3 / 7, context],
        ]
        true_recs_schema = ['user_id', 'item_id', 'relevance', 'context']
        true_recs = self.spark.createDataFrame(data=true_recs_data,
                                               schema=true_recs_schema)

        self.model.set_params(**{'alpha': 0, 'beta': 0})

        test_recs = self.model.fit_predict(
            k=4, users=None,
            items=None,
            context=context,
            log=log,
            user_features=None,
            item_features=None,
            to_filter_seen_items=False)

        self.assertSparkDataFrameEqual(true_recs, test_recs)


if __name__ == '__main__':
    unittest.main()
