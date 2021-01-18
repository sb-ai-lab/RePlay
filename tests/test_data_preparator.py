# pylint: disable-all
from datetime import datetime
from unittest.mock import Mock

from parameterized import parameterized
from pyspark.sql import functions as sf
from pyspark.sql.types import StringType, StructType
from tests.pyspark_testcase import PySparkTest

from replay.constants import LOG_SCHEMA
from replay.data_preparator import DataPreparator


class DataPreparatorTest(PySparkTest):
    def setUp(self):
        self.data_preparator = DataPreparator()

    # тестим read данных
    def test_read_data_wrong_columns_exception(self):
        self.assertRaises(
            ValueError,
            self.data_preparator._read_data,
            path="",
            format_type="blabla",
        )

    # тестим преобразование лога
    # тестим эксепшены
    def test_transform_log_empty_dataframe_exception(self):
        log = self.spark.createDataFrame(data=[], schema=StructType([]))
        self.data_preparator._read_data = Mock(return_value=log)

        self.assertRaises(
            ValueError,
            self.data_preparator.transform,
            path="",
            format_type="",
            columns_names={"user_id": "", "item_id": ""},
        )

    @parameterized.expand(
        [
            # columns_names
            ({"user_id": ""},),
            ({"item_id": ""},),
        ]
    )
    def test_transform_log_required_columns_exception(self, columns_names):
        self.assertRaises(
            ValueError,
            self.data_preparator.transform,
            path="",
            format_type="",
            columns_names=columns_names,
        )

    @parameterized.expand(
        [
            # log_data, log_schema, columns_names
            (
                [["user1", "item1"], ["user1", "item2"], ["user2", None]],
                ["user", "item"],
                {"user_id": "user", "item_id": "item"},
            ),
            (
                [
                    ["1", "1", "2019-01-01"],
                    ["1", "2", None],
                    ["2", "3", "2019-01-01"],
                ],
                ["user", "item", "ts"],
                {"user_id": "user", "item_id": "item", "timestamp": "ts"},
            ),
            (
                [["1", "1"], ["1", "2"], ["2", "3"]],
                ["user", "item"],
                {"user_id": "user", "item_id": "item"},
            ),
            (
                [["1", "1", 1.0], ["1", "2", 1.0], ["2", "3", None]],
                ["user", "item", "r"],
                {"user_id": "user", "item_id": "item", "relevance": "r"},
            ),
        ]
    )
    def test_transform_log_null_column_exception(
        self, log_data, log_schema, columns_names
    ):
        log = self.spark.createDataFrame(data=log_data, schema=log_schema)
        self.data_preparator._read_data = Mock(return_value=log)

        self.assertRaises(
            ValueError,
            self.data_preparator.transform,
            path="",
            format_type="",
            columns_names=columns_names,
        )

    @parameterized.expand(
        [
            # columns_names для необязательных колонок
            ({"blabla": ""},),
            ({"timestamp": "", "blabla": ""},),
            ({"relevance": "", "blabla": ""},),
            ({"timestamp": "", "blabla": ""},),
            ({"timestamp": "", "relevance": "", "blabla": ""},),
        ]
    )
    def test_transform_log_redundant_columns_exception(self, columns_names):
        # добавим обязательные колонки
        columns_names.update({"user_id": "", "item_id": ""})
        self.assertRaises(
            ValueError,
            self.data_preparator.transform,
            path="",
            format_type="",
            columns_names=columns_names,
        )

    # тестим работу функции
    @parameterized.expand(
        [
            # log_data, log_schema, true_log_data, columns_names
            (
                [["user1", "item1"], ["user1", "item2"], ["user2", "item1"]],
                ["user", "item"],
                [
                    ["user1", "item1", datetime(1999, 5, 1), 1.0],
                    ["user1", "item2", datetime(1999, 5, 1), 1.0],
                    ["user2", "item1", datetime(1999, 5, 1), 1.0],
                ],
                {"user_id": "user", "item_id": "item"},
            ),
            (
                [
                    ["u1", "i10", "2045-09-18"],
                    ["u2", "12", "1935-12-15"],
                    ["u5", "303030", "1989-06-26"],
                ],
                ["user_like", "item_like", "ts"],
                [
                    ["u1", "i10", datetime(2045, 9, 18), 1.0],
                    ["u2", "12", datetime(1935, 12, 15), 1.0],
                    ["u5", "303030", datetime(1989, 6, 26), 1.0],
                ],
                {
                    "user_id": "user_like",
                    "item_id": "item_like",
                    "timestamp": "ts",
                },
            ),
            (
                [
                    ["1010", "4944", "1945-05-25"],
                    ["4565", "134232", "2045-11-18"],
                    ["56756", "item1", "2019-02-05"],
                ],
                ["a", "b", "c"],
                [
                    ["1010", "4944", datetime(1945, 5, 25), 1.0],
                    ["4565", "134232", datetime(2045, 11, 18), 1.0],
                    ["56756", "item1", datetime(2019, 2, 5), 1.0],
                ],
                {"user_id": "a", "item_id": "b", "timestamp": "c"},
            ),
            (
                [
                    ["1945-01-25", 123.0, "12", "ue123"],
                    ["2045-07-18", 1.0, "1", "u6788888"],
                    ["2019-09-30", 0.001, "item10000", "1222222"],
                ],
                ["d", "r", "i", "u"],
                [
                    ["ue123", "12", datetime(1945, 1, 25), 123.0],
                    ["u6788888", "1", datetime(2045, 7, 18), 1.0],
                    ["1222222", "item10000", datetime(2019, 9, 30), 0.001],
                ],
                {
                    "user_id": "u",
                    "item_id": "i",
                    "timestamp": "d",
                    "relevance": "r",
                },
            ),
        ]
    )
    def test_transform_log(
        self, log_data, log_schema, true_log_data, columns_names
    ):
        log = self.spark.createDataFrame(data=log_data, schema=log_schema)
        # явно преобразовываем все к стрингам
        for column in log.columns:
            log = log.withColumn(column, sf.col(column).cast(StringType()))

        true_log = self.spark.createDataFrame(
            data=true_log_data, schema=LOG_SCHEMA
        )

        test_log = self.data_preparator.transform(
            data=log, columns_names=columns_names
        )
        self.assertSparkDataFrameEqual(true_log, test_log)

    @parameterized.expand(
        [
            # log_data, log_schema, true_log_data, columns_names
            (
                [
                    ["user1", "item1", 32],
                    ["user1", "item2", 12],
                    ["user2", "item1", 0],
                ],
                ["user", "item", "ts"],
                [
                    ["user1", "item1", datetime.fromtimestamp(32), 1.0],
                    ["user1", "item2", datetime.fromtimestamp(12), 1.0],
                    ["user2", "item1", datetime.fromtimestamp(0), 1.0],
                ],
                {"user_id": "user", "item_id": "item", "timestamp": "ts"},
            ),
            (
                [
                    ["user1", "item1", 3],
                    ["user1", "item2", 2 * 365],
                    ["user2", "item1", 365],
                ],
                ["user", "item", "ts"],
                [
                    ["user1", "item1", datetime.fromtimestamp(3), 1.0],
                    ["user1", "item2", datetime.fromtimestamp(730), 1.0],
                    ["user2", "item1", datetime.fromtimestamp(365), 1.0],
                ],
                {"user_id": "user", "item_id": "item", "timestamp": "ts"},
            ),
        ]
    )
    def test_transform_log_timestamp_column(
        self, log_data, log_schema, true_log_data, columns_names
    ):
        log = self.spark.createDataFrame(data=log_data, schema=log_schema)

        true_log = self.spark.createDataFrame(
            data=true_log_data, schema=LOG_SCHEMA
        )

        test_log = self.data_preparator.transform(
            data=log, columns_names=columns_names
        )
        self.assertSparkDataFrameEqual(true_log, test_log)

    @parameterized.expand(
        [
            # log_data, log_schema, true_log_data, columns_names
            (
                [
                    ["u1", "f1", "2019-01-01 10:00:00"],
                    ["u1", "f2", "1995-11-01 00:00:00"],
                    ["u2", "f1", "2000-03-30 00:00:00"],
                ],
                ["user", "item", "string_time"],
                [
                    ["u1", "f1", datetime(2019, 1, 1, 10), 1.0],
                    ["u1", "f2", datetime(1995, 11, 1), 1.0],
                    ["u2", "f1", datetime(2000, 3, 30), 1.0],
                ],
                {
                    "user_id": "user",
                    "item_id": "item",
                    "timestamp": "string_time",
                },
            ),
        ]
    )
    def test_transform_log_timestamp_format(
        self, log_data, log_schema, true_log_data, columns_names
    ):
        log = self.spark.createDataFrame(data=log_data, schema=log_schema)
        log.show()
        print(LOG_SCHEMA)
        true_log = self.spark.createDataFrame(
            data=true_log_data, schema=LOG_SCHEMA
        )

        test_log = self.data_preparator.transform(
            data=log,
            columns_names=columns_names,
            date_format="yyyy-MM-dd HH:mm:ss",
        )
        test_log.show()
        self.assertSparkDataFrameEqual(true_log, test_log)

    # тестим преобразование фичей
    # тестим эксепшены
    def test_transform_features_empty_dataframe_exception(self):
        features = self.spark.createDataFrame(data=[], schema=StructType([]))
        self.data_preparator._read_data = Mock(return_value=features)

        self.assertRaises(
            ValueError,
            self.data_preparator.transform,
            path="",
            format_type="",
            columns_names={"user_id": ""},
        )

    @parameterized.expand(
        [
            # columns_names
            ({"timestamp": ""},),
            ({"": ""},),
            ({"blabla": ""},),
        ]
    )
    def test_transform_features_required_columns_exception(
        self, columns_names
    ):
        self.assertRaises(
            ValueError,
            self.data_preparator.transform,
            path="",
            format_type="",
            columns_names=columns_names,
        )

    @parameterized.expand(
        [
            # feature_data, feature_schema, columns_names
            (
                [["user1", 1], ["user1", 1], ["user2", None]],
                ["user", "feature"],
                {"user_id": "user", "feature": "feature"},
            ),
            (
                [["1", "2019-01-01"], ["2", None], ["3", "2019-01-01"]],
                ["item", "ts"],
                {"item_id": "item", "timestamp": "ts"},
            ),
            (
                [
                    ["1", 1, None],
                    ["1", 2, "2019-01-01"],
                    ["2", 3, "2019-01-01"],
                ],
                ["user", "feature", "timestamp"],
                {
                    "user_id": "user",
                    "feature": "feature",
                    "timestamp": "timestamp",
                },
            ),
            (
                [
                    ["1", 1, 100, "2019-01-01"],
                    ["1", 2, 100, "2019-01-01"],
                    ["2", 3, None, "2019-01-01"],
                ],
                ["user", "f1", "f2", "timestamp"],
                {
                    "user_id": "user",
                    "feature": ["f1", "f2"],
                    "timestamp": "timestamp",
                },
            ),
        ]
    )
    def test_transform_features_null_column_exception(
        self, feature_data, feature_schema, columns_names
    ):
        features = self.spark.createDataFrame(
            data=feature_data, schema=feature_schema
        )
        self.data_preparator._read_data = Mock(return_value=features)

        self.assertRaises(
            ValueError,
            self.data_preparator.transform,
            path="",
            format_type="",
            columns_names=columns_names,
        )

    @parameterized.expand(
        [
            # columns_names
            ({"item_id": "", "blabla": ""},),
            ({"item_id": "", "timestamp": "", "blabla": ""},),
            ({"user_id": "", "blabla": ""},),
            ({"user_id": "", "timestamp": "", "blabla": ""},),
        ]
    )
    def test_transform_features_redundant_columns_exception(
        self, columns_names
    ):
        self.assertRaises(
            ValueError,
            self.data_preparator.transform,
            path="",
            format_type="",
            columns_names=columns_names,
        )

    @parameterized.expand(
        [
            # columns_names, features_schema
            ({"item_id": "item"}, ["item"]),
            ({"user_id": "user"}, ["user"]),
            ({"item_id": "item", "timestamp": "ts"}, ["item", "ts"]),
            ({"user_id": "user", "timestamp": "ts"}, ["user", "ts"]),
        ]
    )
    def test_transform_features_no_feature_columns_exception(
        self, columns_names, features_schema
    ):
        feature_data = [
            ["id1", "2019-01-01"],
            ["id1", "2019-01-01"],
            ["id2", "2019-01-01"],
        ]
        features = self.spark.createDataFrame(
            data=feature_data, schema=features_schema
        )
        features = features.select(features_schema)
        # явно преобразовываем все к стрингам
        for column in features.columns:
            features = features.withColumn(
                column, sf.col(column).cast(StringType())
            )

        self.data_preparator._read_data = Mock(return_value=features)

        self.assertRaises(
            ValueError,
            self.data_preparator.transform,
            path="",
            format_type="",
            columns_names=columns_names,
        )

    # тестим работу функции
    @parameterized.expand(
        [
            # feature_data, feature_schema, true_feature_data, columns_names, features_columns
            (
                [
                    ["user1", "feature1"],
                    ["user1", "feature2"],
                    ["user2", "feature1"],
                ],
                ["user", "f0"],
                [
                    ["user1", "feature1"],
                    ["user1", "feature2"],
                    ["user2", "feature1"],
                ],
                {"user_id": "user"},
                "f0",
            ),
            (
                [
                    ["u1", "f1", "2019-01-01", 1],
                    ["u1", "f2", "2019-01-01", 2],
                    ["u2", "f1", "2019-01-01", 3],
                ],
                ["user", "f0", "f1", "f2"],
                [
                    ["u1", "f1", "2019-01-01", 1],
                    ["u1", "f2", "2019-01-01", 2],
                    ["u2", "f1", "2019-01-01", 3],
                ],
                {"user_id": "user"},
                ["f0", "f1", "f2"],
            ),
        ]
    )
    def test_transform_features(
        self,
        feature_data,
        feature_schema,
        true_feature_data,
        columns_names,
        features_columns,
    ):
        features = self.spark.createDataFrame(
            data=feature_data, schema=feature_schema
        )

        if "timestamp" in columns_names:
            schema = ["user_id", "timestamp"] + [
                f"f{i}" for i in range(len(true_feature_data[0]) - 2)
            ]
        else:
            schema = ["user_id"] + [
                f"f{i}" for i in range(len(true_feature_data[0]) - 1)
            ]

        true_features = self.spark.createDataFrame(
            data=true_feature_data, schema=schema
        )
        true_features = true_features.withColumn(
            "user_id", sf.col("user_id").cast(StringType())
        )
        test_features = self.data_preparator.transform(
            data=features,
            columns_names=columns_names,
            features_columns=features_columns,
        )
        self.assertSparkDataFrameEqual(true_features, test_features)
