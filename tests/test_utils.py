# pylint: disable-all
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql import SparkSession

import replay.session_handler
from replay import utils
from tests.utils import (
    item_features,
    long_log_with_features,
    short_log_with_features,
    spark,
)


def test_func_get():
    vector = np.arange(2)
    assert utils.func_get(vector, 0) == 0.0


def test_get_spark_session():
    spark = replay.session_handler.get_spark_session(1)
    assert isinstance(spark, SparkSession)
    assert spark.conf.get("spark.driver.memory") == "1g"
    assert replay.session_handler.State(spark).session is spark
    assert replay.session_handler.State().session is spark


def test_convert():
    df = pd.DataFrame([[1, "a", 3.0], [3, "b", 5.0]], columns=["a", "b", "c"])
    sf = utils.convert2spark(df)
    pd.testing.assert_frame_equal(df, sf.toPandas())
    assert utils.convert2spark(sf) is sf


def del_files_by_pattern(directory: str, pattern: str) -> None:
    """
    Удаляет файлы из директории в соответствии с заданным паттерном имени файла
    """
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            os.remove(os.path.join(directory, filename))


def find_file_by_pattern(directory: str, pattern: str) -> Optional[str]:
    """
    Возвращает путь к первому найденному файлу в директории, соответствующему паттерну,
    или None, если таких файлов нет
    """
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            return os.path.join(directory, filename)


def get_transformed_features(transformer, train, test):
    transformer.fit(train)
    return transformer.transform(test)


def test_cat_features_transformer_with_col_list(
    long_log_with_features, item_features
):
    transformed = get_transformed_features(
        # при передаче списка можно сделать кодирование и по числовой фиче
        transformer=utils.CatFeaturesTransformer(
            cat_cols_list=["iq", "class"]
        ),
        train=item_features.filter(sf.col("class") != "dog"),
        test=item_features,
    )
    assert "class" not in transformed.columns
    assert "iq" in transformed.columns and "color" in transformed.columns
    assert (
        "ohe_class_dog" not in transformed.columns
        and "ohe_class_cat" in transformed.columns
    )
    assert (
        transformed.filter(sf.col("item_id") == "i6")
        .select("ohe_class_mouse")
        .collect()[0][0]
        == 1.0
    )


def test_cat_features_transformer_with_col_list_date(
    long_log_with_features, short_log_with_features,
):
    transformed = get_transformed_features(
        transformer=utils.CatFeaturesTransformer(),
        train=long_log_with_features,
        test=short_log_with_features,
    )
    transformed.show()
    assert (
        "ohe_timestamp_20190101000000" in transformed.columns
        and "item_id" in transformed.column
    )


def test_cat_features_transformer_without_col_list(
    long_log_with_features, item_features
):
    transformed = get_transformed_features(
        transformer=utils.CatFeaturesTransformer(
            cat_cols_list=None, threshold=3
        ),
        train=item_features.filter(sf.col("class") != "dog"),
        test=item_features,
    )
    assert "iq" in transformed.columns and "color" not in transformed.columns
    assert (
        "ohe_class_dog" not in transformed.columns
        and "ohe_class_cat" in transformed.columns
    )
    assert sorted(transformed.columns) == [
        "iq",
        "item_id",
        "ohe_class_cat",
        "ohe_class_mouse",
        "ohe_color_black",
        "ohe_color_yellow",
    ]

    # в категориальных колонках больше значений, чем threshold
    transformed = get_transformed_features(
        transformer=utils.CatFeaturesTransformer(
            cat_cols_list=None, threshold=1
        ),
        train=item_features.filter(sf.col("class") != "dog"),
        test=item_features,
    )
    assert "iq" in transformed.columns and "color" not in transformed.columns
    assert sorted(transformed.columns) == ["iq", "item_id"]

    # обработка None и случаев, когда все колонки отфильтровались
    transformer = utils.CatFeaturesTransformer(cat_cols_list=None, threshold=1)
    transformed = get_transformed_features(
        transformer=transformer,
        train=item_features.select("color"),
        test=item_features,
    )
    assert transformer.no_cols_left is True and transformed is None
    assert (
        get_transformed_features(
            transformer=transformer, train=None, test=item_features,
        )
        is None
    )
