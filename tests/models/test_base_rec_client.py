from os.path import dirname, join

import numpy as np
import pandas as pd
import polars as pl
import pytest

import replay
from replay.metrics import NDCG
from replay.data.dataset import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.models import UCB, PopRec, client_model_list, BaseRecommenderClient, NonPersonolizedRecommenderClient
from replay.models.base_rec_client import NotFittedModelError
from replay.utils.common import convert2pandas, convert2polars, convert2spark
from replay.utils.types import DataFrameLike
from tests.utils import (
    SparkDataFrame,
    get_dataset_any_type,
    isDataFrameEqual,
)
from tests.models.test_all_models_via_client import datasets, pandas_interactions, spark_interactions, polars_interactions


pyspark = pytest.importorskip("pyspark")
from pyspark.sql import functions as sf

SEED = 123

@pytest.mark.core
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize("type_of_impl, is_pandas, is_spark, is_polars", [
    ("pandas", True, False, False),
    ("spark", False, True, False),
    ("polars", False, False, True)
])
def test_assign_implementation_type_valid(base_model, arguments, type_of_impl, is_pandas, is_spark, is_polars):
    model = base_model(**arguments)
    model._assign_implementation_type(type_of_impl)
    assert model.is_pandas == is_pandas
    assert model.is_spark == is_spark
    assert model.is_polars == is_polars

@pytest.mark.core
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize("invalid_type_of_model", [
    None,
    "wrong_value",
    123,
    []
])
def test_assign_implementation_type_invalid(base_model, arguments, invalid_type_of_model):
    model = base_model(**arguments)
    with pytest.raises(ValueError, match="Argument type_of_model can be spark|pandas|polars"):
        model._assign_implementation_type(invalid_type_of_model)
    
@pytest.mark.core
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize(
    "attribute, expected_result",
    [
        ("is_pandas", "pandas"),
        ("is_spark", "spark"),
        ("is_polars", "polars"),
        (None, None), 
    ]
)
def test_get_implementation_type(base_model, arguments, attribute, expected_result):
    model = base_model(**arguments)
    if attribute:
        model._assign_implementation_type(expected_result)
    if attribute:
        setattr(model, attribute, True)
    if attribute is not None:
        assert model._get_implementation_type() == expected_result
    else:
        with pytest.raises(AttributeError, match="does not have the 'logger' attribut"):
            model._get_implementation_type()


class MockImpl:
    def method_1(self):
        pass
    
    def method_2(self):
        pass
    
    attr_1 = 'value'
    attr_2 = 'another value'

    def __init__(self):
        self.instance_attr = 'instance attribute'


@pytest.mark.core
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
def test_when_impl_is_none(base_model, arguments):
    model = base_model(**arguments)
    result = model._get_all_attributes_or_functions()
    assert result == []


@pytest.mark.core
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
def test_with_mock_impl(base_model, arguments, datasets):
    dataset_pl = datasets["polars"]
    dataset_pd = datasets["pandas"]
    dataset_spark = datasets["spark"]
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    model_pd.fit(dataset_pd)
    model_pl.fit(dataset_pl)
    model_spark.fit(dataset_spark)
    result_pd = model_pd._get_all_attributes_or_functions()
    result_pl = model_pl._get_all_attributes_or_functions()
    result_spark = model_spark._get_all_attributes_or_functions()
    expected_attrs = ['_init_args', '_dataframes', 'fit_items', 'fit_queries', 'queries_count', 'items_count', 'logger', 'can_predict_cold_items', 'can_predict_cold_queries', 'query_column', 'item_column', "rating_column", "timestamp_column", '_num_queries', "_num_items", "set_params", "fit_predict", "fit", "predict", "predict_pairs"]
    for attr in expected_attrs:
        assert attr in result_pd
        assert attr in result_pl
        assert attr in result_spark
    assert len(result) >= len(expected_attrs)
