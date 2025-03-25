import pandas as pd
import polars as pl
import pytest

from replay.models import NonPersonolizedRecommenderClient, PopRec
from replay.models.base_rec_client import DataModelMissmatchError, NotFittedModelError
from replay.models.implementations import _PopRecPandas, _PopRecPolars, _PopRecSpark
from replay.utils.common import convert2spark
from tests.utils import isDataFrameEqual

pyspark = pytest.importorskip("pyspark")

SEED = 123


@pytest.mark.core
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize(
    "type_of_impl, is_pandas, is_spark, is_polars",
    [("pandas", True, False, False), ("spark", False, True, False), ("polars", False, False, True)],
)
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
@pytest.mark.parametrize("invalid_type_of_model", [None, "wrong_value", 123, []])
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
    ],
)
def test_get_implementation_type(base_model, arguments, attribute, expected_result):
    model = base_model(**arguments)
    if attribute:
        model._assign_implementation_type(expected_result)
    if attribute:
        setattr(model, attribute, True)
    assert model._get_implementation_type() == expected_result


@pytest.mark.spark
@pytest.mark.parametrize("base_model, arguments", [(PopRec, {})], ids=["pop_rec_spark"])
def test_cached_dfs(base_model, arguments, spark_interactions):
    model = base_model(**arguments)
    model._impl = model._class_map["spark"]()
    model._assign_implementation_type("spark")
    assert model.cached_dfs is None
    df_name = "interactions"
    model._impl._cache_model_temp_view(spark_interactions, "interactions")
    full_name = f"id_{id(model._impl)}_model_{model._impl!s}_{df_name}"
    assert model.cached_dfs == set([full_name])  # noqa: C405
    model._impl._clear_model_temp_view("interactions")
    assert model.cached_dfs == set()


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize(
    "type_of_impl", ["pandas", "spark", "polars"], ids=["invalid_pandas", "valid_impl_spark", "invalid_polars"]
)
def test_cached_dfs_invalid(base_model, arguments, type_of_impl):
    model = base_model(**arguments)
    model._impl = model._class_map[type_of_impl]()
    model._assign_implementation_type(type_of_impl)
    if type_of_impl != "spark":
        with pytest.raises(AttributeError, match="does not have the 'cached_dfs' attribute"):
            model.cached_dfs
    else:
        assert model.cached_dfs is None


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize("type_of_impl", ["pandas", "spark", "polars"])
def test_logger(base_model, arguments, type_of_impl):
    model = base_model(**arguments)
    model._impl = model._class_map[type_of_impl]()
    assert model._impl._logger is None  # In implementations _logger is None after __init__
    assert model.logger is not None  # but after first time logger called, it is not None
    model.logger.debug("Logger works")


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
def test_logger_invalid(base_model, arguments):
    model = base_model(**arguments)
    with pytest.raises(AttributeError, match="does not have the 'logger' attribute"):
        model.logger.debug("This call is not working")


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments, impl",
    [(PopRec, {}, "spark"), (PopRec, {}, "pandas"), (PopRec, {}, "polars")],
    ids=["pop_rec_spark", "pop_rec_pandas", "pop_rec_polars"],
)
@pytest.mark.parametrize("attribute_name", ["queries_count", "items_count"])
def test_queries_items_count(base_model, arguments, impl, attribute_name, datasets):
    model = base_model(**arguments)
    with pytest.raises(AttributeError, match=f"does not have the '{attribute_name}' attribute"):
        assert getattr(model, attribute_name)
    assert model._impl is None
    assert not model.is_fitted
    model.fit_items = pd.DataFrame([1, 2])
    model.fit_queries = pd.DataFrame([1])
    assert model.items_count == 2
    assert model.queries_count == 1
    model.fit(datasets[impl])
    assert model._impl is not None
    assert model.items_count == 3
    assert model.queries_count == 4


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize(
    "attribute_name", ["model", "can_predict_cold_queries", "can_predict_cold_items", "_search_space", "_objective"]
)
@pytest.mark.parametrize("type_of_impl", ["pandas", "spark", "polars"])
def test_attributes(base_model, arguments, type_of_impl, attribute_name):
    class Attribute:
        value = 123

    model = base_model(**arguments)
    model._impl = model._class_map[type_of_impl]()
    model._assign_implementation_type(type_of_impl)
    setattr(model._impl, attribute_name, Attribute())
    assert getattr(model, attribute_name).value == 123


@pytest.mark.spark
@pytest.mark.parametrize("base_model, arguments", [(PopRec, {})], ids=["pop_rec"])
@pytest.mark.parametrize("attribute_name", ["model", "study", "criterion"])
def test_setters_of_attributes_not_fitted(base_model, arguments, attribute_name):
    class Attribute:
        value = 123

    model = base_model(**arguments)
    setattr(model, attribute_name, Attribute())
    attrs_dict = model._init_when_first_impl_arrived_args
    assert attrs_dict[attribute_name].value == 123
    with pytest.raises(AttributeError, match=f"does not have the '{attribute_name}' attribute"):
        assert getattr(model, attribute_name).value == 123


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments, type_of_impl",
    [(PopRec, {}, "polars"), (PopRec, {}, "pandas"), (PopRec, {}, "spark")],
    ids=["pop_rec_polars", "pop_rec_pandas", "pop_rec_spark"],
)
@pytest.mark.parametrize("attribute_name", ["model", "study", "criterion"])
def test_setters_of_attributes_fitted(base_model, arguments, attribute_name, type_of_impl, datasets):
    class Attribute:
        value = 123

    model = base_model(**arguments)
    setattr(model, attribute_name, Attribute())
    attrs_dict = model._init_when_first_impl_arrived_args
    assert attrs_dict[attribute_name].value == 123
    model.fit(datasets[type_of_impl])
    assert getattr(model, attribute_name).value == 123


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments, type_of_impl",
    [(PopRec, {}, "polars"), (PopRec, {}, "pandas"), (PopRec, {}, "spark")],
    ids=["pop_rec_polars", "pop_rec_pandas", "pop_rec_spark"],
)
@pytest.mark.parametrize("attribute_name", ["model", "study", "criterion"])
def test_setters_of_attributes_after_fit(base_model, arguments, attribute_name, type_of_impl, datasets):
    class Attribute:
        value = 123

    model = base_model(**arguments)
    model.fit(datasets[type_of_impl])
    if attribute_name not in ["items_count", "queries_count"]:  # alredy init properties
        setattr(model, attribute_name, Attribute())
    assert getattr(model, attribute_name).value == 123


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize(
    "attribute_name", ["model", "can_predict_cold_queries", "can_predict_cold_items", "_search_space", "_objective"]
)
def test_attrubutes_invalid(base_model, arguments, attribute_name):
    model = base_model(**arguments)
    with pytest.raises(AttributeError, match=f"does not have the '{attribute_name}' attribute"):
        getattr(model, attribute_name)


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments, type_of_impl",
    [(PopRec, {}, "polars"), (PopRec, {}, "pandas"), (PopRec, {}, "spark")],
    ids=["pop_rec_polars", "pop_rec_pandas", "pop_rec_spark"],
)
@pytest.mark.parametrize("attribute_name", PopRec.attributes_after_fit)
def test_invalid_attributes_after_fit(base_model, arguments, type_of_impl, attribute_name, datasets):
    class Attribute:
        value = 123

    model = base_model(**arguments)
    model.fit(datasets[type_of_impl])
    setattr(model._impl, attribute_name, Attribute())
    assert getattr(model, attribute_name).value == 123


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments, type_of_impl",
    [(PopRec, {}, "polars"), (PopRec, {}, "pandas"), (PopRec, {}, "spark")],
    ids=["pop_rec_polars", "pop_rec_pandas", "pop_rec_spark"],
)
@pytest.mark.parametrize("attribute_name", PopRec.attributes_after_fit_with_setter)
def test_set_attributes_to_client_after_fit(base_model, arguments, type_of_impl, attribute_name, datasets):
    example_df = datasets[type_of_impl].interactions
    model = base_model(**arguments)
    model.fit(datasets[type_of_impl])
    setattr(model, attribute_name, example_df)
    assert isDataFrameEqual(getattr(model, attribute_name), example_df)


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize("attribute_name", PopRec.attributes_after_fit)
def test_invalid_attrubutes_after_fit(base_model, arguments, attribute_name):
    model = base_model(**arguments)
    with pytest.raises(AttributeError, match=f"does not have the '{attribute_name}'"):
        getattr(model, attribute_name)


pytest.mark.spark


@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
def test_get_features(base_model, arguments, datasets):
    dataset_pd = datasets["pandas"]
    dataset_spark = datasets["spark"]
    dataset_pl = datasets["polars"]
    item_col = dataset_pd._feature_schema.item_id_column
    model_pd = base_model(**arguments)
    model_spark = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_pd.fit(dataset_pd)
    model_pl.fit(dataset_pl)
    model_spark.fit(dataset_spark)
    _, rank_spark = model_spark.get_features(ids=dataset_spark.interactions.select(item_col).distinct())
    _, rank_pd = model_pd.get_features(ids=pd.DataFrame(dataset_pd.interactions[item_col].unique(), columns=[item_col]))
    _, rank_pl = model_pl.get_features(ids=dataset_pl.interactions.select(item_col).unique())
    assert rank_spark == rank_pd == rank_pl  # If add new client_models in future, add comparison of dfs
    fake_dataset = pd.DataFrame({"fake_id": [1, 2], "fake_item": [2, 3]})
    assert item_col not in fake_dataset.columns
    with pytest.raises(ValueError, match="missing"):
        model_spark.get_features(ids=fake_dataset)
    with pytest.raises(ValueError, match="missing"):
        model_pd.get_features(ids=fake_dataset)
    with pytest.raises(ValueError, match="missing"):
        model_pl.get_features(ids=fake_dataset)


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments, type_of_impl",
    [(PopRec, {}, "polars"), (PopRec, {}, "pandas"), (PopRec, {}, "spark")],
    ids=["pop_rec_polars", "pop_rec_pandas", "pop_rec_spark"],
)
def test_convertation_not_fitted(base_model, arguments, type_of_impl, datasets):
    model = base_model(**arguments)
    if not model.is_fitted:
        with pytest.raises(NotFittedModelError):
            model.to_spark()
        with pytest.raises(NotFittedModelError):
            model.to_pandas()
        with pytest.raises(NotFittedModelError):
            model.to_polars()
    model.fit(datasets[type_of_impl])
    old_id = id(model)
    if model.is_spark:
        model.to_spark()
    elif model.is_pandas:
        model.to_pandas()
    elif model.is_polars:
        model.to_polars()
    assert id(model) == old_id


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments, type_of_impl",
    [(PopRec, {}, "polars"), (PopRec, {}, "pandas"), (PopRec, {}, "spark")],
    ids=["pop_rec_polars", "pop_rec_pandas", "pop_rec_spark"],
)
def test_convertation_in_same_type(base_model, arguments, type_of_impl, datasets):
    model = base_model(**arguments)
    model.fit(datasets[type_of_impl])
    old_id = id(model)
    if model.is_spark:
        res = model.to_spark()
    elif model.is_pandas:
        res = model.to_pandas()
    elif model.is_polars:
        res = model.to_polars()
    assert id(res) == old_id


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments, type_of_impl",
    [(PopRec, {}, "polars"), (PopRec, {}, "pandas"), (PopRec, {}, "spark")],
    ids=["pop_rec_polars", "pop_rec_pandas", "pop_rec_spark"],
)
def test_get_fit_counts(base_model, arguments, type_of_impl, datasets, fake_fit_items, fake_fit_queries):
    dataset = datasets[type_of_impl]
    model = base_model(**arguments)
    assert not hasattr(model, "_num_items")
    assert not hasattr(model, "_num_queries")
    model.fit_items = fake_fit_items
    model.fit_queries = fake_fit_queries
    assert model.items_count == 7
    assert model.queries_count == 6
    assert not hasattr(model, "_num_items")
    assert not hasattr(model, "_num_queries")
    model.fit(dataset)
    # ---- after fit setter of fit_items not working, because when model is fitted, it use  _num_items / _num_queries --
    model.fit_items = fake_fit_items
    model.fit_queries = fake_fit_queries
    assert hasattr(model, "_num_items")
    assert hasattr(model, "_num_queries")
    assert model.items_count == 3
    assert model.queries_count == 4


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize("type_of_impl", ["pandas", "spark", "polars"])
def test_string_name(base_model, arguments, type_of_impl):
    model = base_model(**arguments)
    assert str(model) == base_model.__name__
    model._impl = model._class_map[type_of_impl]()
    assert str(model._impl) == model._class_map[type_of_impl].__name__


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize("type_of_impl", ["pandas", "spark", "polars"])
def test_clear_cache(base_model, arguments, type_of_impl):
    model = base_model(**arguments)
    model._impl = model._class_map[type_of_impl]()
    if not model.is_spark:
        with pytest.raises(AttributeError, match=r"does not have the '_clear_cache\(\)' method"):
            model._clear_cache()
    else:
        model._clear_cache()


# ----------------- NonPersonalizedRecommenderClient -----------------


class NonPersRec(NonPersonolizedRecommenderClient):
    _class_map = {"spark": _PopRecSpark, "pandas": _PopRecPandas, "polars": _PopRecPolars}

    def _fit(
        self,
        dataset,
    ) -> None:
        pass


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {"add_cold_items": True, "cold_weight": 0.5}), (NonPersRec, {"add_cold_items": True, "cold_weight": 1})],
    ids=["pop_rec", "non_pers_rec"],
)
def test_nonpersonalized_client_valid_init_args(base_model, arguments):
    model = base_model(**arguments)
    for i, val in arguments.items():
        assert model._init_args[i] == val
    assert model._init_when_first_impl_arrived_args["can_predict_cold_items"] is True
    assert model._init_when_first_impl_arrived_args["can_predict_cold_queries"] is True
    assert model._init_when_first_impl_arrived_args["seed"] is None


@pytest.mark.spark
@pytest.mark.parametrize("invalid_weight", [0, -0.1, 1.1])
@pytest.mark.parametrize(
    "base_model",
    [PopRec],
    ids=["pop_rec"],
)
def test_nonpersonalized_client_invalid_cold_weight(base_model, invalid_weight):
    with pytest.raises(ValueError, match="'cold_weight' value should be in interval"):
        base_model(cold_weight=invalid_weight)


@pytest.mark.spark
@pytest.mark.parametrize("invalid_weight", [0, -0.1, 1.1])
@pytest.mark.parametrize(
    "base_model",
    [PopRec],
    ids=["pop_rec"],
)
def test_nonpersonalized_client_invalid_cold_weight_setter(base_model, invalid_weight):
    model = base_model()
    with pytest.raises(AttributeError, match="does not have the 'cold_weight' attribute"):
        model.cold_weight
    with pytest.raises(ValueError, match=r"'cold_weight' value should be float in interval \(0, 1]"):
        model.cold_weight = invalid_weight


@pytest.mark.spark
@pytest.mark.parametrize("valid_weight", [0.00001, 0.9999])
@pytest.mark.parametrize(
    "base_model",
    [PopRec],
    ids=["pop_rec"],
)
def test_nonpersonalized_client_valid_cold_weight_setter(base_model, valid_weight, datasets):
    dataset = datasets["spark"]
    model = base_model()
    model.cold_weight = valid_weight
    assert model._cold_weight == valid_weight
    assert model._init_when_first_impl_arrived_args["cold_weight"] == valid_weight
    model.fit(dataset)
    model.cold_weight = valid_weight
    assert model._impl.cold_weight == valid_weight


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model",
    [PopRec],
    ids=["pop_rec"],
)
def test_nonpersonalized_client_add_cold_items(base_model, datasets):
    model = base_model(add_cold_items=True)
    incorrect_value = "string"
    with pytest.raises(ValueError, match="incorrect type of argument 'value'"):
        model.add_cold_items = incorrect_value
    value = True
    dataset = datasets["spark"]
    model = base_model()
    with pytest.raises(AttributeError, match="does not have the 'add_cold_items' attribute"):
        model.add_cold_items
    model.add_cold_items = value
    assert model._add_cold_items == value
    model.fit(dataset)
    assert model.add_cold_items


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model",
    [PopRec],
    ids=["pop_rec"],
)
def test_nonpersonalized_client_valid_add_cold_items_setter(base_model, datasets):
    value = True
    dataset = datasets["spark"]
    model = base_model()
    model.add_cold_items = value
    assert model._add_cold_items == value
    assert model._init_when_first_impl_arrived_args["add_cold_items"] == value
    model.fit(dataset)
    model.add_cold_items = value
    assert model._impl.add_cold_items == value


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model",
    [PopRec],
    ids=["pop_rec"],
)
def test_nonpersonalized_client_fill(base_model, datasets):
    value = True
    dataset = datasets["spark"]
    model = base_model()
    with pytest.raises(AttributeError, match="does not have the 'fill' attribute"):
        model.fill
    with pytest.raises(AttributeError, match="Can't set to 'fill' value"):
        model.fill = value
    model.fit(dataset)
    assert model.fill is not None
    model.fill = value
    assert model.fill is not None


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model",
    [PopRec],
    ids=["pop_rec"],
)
def test_nonpersonalized_client_sample(base_model, datasets):
    model = base_model(add_cold_items=True)
    dataset = datasets["spark"]
    model = base_model()
    with pytest.raises(AttributeError, match="does not have the 'sample' attribute"):
        model.sample
    model.fit(dataset)
    assert model.sample is not None


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model",
    [PopRec],
    ids=["pop_rec"],
)
def test_nonpersonalized_client_seed(base_model, datasets):
    model = base_model()
    model = base_model()
    with pytest.raises(AttributeError, match="does not have the 'seed' attribute"):
        model.seed
    model._impl = model._class_map["pandas"]()
    model._impl.seed = SEED
    assert model.seed == SEED


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments, type_of_impl",
    [(PopRec, {}, "polars"), (PopRec, {}, "pandas"), (PopRec, {}, "spark")],
    ids=["pop_rec_polars", "pop_rec_pandas", "pop_rec_spark"],
)
def test_nonpersonalized_client_dataframes(base_model, arguments, type_of_impl, datasets):
    model = base_model(**arguments)
    with pytest.raises(AttributeError, match="does not have the '_dataframes' attribute"):
        model._dataframes
    model.fit(datasets[type_of_impl])
    assert model._dataframes is not None


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model",
    [PopRec],
    ids=["pop_rec"],
)
def test_nonpersonalized_client_item_popularity(base_model, datasets, spark):
    dataset_spark = datasets["spark"]
    dataset_pd = datasets["pandas"]
    dataset_pl = datasets["polars"]
    model_spark = base_model()
    model_pd = base_model()
    model_pl = base_model()
    with pytest.raises(AttributeError, match="does not have the 'item_popularity' attribute"):
        model_pd.item_popularity
    with pytest.raises(DataModelMissmatchError):
        model_spark.item_popularity = pd.DataFrame([1, 2, 3])
    with pytest.raises(DataModelMissmatchError):
        model_pd.item_popularity = pl.DataFrame([1, 2, 3])
    with pytest.raises(DataModelMissmatchError):
        model_pl.item_popularity = spark.createDataFrame(pd.DataFrame([1, 2, 3]))
    model_spark.fit(dataset_spark)
    model_pd.fit(dataset_pd)
    model_pl.fit(dataset_pl)
    model_spark.item_popularity = spark.createDataFrame(pd.DataFrame([1, 2, 3]))
    model_pd.item_popularity = pd.DataFrame([1, 2, 3])
    model_pl.item_popularity = pl.DataFrame([1, 2, 3])
    assert isDataFrameEqual(model_pd.item_popularity, pd.DataFrame([1, 2, 3]))
    assert isDataFrameEqual(model_pl.item_popularity, pl.DataFrame([1, 2, 3]))
    assert isDataFrameEqual(model_spark.item_popularity, spark.createDataFrame(pd.DataFrame([1, 2, 3])))


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments, type_of_impl, items",
    [
        (PopRec, {}, "polars", pl.DataFrame({"item_idx": [1, 2]})),
        (PopRec, {}, "pandas", pd.DataFrame({"item_idx": [1, 2]})),
        (PopRec, {}, "spark", convert2spark(pd.DataFrame({"item_idx": [1, 2]}))),
    ],
    ids=["pop_rec_polars", "pop_rec_pandas", "pop_rec_spark"],
)
def test_nonpersonalized_client_get_items_pd(base_model, arguments, type_of_impl, items, datasets):
    model = base_model(**arguments)
    with pytest.raises(AttributeError, match="does not have the 'get_items_pd' function"):
        model.get_items_pd(items)

    model.fit(datasets[type_of_impl])
    if type_of_impl == "pandas":
        with pytest.raises(AttributeError, match="does not have the 'get_items_pd' function"):
            model.get_items_pd(items) is not None
    else:
        assert isinstance(model.get_items_pd(items), pd.DataFrame)


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model",
    [PopRec],
    ids=["pop_rec"],
)
@pytest.mark.parametrize("wrong_implementation", [1, None, {}])
def test_wrong_implementation(base_model, wrong_implementation):
    model = base_model()
    with pytest.raises(ValueError, match="Model can be one of these classes:"):
        model._impl = wrong_implementation
