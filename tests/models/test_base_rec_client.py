import pytest

from replay.models import PopRec

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
