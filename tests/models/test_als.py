import numpy as np
import pytest

from replay.models import ALSWrap, AssociationRulesItemRec
from replay.utils import PYSPARK_AVAILABLE
from tests.utils import create_dataset

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf

    from replay.utils.spark_utils import array_mult, horizontal_explode, join_or_return

SEED = 123


def fit_predict_selected(model, train_log, inf_log, user_features, users):
    train_dataset = create_dataset(train_log, user_features)
    pred_dataset = create_dataset(inf_log, user_features)
    model.fit(train_dataset)
    return model.predict(pred_dataset)


@pytest.fixture(scope="class")
def model():
    model = ALSWrap(2, implicit_prefs=False)
    model._seed = 42
    return model


def get_first_level_model_features(
    model, pairs, user_features=None, item_features=None, add_factors_mult=True, prefix=""
):
    users = pairs.select("user_idx").distinct()
    items = pairs.select("item_idx").distinct()
    user_factors, user_vector_len = model._get_features_wrap(users, user_features)
    item_factors, item_vector_len = model._get_features_wrap(items, item_features)

    pairs_with_features = join_or_return(pairs, user_factors, how="left", on="user_idx")
    pairs_with_features = join_or_return(
        pairs_with_features,
        item_factors,
        how="left",
        on="item_idx",
    )

    factors_to_explode = []
    if user_factors is not None:
        pairs_with_features = pairs_with_features.withColumn(
            "user_factors",
            sf.coalesce(
                sf.col("user_factors"),
                sf.array([sf.lit(0.0)] * user_vector_len),
            ),
        )
        factors_to_explode.append(("user_factors", "uf"))

    if item_factors is not None:
        pairs_with_features = pairs_with_features.withColumn(
            "item_factors",
            sf.coalesce(
                sf.col("item_factors"),
                sf.array([sf.lit(0.0)] * item_vector_len),
            ),
        )
        factors_to_explode.append(("item_factors", "if"))

    if model.__str__() == "LightFMWrap":
        pairs_with_features = (
            pairs_with_features.fillna({"user_bias": 0, "item_bias": 0})
            .withColumnRenamed("user_bias", f"{prefix}_user_bias")
            .withColumnRenamed("item_bias", f"{prefix}_item_bias")
        )

    if add_factors_mult and user_factors is not None and item_factors is not None:
        pairs_with_features = pairs_with_features.withColumn(
            "factors_mult",
            array_mult(sf.col("item_factors"), sf.col("user_factors")),
        )
        factors_to_explode.append(("factors_mult", "fm"))

    for col_name, feature_prefix in factors_to_explode:
        col_set = set(pairs_with_features.columns)
        col_set.remove(col_name)
        pairs_with_features = horizontal_explode(
            data_frame=pairs_with_features,
            column_to_explode=col_name,
            other_columns=[sf.col(column) for column in sorted(col_set)],
            prefix=f"{prefix}_{feature_prefix}",
        )

    return pairs_with_features


@pytest.mark.spark
def test_works(log, model):
    dataset = create_dataset(log)
    pred = model.fit_predict(dataset, k=1)
    assert pred.count() == 4


@pytest.mark.spark
def test_diff_feedback_type(log, model):
    dataset = create_dataset(log)
    pred_exp = model.fit_predict(dataset, k=1)
    model.implicit_prefs = True
    pred_imp = model.fit_predict(dataset, k=1)
    assert not np.allclose(
        pred_exp.toPandas().sort_values("user_idx")["relevance"].values,
        pred_imp.toPandas().sort_values("user_idx")["relevance"].values,
    )


@pytest.mark.spark
def test_enrich_with_features(log, model):
    dataset = create_dataset(log.filter(sf.col("user_idx").isin([0, 2])))
    model.fit(dataset)
    res = get_first_level_model_features(model, log.filter(sf.col("user_idx").isin([0, 1])))

    cold_user_and_item = res.filter((sf.col("user_idx") == 1) & (sf.col("item_idx") == 3))
    row_dict = cold_user_and_item.collect()[0].asDict()
    assert row_dict["_if_0"] == row_dict["_uf_0"] == row_dict["_fm_1"] == 0.0

    warm_user_and_item = res.filter((sf.col("user_idx") == 0) & (sf.col("item_idx") == 0))
    row_dict = warm_user_and_item.collect()[0].asDict()
    np.allclose(
        [row_dict["_fm_1"], row_dict["_if_1"] * row_dict["_uf_1"]],
        [4.093189725967505, row_dict["_fm_1"]],
    )

    cold_user_warm_item = res.filter((sf.col("user_idx") == 1) & (sf.col("item_idx") == 0))
    row_dict = cold_user_warm_item.collect()[0].asDict()
    np.allclose(
        [row_dict["_if_1"], row_dict["_if_1"] * row_dict["_uf_1"]],
        [-2.938199281692505, 0],
    )


@pytest.mark.core
def test_init_args(model):
    args = model._init_args

    assert args["rank"] == 2
    assert args["implicit_prefs"] is False
    assert args["seed"] == 42


@pytest.mark.spark
def test_predict_pairs_raises_pairs_format(log):
    model = ALSWrap(seed=SEED)
    with pytest.raises(ValueError, match="pairs must be a dataframe with .*"):
        dataset = create_dataset(log)
        model.fit(dataset)
        model.predict_pairs(log, dataset)


@pytest.mark.spark
@pytest.mark.parametrize("metric", ["absent", None])
def test_nearest_items_raises(log, metric):
    model = AssociationRulesItemRec(session_column="user_idx")
    dataset = create_dataset(log.filter(sf.col("item_idx") != 3))
    model.fit(dataset)
    with pytest.raises(ValueError, match=r"Select one of the valid distance metrics.*"):
        model.get_nearest_items(items=[0, 1], k=2, metric=metric)
    model = ALSWrap()
    dataset = create_dataset(log)
    model.fit(dataset)
    with pytest.raises(ValueError, match=r"Select one of the valid distance metrics.*"):
        model.get_nearest_items(items=[0, 1], k=2, metric=metric)


@pytest.mark.core
@pytest.mark.parametrize(
    "borders",
    [
        {"wrong_name": None},
        {"rank": None},
        {"rank": 2},
        {"rank": [1]},
        {"rank": [1, 2, 3]},
    ],
    ids=[
        "wrong name",
        "None border",
        "int border",
        "border's too short",
        "border's too long",
    ],
)
def test_bad_borders(borders):
    model = ALSWrap()
    with pytest.raises(ValueError):
        model._prepare_param_borders(borders)


@pytest.mark.core
@pytest.mark.parametrize("borders", [None, {"rank": [5, 9]}])
def test_correct_borders(borders):
    model = ALSWrap()
    res = model._prepare_param_borders(borders)
    assert res.keys() == model._search_space.keys()
    assert "rank" in res
    assert isinstance(res["rank"], dict)
    assert res["rank"].keys() == model._search_space["rank"].keys()


@pytest.mark.core
@pytest.mark.parametrize("borders,answer", [(None, True), ({"rank": [-10, -1]}, False)])
def test_param_in_borders(borders, answer):
    model = ALSWrap()
    search_space = model._prepare_param_borders(borders)
    assert model._init_params_in_search_space(search_space) == answer


@pytest.mark.spark
def test_it_works(log):
    model = ALSWrap()
    dataset = create_dataset(log)
    assert model._params_tried() is False
    res = model.optimize(dataset, dataset, k=2, budget=1)
    assert isinstance(res["rank"], int)
    assert model._params_tried() is True
    model.optimize(dataset, dataset, k=2, budget=1)
    assert len(model.study.trials) == 1
    model.optimize(dataset, dataset, k=2, budget=1, new_study=False)
    assert len(model.study.trials) == 2
