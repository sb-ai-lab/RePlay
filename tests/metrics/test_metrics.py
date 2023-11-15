import random
import string

import pytest
from pytest import approx

from replay.metrics import (
    MAP,
    MRR,
    NDCG,
    CategoricalDiversity,
    Coverage,
    HitRate,
    Mean,
    Novelty,
    OfflineMetrics,
    PerUser,
    Precision,
    Recall,
    RocAuc,
    Surprisal,
    Unexpectedness,
)
from replay.utils import DataFrameLike, PandasDataFrame, SparkDataFrame

ABS = 1e-5
QUERY_COLUMN = "uid"
ITEM_COLUMN = "iid"
CATEGORY_COLUMN = "cid"
RATING_COLUMN = "scores"
INIT_DICT = {
    "query_column": QUERY_COLUMN,
    "item_column": ITEM_COLUMN,
    "rating_column": RATING_COLUMN,
}
DIVERSITY_DICT = {
    "query_column": QUERY_COLUMN,
    "category_column": CATEGORY_COLUMN,
    "rating_column": RATING_COLUMN,
}


def compute_mean_from_result_per_user(result_per_user):
    mean_result = {}
    for metric_name, user_metrics in result_per_user.items():
        value = 0
        for _, metric_value in user_metrics.items():
            value += metric_value
        value /= max(1, len(user_metrics))
        mean_result[metric_name] = value
    return mean_result


@pytest.mark.spark
@pytest.mark.parametrize(
    "metric",
    [
        MAP,
        MRR,
        NDCG,
        Coverage,
        HitRate,
        Novelty,
        Precision,
        Recall,
        RocAuc,
    ],
)
@pytest.mark.usefixtures("predict_spark", "gt_spark")
def test_metric_with_different_column_names(
    metric, predict_spark: SparkDataFrame, gt_spark: SparkDataFrame
):
    metric_value = metric(topk=[5], **INIT_DICT)(predict_spark, gt_spark)

    def generate_string():
        len = random.randint(1, 10)
        letters = string.ascii_letters
        return "".join(random.choice(letters) for _ in range(len))

    new_query_column = generate_string()
    new_item_column = generate_string()
    new_rating_column = generate_string()
    predict_spark = (
        predict_spark.withColumnRenamed(QUERY_COLUMN, new_query_column)
        .withColumnRenamed(ITEM_COLUMN, new_item_column)
        .withColumnRenamed(RATING_COLUMN, new_rating_column)
    )
    gt_spark = gt_spark.withColumnRenamed(
        QUERY_COLUMN, new_query_column
    ).withColumnRenamed(ITEM_COLUMN, new_item_column)
    new_metric_value = metric(
        topk=[5],
        query_column=new_query_column,
        item_column=new_item_column,
        rating_column=new_rating_column,
    )(predict_spark, gt_spark)
    assert list(metric_value.values()) == approx(
        list(new_metric_value.values()), abs=ABS
    )


@pytest.mark.parametrize(
    "metric, topk, answer",
    [
        (Precision, [3, 5, 10], [0.55555, 0.333333, 0.166666]),
        (HitRate, [3, 5, 10], [1.0, 1.0, 1.0]),
        (MRR, [3, 5, 10], [0.61111, 0.61111, 0.61111]),
        (MAP, [3, 5, 10], [0.35185, 0.21111, 0.198148]),
        (NDCG, [3, 5, 10], [0.48975, 0.35396, 0.34018]),
        (RocAuc, [3, 5, 10], [0.16666, 0.55555, 0.55555]),
    ],
)
@pytest.mark.parametrize(
    "predict_data, gt_data",
    [
        pytest.param("predict_spark", "gt_spark", marks=pytest.mark.spark),
        pytest.param("predict_pd", "gt_pd", marks=pytest.mark.core),
        pytest.param("predict_sorted_dict", "gt_dict", marks=pytest.mark.core),
        pytest.param("predict_unsorted_dict", "gt_dict", marks=pytest.mark.core),
    ],
)
@pytest.mark.parametrize("per_user", [False, True])
def test_metric(metric, topk, answer, predict_data, gt_data, per_user, request):
    predict_data = request.getfixturevalue(predict_data)
    gt_data = request.getfixturevalue(gt_data)

    mode = Mean() if per_user is False else PerUser()
    result = metric(topk, mode=mode, **INIT_DICT)(predict_data, gt_data)
    if per_user:
        result = compute_mean_from_result_per_user(result)

    assert list(result.values()) == approx(answer, abs=ABS)


@pytest.mark.parametrize(
    "topk, answer",
    [
        ([3, 5], [1.0, 0.866666]),
    ],
)
@pytest.mark.parametrize(
    "predict_data",
    [
        pytest.param("predict_pd", marks=pytest.mark.core),
        pytest.param("predict_spark", marks=pytest.mark.spark),
    ],
)
@pytest.mark.parametrize("per_user", [False, True])
def test_diversity_metric(topk, answer, predict_data, per_user, request):
    predict_data = request.getfixturevalue(predict_data)

    def rename_cols(data: DataFrameLike, map):
        if isinstance(data, PandasDataFrame):
            return data.rename(columns=map)
        for _from, _to in map.items():
            data = data.withColumnRenamed(_from, _to)
        return data

    predict_data = rename_cols(predict_data, {ITEM_COLUMN: CATEGORY_COLUMN})
    mode = Mean() if per_user is False else PerUser()
    result = CategoricalDiversity(topk, mode=mode, **DIVERSITY_DICT)(predict_data)
    if per_user:
        result = compute_mean_from_result_per_user(result)
    assert list(result.values()) == approx(answer, abs=ABS)


@pytest.mark.parametrize(
    "topk, answer",
    [
        ([3, 5], [0.9, 1.0]),
    ],
)
@pytest.mark.parametrize(
    "predict_data",
    [
        pytest.param("predict_pd", marks=pytest.mark.core),
        pytest.param("predict_spark", marks=pytest.mark.spark),
    ],
)
def test_coverage_metric(topk, answer, predict_data, request):
    predict_data = request.getfixturevalue(predict_data)
    metric_value = Coverage(topk, **INIT_DICT)(predict_data, predict_data)
    assert list(metric_value.values()) == approx(answer, abs=ABS)


@pytest.mark.parametrize(
    "topk, answer, base_recs, recs",
    [
        pytest.param(5, [0.133333], "predict_pd", "predict_pd", marks=pytest.mark.core),
        pytest.param(5, [0.133333], "predict_spark", "predict_spark", marks=pytest.mark.spark),
        pytest.param(
            [3, 5],
            [0.111111111, 0.133333],
            "base_recs_pd",
            "predict_pd",
            marks=pytest.mark.core,
        ),
        pytest.param(
            [3, 5],
            [0.111111111, 0.133333],
            "base_recs_spark",
            "predict_spark",
            marks=pytest.mark.spark,
        ),
    ],
)
def test_unexpectedness(topk, answer, base_recs, recs, request):
    base_recs = request.getfixturevalue(base_recs)
    recs = request.getfixturevalue(recs)
    metric_value = Unexpectedness(topk, **INIT_DICT)(recs, base_recs)
    assert list(metric_value.values()) == approx(answer, abs=ABS)


@pytest.mark.parametrize(
    "topk, answer",
    [
        ([5, 10], [0.31111, 0.31111]),
    ],
)
@pytest.mark.parametrize(
    "predict_data, gt_data",
    [
        pytest.param("predict_pd", "gt_pd", marks=pytest.mark.core),
        pytest.param("predict_spark", "gt_spark", marks=pytest.mark.spark),
    ],
)
def test_recall(topk, answer, predict_data, gt_data, request):
    predict_data = request.getfixturevalue(predict_data)
    gt_data = request.getfixturevalue(gt_data)
    metric_value = Recall(topk, **INIT_DICT)(predict_data, gt_data)
    assert list(metric_value.values()) == approx(answer, abs=ABS)


@pytest.mark.parametrize(
    "topk, answer",
    [(5, [0.333333])],
)
@pytest.mark.parametrize(
    "predict_data, gt_data",
    [
        pytest.param("predict_pd", "gt_pd", marks=pytest.mark.core),
        pytest.param("predict_spark", "gt_spark", marks=pytest.mark.spark),
    ],
)
def test_precision(topk, answer, predict_data, gt_data, request):
    predict_data = request.getfixturevalue(predict_data)
    gt_data = request.getfixturevalue(gt_data)
    metric_value = Precision(topk=topk, **INIT_DICT)(predict_data, gt_data)
    assert list(metric_value.values()) == approx(answer, abs=ABS)


@pytest.mark.parametrize(
    "topk, answer, recs, train",
    [
        pytest.param([3, 5], [0, 0], "predict_pd", "predict_pd", marks=pytest.mark.core),
        pytest.param([3, 5], [0, 0], "predict_spark", "predict_spark", marks=pytest.mark.spark),
        pytest.param([3, 5], [0.444444, 0.577777], "predict_pd", "gt_pd", marks=pytest.mark.core),
        pytest.param(
            [3, 5], [0.444444, 0.577777], "predict_spark", "gt_spark", marks=pytest.mark.spark
        ),
    ],
)
def test_novelty(topk, answer, recs, train, request):
    recs = request.getfixturevalue(recs)
    train = request.getfixturevalue(train)
    metric_value = Novelty(topk, **INIT_DICT)(recs, train)
    assert list(metric_value.values()) == approx(answer, abs=ABS)


@pytest.mark.parametrize(
    "topk, answer, recs, train",
    [
        pytest.param(
            [3, 5], [0.78969, 0.614294], "predict_pd", "predict_pd", marks=pytest.mark.core
        ),
        pytest.param(
            [3, 5],
            [0.78969, 0.614294],
            "predict_spark",
            "predict_spark",
            marks=pytest.mark.spark,
        ),
        pytest.param(
            [3, 5], [0.719586, 0.698418], "predict_pd", "gt_pd", marks=pytest.mark.core
        ),
        pytest.param(
            [3, 5],
            [0.719586, 0.698418],
            "predict_spark",
            "gt_spark",
            marks=pytest.mark.spark,
        ),
    ],
)
def test_surprisal(topk, answer, recs, train, request):
    recs = request.getfixturevalue(recs)
    train = request.getfixturevalue(train)
    metric_value = Surprisal(topk, **INIT_DICT)(recs, train)
    assert list(metric_value.values()) == approx(answer, abs=ABS)


@pytest.mark.parametrize(
    "metrics, answer",
    [
        (
            [
                Coverage(5),
                Recall(5),
                Precision(5),
                Novelty(5),
            ],
            [
                1.0,
                0.311111,
                0.333333,
                0,
            ],
        ),
        (
            [
                Recall(5),
                Precision(5),
                Novelty(5),
            ],
            [0.31111, 0.333333, 0],
        ),
    ],
)
@pytest.mark.parametrize(
    "predict_data, gt_data, train_data",
    [
        pytest.param(
            "predict_spark", "gt_spark", "predict_spark", marks=pytest.mark.spark
        ),
        pytest.param("predict_pd", "gt_pd", "predict_pd", marks=pytest.mark.core),
        pytest.param(
            "predict_sorted_dict", "gt_dict", "fake_train_dict", marks=pytest.mark.core
        ),
    ],
)
def test_offline_metrics(metrics, answer, predict_data, gt_data, train_data, request):
    predict_data = request.getfixturevalue(predict_data)
    gt_data = request.getfixturevalue(gt_data)
    train_data = request.getfixturevalue(train_data)
    result = OfflineMetrics(metrics, **INIT_DICT)(predict_data, gt_data, train_data)
    assert list(result.values()) == approx(answer, abs=ABS)


@pytest.mark.parametrize(
    "metrics, answer",
    [
        (
            [
                Unexpectedness([2, 5, 10, 20]),
                CategoricalDiversity([2, 5, 10, 20]),
            ],
            [0.16666, 0.133333, 0.566666, 0.783333, 1, 0.86666, 0.43333, 0.21666],
        )
    ],
)
@pytest.mark.parametrize(
    "predict_data, gt_data, base_recs",
    [
        pytest.param(
            "predict_spark", "gt_spark", "base_recs_spark", marks=pytest.mark.spark
        ),
        pytest.param("predict_pd", "gt_pd", "base_recs_pd", marks=pytest.mark.core),
        pytest.param(
            "predict_sorted_dict", "gt_dict", "base_recs_dict", marks=pytest.mark.core
        ),
    ],
)
def test_offline_metrics_unexpectedness_and_diversity(
    metrics, answer, predict_data, gt_data, base_recs, request
):
    predict_data = request.getfixturevalue(predict_data)
    gt_data = request.getfixturevalue(gt_data)
    base_recs = request.getfixturevalue(base_recs)
    result = OfflineMetrics(
        metrics,
        query_column=QUERY_COLUMN,
        category_column=ITEM_COLUMN,
        rating_column=RATING_COLUMN,
        item_column=ITEM_COLUMN,
    )(
        predict_data,
        gt_data,
        predict_data,
        base_recs,
    )
    assert list(result.values()) == approx(answer, abs=ABS)


@pytest.mark.parametrize(
    "cnt_base_recommendations, answer",
    [
        (
            None,
            [],
        ),
        (
            1,
            [0.16666, 0.5666666],
        ),
        (
            2,
            [0.16666, 0.5666666, 0, 0.5666666],
        ),
    ],
)
@pytest.mark.parametrize(
    "predict_data, gt_data, base_recommendations",
    [
        pytest.param(
            "predict_spark", "gt_spark", "base_recs_spark", marks=pytest.mark.spark
        ),
        pytest.param("predict_pd", "gt_pd", "base_recs_pd", marks=pytest.mark.core),
        pytest.param(
            "predict_sorted_dict", "gt_dict", "base_recs_dict", marks=pytest.mark.core
        ),
    ],
)
def test_offline_metrics_unexpectedness_different_base_recs(
    cnt_base_recommendations,
    answer,
    predict_data,
    gt_data,
    base_recommendations,
    request,
):
    predict_data = request.getfixturevalue(predict_data)
    gt_data = request.getfixturevalue(gt_data)
    base_recommendations = request.getfixturevalue(base_recommendations)

    base_recs = None
    if cnt_base_recommendations == 1:
        base_recs = base_recommendations
    elif cnt_base_recommendations == 2:
        base_recs = {"1": base_recommendations, "2": predict_data}
    offline_metrics_instance = OfflineMetrics([Unexpectedness([2, 10])], **INIT_DICT)
    if base_recs is None:
        with pytest.raises(ValueError):
            offline_metrics_instance(
                predict_data, gt_data, base_recommendations=base_recs
            )
    else:
        result = offline_metrics_instance(
            predict_data,
            gt_data,
            base_recommendations=base_recs,
        )
        assert list(result.values()) == approx(answer, abs=ABS)


@pytest.mark.parametrize("metric", [MRR])
@pytest.mark.parametrize(
    "predict_data, gt_data",
    [
        pytest.param("predict_spark", "gt_pd", marks=pytest.mark.spark),
        pytest.param("predict_spark", "gt_dict", marks=pytest.mark.spark),
        pytest.param("predict_pd", "gt_spark", marks=pytest.mark.spark),
        pytest.param("predict_pd", "gt_dict", marks=pytest.mark.core),
        pytest.param("predict_sorted_dict", "gt_spark", marks=pytest.mark.spark),
        pytest.param("predict_sorted_dict", "gt_pd", marks=pytest.mark.core),
    ],
)
def test_check_types(metric, predict_data, gt_data, request):
    predict_data = request.getfixturevalue(predict_data)
    gt_data = request.getfixturevalue(gt_data)
    with pytest.raises(ValueError):
        metric(topk=2)(predict_data, gt_data)


@pytest.mark.core
@pytest.mark.parametrize(
    "metric",
    [MAP, MRR, NDCG, Coverage, CategoricalDiversity, HitRate, Novelty, Precision, Recall, RocAuc],
)
@pytest.mark.parametrize("topk", ["2", ["2", "3"], "['2', '3']"])
def test_topk_instance(metric, topk):
    with pytest.raises(ValueError):
        metric(topk)
