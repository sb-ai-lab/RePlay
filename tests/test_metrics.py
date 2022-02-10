# pylint: skip-file

import numpy as np
from replay.metrics import *

from replay.distributions import item_distribution
from replay.metrics.base_metric import sorter
from tests.utils import *


@pytest.fixture
def one_user():
    df = pd.DataFrame({"user_idx": [1], "item_idx": [1], "relevance": [1]})
    return df


@pytest.fixture
def two_users():
    df = pd.DataFrame(
        {"user_idx": [1, 2], "item_idx": [1, 2], "relevance": [1, 1]}
    )
    return df


@pytest.fixture
def recs(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, 3.0],
            [0, 1, 2.0],
            [0, 2, 1.0],
            [1, 0, 3.0],
            [1, 1, 4.0],
            [1, 4, 1.0],
            [2, 0, 5.0],
            [2, 2, 1.0],
            [2, 3, 2.0],
        ],
        schema=REC_SCHEMA,
    )


@pytest.fixture
def recs2(spark):
    return spark.createDataFrame(
        data=[[0, 3, 4.0], [0, 4, 5.0]], schema=REC_SCHEMA,
    )


@pytest.fixture
def empty_recs(spark):
    return spark.createDataFrame(data=[], schema=REC_SCHEMA,)


@pytest.fixture
def true(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, datetime(2019, 9, 12), 3.0],
            [0, 4, datetime(2019, 9, 13), 2.0],
            [0, 1, datetime(2019, 9, 17), 1.0],
            [1, 5, datetime(2019, 9, 14), 4.0],
            [1, 0, datetime(2019, 9, 15), 3.0],
            [2, 1, datetime(2019, 9, 15), 3.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def quality_metrics():
    return [NDCG(), HitRate(), Precision(), Recall(), MAP(), MRR(), RocAuc()]


@pytest.fixture
def duplicate_recs(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, 3.0],
            [0, 1, 2.0],
            [0, 2, 1.0],
            [0, 0, 3.0],
            [1, 0, 3.0],
            [1, 1, 4.0],
            [1, 4, 1.0],
            [1, 1, 2.0],
            [2, 0, 5.0],
            [2, 2, 1.0],
            [2, 3, 2.0],
        ],
        schema=REC_SCHEMA,
    )


def test_test_is_bigger(quality_metrics, one_user, two_users):
    for metric in quality_metrics:
        assert metric(one_user, two_users, 1) == 0.5, str(metric)


def test_pred_is_bigger(quality_metrics, one_user, two_users):
    for metric in quality_metrics:
        assert metric(two_users, one_user, 1) == 1.0, str(metric)


def test_hit_rate_at_k(recs, true):
    assertDictAlmostEqual(
        HitRate()(recs, true, [3, 1]), {3: 2 / 3, 1: 1 / 3},
    )


def test_user_dist(log, recs, true):
    vals = HitRate().user_distribution(log, recs, true, 1)["value"].to_list()
    assert_allclose(vals, [0.0, 0.5])


def test_item_dist(log, recs):
    assert_allclose(
        item_distribution(log, recs, 1)["rec_count"].to_list(), [0, 0, 1, 2],
    )


def test_ndcg_at_k(recs, true):
    assertDictAlmostEqual(
        NDCG()(recs, true, [1, 3]),
        {
            1: 1 / 3,
            3: 1
            / 3
            * (
                1
                / (1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4))
                * (1 / np.log2(2) + 1 / np.log2(3))
                + 1 / (1 / np.log2(2) + 1 / np.log2(3)) * (1 / np.log2(3))
            ),
        },
    )


def test_precision_at_k(recs, true):
    assertDictAlmostEqual(
        Precision()(recs, true, [1, 2, 3]), {3: 1 / 3, 1: 1 / 3, 2: 1 / 2},
    )


def test_map_at_k(recs, true):
    assertDictAlmostEqual(
        MAP()(recs, true, [1, 3]), {3: 11 / 36, 1: 1 / 3},
    )


def test_recall_at_k(recs, true):
    assertDictAlmostEqual(
        Recall()(recs, true, [1, 3]), {3: (1 / 2 + 2 / 3) / 3, 1: 1 / 9},
    )


def test_surprisal_at_k(true, recs, recs2):
    assertDictAlmostEqual(Surprisal(true)(recs2, [1, 2]), {1: 1.0, 2: 1.0})

    assert_allclose(
        Surprisal(true)(recs, 3), 5 * (1 - 1 / np.log2(3)) / 9 + 4 / 9,
    )


def test_unexpectedness_at_k(true, recs, recs2):
    assert Unexpectedness._get_metric_value_by_user(2, (), (2, 3)) == 0
    assert Unexpectedness._get_metric_value_by_user(2, (1, 2), (1,)) == 0.5


def test_coverage(true, recs, empty_recs):
    coverage = Coverage(recs.union(true.drop("timestamp")))
    assertDictAlmostEqual(
        coverage(recs, [1, 3, 5]),
        {1: 0.3333333333333333, 3: 0.8333333333333334, 5: 0.8333333333333334},
    )
    assertDictAlmostEqual(
        coverage(empty_recs, [1, 3, 5]), {1: 0.0, 3: 0.0, 5: 0.0},
    )


def test_bad_coverage(true, recs):
    assert_allclose(Coverage(true)(recs, 3), 1.25)


def test_empty_recs(quality_metrics):
    for metric in quality_metrics:
        assert_allclose(
            metric._get_metric_value_by_user(
                k=4, pred=[], ground_truth=[2, 4]
            ),
            0,
            err_msg=str(metric),
        )


def test_bad_recs(quality_metrics):
    for metric in quality_metrics:
        assert_allclose(
            metric._get_metric_value_by_user(
                k=4, pred=[1, 3], ground_truth=[2, 4]
            ),
            0,
            err_msg=str(metric),
        )


def test_not_full_recs(quality_metrics):
    for metric in quality_metrics:
        assert_allclose(
            metric._get_metric_value_by_user(
                k=4, pred=[4, 1, 2], ground_truth=[2, 4]
            ),
            metric._get_metric_value_by_user(
                k=3, pred=[4, 1, 2], ground_truth=[2, 4]
            ),
            err_msg=str(metric),
        )


def test_duplicate_recs(quality_metrics, duplicate_recs, recs, true):
    for metric in quality_metrics:
        assert_allclose(
            metric(k=4, recommendations=duplicate_recs, ground_truth=true),
            metric(k=4, recommendations=recs, ground_truth=true),
            err_msg=str(metric),
        )


def test_sorter():
    result = sorter(((1, 2), (2, 3), (3, 2)))
    assert result == [2, 3]


def test_sorter_index():
    result = sorter([(1, 2, 3), (2, 3, 4), (3, 3, 5)], index=2)
    assert result == [5, 3]
