# pylint: skip-file
from typing import Dict, Optional

import pytest
import pandas as pd
from numpy.testing import assert_allclose
from replay.metrics import *
from replay.constants import LOG_SCHEMA, REC_SCHEMA
from replay.session_handler import get_spark_session

from datetime import datetime
from math import log2

from replay.distributions import item_distribution


@pytest.fixture
def one_user():
    df = pd.DataFrame({"user_id": [1], "item_id": [1], "relevance": [1],})
    return df


@pytest.fixture
def two_users():
    df = pd.DataFrame(
        {"user_id": [1, 2], "item_id": [1, 2], "relevance": [1, 1],}
    )
    return df


@pytest.fixture
def quality_metrics():
    return [NDCG(), HitRate(), Precision(), Recall(), MAP(), MRR(), RocAuc()]


@pytest.fixture
def spark():
    return get_spark_session(1, 1)


@pytest.fixture
def recs(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", 3.0],
            ["user1", "item2", 2.0],
            ["user1", "item3", 1.0],
            ["user2", "item1", 3.0],
            ["user2", "item2", 4.0],
            ["user2", "item5", 1.0],
            ["user3", "item1", 5.0],
            ["user3", "item3", 1.0],
            ["user3", "item4", 2.0],
        ],
        schema=REC_SCHEMA,
    )


@pytest.fixture
def recs2(spark):
    return spark.createDataFrame(
        data=[["user1", "item4", 4.0], ["user1", "item5", 5.0]],
        schema=REC_SCHEMA,
    )


@pytest.fixture
def empty_recs(spark):
    return spark.createDataFrame(data=[], schema=REC_SCHEMA,)


@pytest.fixture
def ground_truth_recs(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", datetime(2019, 9, 12), 3.0],
            ["user1", "item5", datetime(2019, 9, 13), 2.0],
            ["user1", "item2", datetime(2019, 9, 17), 1.0],
            ["user2", "item6", datetime(2019, 9, 14), 4.0],
            ["user2", "item1", datetime(2019, 9, 15), 3.0],
            ["user3", "item2", datetime(2019, 9, 15), 3.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def log2(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", datetime(2019, 9, 12), 3.0],
            ["user1", "item5", datetime(2019, 9, 13), 2.0],
            ["user1", "item2", datetime(2019, 9, 17), 1.0],
            ["user2", "item6", datetime(2019, 9, 14), 4.0],
            ["user2", "item1", datetime(2019, 9, 15), 3.0],
            ["user3", "item2", datetime(2019, 9, 15), 3.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", datetime(2019, 8, 22), 4.0],
            ["user1", "item3", datetime(2019, 8, 23), 3.0],
            ["user1", "item2", datetime(2019, 8, 27), 2.0],
            ["user2", "item4", datetime(2019, 8, 24), 3.0],
            ["user2", "item1", datetime(2019, 8, 25), 4.0],
            ["user3", "item2", datetime(2019, 8, 26), 5.0],
            ["user3", "item1", datetime(2019, 8, 26), 5.0],
            ["user3", "item3", datetime(2019, 8, 26), 3.0],
            ["user4", "item2", datetime(2019, 8, 26), 5.0],
            ["user4", "item1", datetime(2019, 8, 26), 5.0],
            ["user4", "item1", datetime(2019, 8, 26), 1.0],
        ],
        schema=LOG_SCHEMA,
    )


def assertDictAlmostEqual(d1: Dict, d2: Dict) -> None:
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        assert_allclose(d1[key], d2[key])


def test_test_is_bigger(quality_metrics, one_user, two_users):
    for metric in quality_metrics:
        assert metric(one_user, two_users, 1) == 0.5, str(metric)


def test_pred_is_bigger(quality_metrics, one_user, two_users):
    for metric in quality_metrics:
        assert metric(two_users, one_user, 1) == 1.0, str(metric)


def test_hit_rate_at_k(recs, ground_truth_recs):
    assertDictAlmostEqual(
        HitRate()(recs, ground_truth_recs, [3, 1]), {3: 2 / 3, 1: 1 / 3},
    )


def test_hit_rate_at_k(recs, ground_truth_recs):
    assertDictAlmostEqual(
        HitRate()(recs, ground_truth_recs, [3, 1]), {3: 2 / 3, 1: 1 / 3},
    )


def test_user_dist(log, recs, ground_truth_recs):
    vals = (
        HitRate()
        .user_distribution(log, recs, ground_truth_recs, 1)["value"]
        .to_list()
    )
    assert_allclose(vals, [0.0, 0.5])


def test_item_dist(log, recs):
    assert_allclose(
        item_distribution(log, recs, 1)["rec_count"].to_list(), [0, 0, 1, 2],
    )


def test_ndcg_at_k(recs, ground_truth_recs):
    pred = [300, 200, 100]
    k_set = [1, 2, 3]
    user_id = 1
    ground_truth = [200, 400]
    ndcg_value = 1 / log2(3) / (1 / log2(2) + 1 / log2(3))
    assert (
        NDCG()._get_metric_value_by_user_all_k(
            k_set, user_id, pred, ground_truth
        )
        == [(1, 0, 1), (1, ndcg_value, 2), (1, ndcg_value, 3)],
    )
    assertDictAlmostEqual(
        NDCG()(recs, ground_truth_recs, [1, 3]),
        {
            1: 1 / 3,
            3: 1
            / 3
            * (
                1
                / (1 / log2(2) + 1 / log2(3) + 1 / log2(4))
                * (1 / log2(2) + 1 / log2(3))
                + 1 / (1 / log2(2) + 1 / log2(3)) * (1 / log2(3))
            ),
        },
    )


def test_precision_at_k(recs, ground_truth_recs):
    assertDictAlmostEqual(
        Precision()(recs, ground_truth_recs, [1, 2, 3]),
        {3: 1 / 3, 1: 1 / 3, 2: 1 / 2},
    )


def test_map_at_k(recs, ground_truth_recs):
    assertDictAlmostEqual(
        MAP()(recs, ground_truth_recs, [1, 3]), {3: 11 / 36, 1: 1 / 3},
    )


def test_recall_at_k(recs, ground_truth_recs):
    assertDictAlmostEqual(
        Recall()(recs, ground_truth_recs, [1, 3]),
        {3: (1 / 2 + 2 / 3) / 3, 1: 1 / 9},
    )


def test_surprisal_at_k(ground_truth_recs, recs2):
    assertDictAlmostEqual(
        Surprisal(ground_truth_recs)(recs2, [1, 2]), {1: 1.0, 2: 1.0}
    )

    assert_allclose(
        Surprisal(ground_truth_recs)(recs, 3),
        5 * (1 - 1 / log2(3)) / 9 + 4 / 9,
    )


def test_coverage(ground_truth_recs, recs, empty_recs):
    coverage = Coverage(recs.union(ground_truth_recs.drop("timestamp")))
    assertDictAlmostEqual(
        coverage(recs, [1, 3, 5]),
        {1: 0.3333333333333333, 3: 0.8333333333333334, 5: 0.8333333333333334,},
    )
    assertDictAlmostEqual(
        coverage(empty_recs, [1, 3, 5]), {1: 0.0, 3: 0.0, 5: 0.0},
    )


def test_bad_coverage(ground_truth_recs, recs):
    assert_allclose(Coverage(ground_truth_recs)(recs, 3), 1.25)


def test_empty_recs(quality_metrics):
    for metric in quality_metrics:
        assert_allclose(
            metric._get_metric_value_by_user(
                k=4, pred=[], ground_truth=[2, 4]
            ),
            0,
            err_msg=metric.__name__,
        )


def test_bad_recs(quality_metrics):
    for metric in quality_metrics:
        assert_allclose(
            metric._get_metric_value_by_user(
                k=4, pred=[1, 3], ground_truth=[2, 4]
            ),
            0,
            err_msg=metric.__name__,
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
            err_msg=metric.__name__,
        )
