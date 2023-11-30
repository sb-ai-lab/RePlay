import numpy as np
import pytest
from pytest import approx

from replay.metrics import MRR, Novelty, OfflineMetrics, Precision, Recall
from replay.utils import TORCH_AVAILABLE, PandasDataFrame

if TORCH_AVAILABLE:
    from replay.metrics.torch_metrics_builder import TorchMetricsBuilder

torch = pytest.importorskip("torch")

TOP_K = [1, 5, 10, 20]
ABS = 1e-5
QUERY_COLUMN = "user_idx"
ITEM_COLUMN = "item_idx"
RATING_COLUMN = "relevance"
COLUMNS = {
    "query_column": QUERY_COLUMN,
    "item_column": ITEM_COLUMN,
    "rating_column": RATING_COLUMN,
}


def compute_evaluation_metrics(metric_list, train, test, recs):
    result = OfflineMetrics(metric_list, **COLUMNS)(recs, test, train)
    lower_result = {}
    for k, v in result.items():
        lower_result[k.lower()] = v
    return lower_result


def _groupby_items(df: PandasDataFrame):
    return df.groupby(QUERY_COLUMN)[ITEM_COLUMN].apply(list)


def _groupby_recs_items(df: PandasDataFrame):
    return df.sort_values(by=RATING_COLUMN, ascending=False).groupby(QUERY_COLUMN)[ITEM_COLUMN].apply(list)


def _build_tensor_from_grouped_items(grouped_items, pad_value) -> torch.Tensor:
    max_shape = 0
    for i in grouped_items:
        max_shape = max(max_shape, len(i))

    items = []
    for i in grouped_items:
        items.extend(i)
        size = len(i)
        if size != max_shape:
            items.extend([pad_value for _ in range(max_shape - size)])

    return torch.from_numpy(np.array(items, dtype=np.int64).reshape(-1, max_shape))


def _convert_recs_to_tensor(recs: PandasDataFrame) -> torch.Tensor:
    return _build_tensor_from_grouped_items(_groupby_recs_items(recs), -3)


def _convert_data_to_tensor(gt: PandasDataFrame, pad_value: int) -> torch.Tensor:
    return _build_tensor_from_grouped_items(_groupby_items(gt), pad_value)


def _compute_unseen_ground_truth(ground_truth: torch.LongTensor, train: torch.LongTensor) -> torch.LongTensor:
    unseen_ground_truth = torch.zeros_like(ground_truth)
    for i in range(ground_truth.shape[0]):
        mask = (train[i].unsqueeze(0) == ground_truth[i].unsqueeze(-1)).any(dim=1)
        unseen_ground_truth[i] = ground_truth[i].masked_fill(mask, -1)
    return unseen_ground_truth


def compute_builder_metrics(metric_list, train, test, recs, unseen_flag: bool):
    builder = TorchMetricsBuilder(metric_list, top_k=TOP_K, item_count=3706)
    builder.reset()
    tensor_predictions = _convert_recs_to_tensor(recs)
    tensor_ground_truth = _convert_data_to_tensor(test, -1)
    tensor_train = _convert_data_to_tensor(train, -2)
    if unseen_flag:
        tensor_ground_truth = _compute_unseen_ground_truth(tensor_ground_truth, tensor_train)
    builder.add_prediction(predictions=tensor_predictions, ground_truth=tensor_ground_truth, train=tensor_train)
    return builder.get_metrics()


@pytest.mark.spark
@pytest.mark.torch
@pytest.mark.usefixtures("random_train_test_recs")
def test_seen_metrics(random_train_test_recs):
    train, test, recs = random_train_test_recs

    evaluation_metrics = compute_evaluation_metrics(
        [
            Recall(TOP_K, **COLUMNS),
            Precision(TOP_K, **COLUMNS),
            MRR(TOP_K, **COLUMNS),
            Novelty(TOP_K, **COLUMNS),
        ],
        train,
        test,
        recs,
    )
    builder_metrics = compute_builder_metrics(
        [
            "recall",
            "precision",
            "ndcg",
            "map",
            "mrr",
            "novelty",
            "coverage",
        ],
        train,
        test,
        recs,
        False,
    )
    for metric_name, evaluation_value in evaluation_metrics.items():
        builder_value = builder_metrics[metric_name]
        assert evaluation_value == approx(
            builder_value, abs=ABS
        ), f"metric = {metric_name}, evaluation = {evaluation_value}, builder = {builder_value}"


@pytest.mark.spark
@pytest.mark.torch
@pytest.mark.usefixtures("random_train_test_recs")
def test_unseen_metrics(random_train_test_recs):
    train, test, recs = random_train_test_recs

    merged = test.merge(
        train,
        how="outer",
        on=["user_idx", "item_idx"],
        indicator=True
    )
    test_without_train = merged[merged['_merge'] == 'left_only'].drop("_merge", axis=1)

    evaluation_metrics = compute_evaluation_metrics(
        [
            Recall(TOP_K, **COLUMNS),
            Precision(TOP_K, **COLUMNS),
            MRR(TOP_K, **COLUMNS),
        ],
        train,
        test_without_train,
        recs,
    )
    builder_metrics = compute_builder_metrics(
        [
            "recall",
            "precision",
            "mrr",
        ],
        train,
        test_without_train,
        recs,
        True,
    )

    for metric_name, evaluation_value in evaluation_metrics.items():
        builder_value = builder_metrics[metric_name]
        assert evaluation_value == approx(
            builder_value, abs=ABS
        ), f"metric = {metric_name}, evaluation = {evaluation_value}, builder = {builder_value}"
