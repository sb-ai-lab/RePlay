import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Set

import numpy as np

from replay.utils import TORCH_AVAILABLE, PandasDataFrame

if TORCH_AVAILABLE:
    import torch

MetricName = Literal[
    "recall",
    "precision",
    "ndcg",
    "map",
    "mrr",
    "novelty",
    "coverage",
]

DEFAULT_METRICS: List[MetricName] = [
    "map",
    "ndcg",
    "recall",
]

DEFAULT_KS: List[int] = [1, 5, 10, 20]


# pylint: disable=too-many-instance-attributes
@dataclass
class _MetricRequirements:
    """
    Stores description of metrics which need to be computed
    """

    top_k: List[int]
    need_recall: bool
    need_precision: bool
    need_ndcg: bool
    need_map: bool
    need_mrr: bool
    need_novelty: bool
    need_coverage: bool

    def __post_init__(self) -> None:
        metrics = []

        for k in self.top_k:
            if self.need_recall:
                metrics.append(f"recall@{k}")

            if self.need_precision:
                metrics.append(f"precision@{k}")

            if self.need_ndcg:
                metrics.append(f"ndcg@{k}")

            if self.need_map:
                metrics.append(f"map@{k}")

            if self.need_mrr:
                metrics.append(f"mrr@{k}")

            if self.need_novelty:
                metrics.append(f"novelty@{k}")

        self._metric_names = metrics

    @property
    def metric_names(self) -> List[str]:
        """
        Getting metric names
        """
        return self._metric_names

    @classmethod
    def from_metrics(cls, metrics: Set[str], top_k: List[int]) -> "_MetricRequirements":
        """
        Creating a class based on a given list of metrics and K values
        """
        return _MetricRequirements(
            top_k=top_k,
            need_recall="recall" in metrics,
            need_precision="precision" in metrics,
            need_ndcg="ndcg" in metrics,
            need_map="map" in metrics,
            need_mrr="mrr" in metrics,
            need_novelty="novelty" in metrics,
            need_coverage="coverage" in metrics,
        )


class _CoverageHelper:
    """
    Computes coverage metric over multiple batches
    """

    def __init__(self, top_k: List[int], item_count: Optional[int]) -> None:
        """
        :param top_k: (list): Consider the highest k scores in the ranking.
        :param item_count: (optional, int): the total number of items in the dataset.
        """
        self._top_k = top_k
        self._item_count = item_count
        self.reset()

    def reset(self) -> None:
        """
        Reload the metric counter
        """
        self._train_hist = torch.zeros(self.item_count)
        self._pred_hist: Dict[int, torch.Tensor] = {k: torch.zeros(self.item_count) for k in self._top_k}

    # pylint: disable=attribute-defined-outside-init
    def _ensure_hists_on_device(self, device: torch.device) -> None:
        self._train_hist = self._train_hist.to(device)
        for k in self._top_k:
            self._pred_hist[k] = self._pred_hist[k].to(device)

    def add_prediction(self, predictions: torch.LongTensor) -> None:
        """
        Add a batch with predictions.

        :param predictions: (torch.LongTensor): A batch with the same number of recommendations for each user.
        """
        self._ensure_hists_on_device(predictions.device)
        for k in self._top_k:
            self._pred_hist[k] += torch.histc(
                predictions[:, :k].float(), bins=self.item_count, min=0, max=self.item_count - 1
            )

    def add_train(self, train: torch.LongTensor) -> None:
        """
        Add a training set batch.

        :param train: (optional, int): A batch corresponding to the train set for each user.
            If users have a train set of different sizes then you need to do the padding using -2.
        """
        self._ensure_hists_on_device(train.device)
        flatten_train = train.flatten()
        filtered_train = torch.masked_select(flatten_train, flatten_train != -2)
        self._train_hist += torch.histc(filtered_train.float(), bins=self.item_count, min=0, max=self.item_count - 1)

    def get_metrics(self) -> Mapping[str, float]:
        """
        Getting calculated metrics.
        """
        train_item_count = (self._train_hist > 0).sum().item()
        result = {}
        for k in self._top_k:
            intersection = (self._train_hist > 0) & (self._pred_hist[k] > 0)
            result[f"coverage@{k}"] = intersection.sum().item() / train_item_count
        return result

    @property
    def item_count(self) -> int:
        """
        The number of items in the training dataset
        """
        assert self._item_count
        return self._item_count

    @item_count.setter
    def item_count(self, value: int) -> None:
        self._item_count = value


class _MetricBuilder(abc.ABC):
    """
    Interface of the metrics builder, which is used to combine
    predictions over multiple batches to calculate single result
    """

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reload the metric counter
        """

    @abc.abstractmethod
    def add_prediction(self, predictions: Any, ground_truth: Any, train: Any = None) -> None:
        """
        Add a batch with predictions
        """

    @abc.abstractmethod
    def get_metrics(self) -> Mapping[str, float]:
        """
        Getting calculated metrics
        """


# pylint: disable=too-many-instance-attributes
class TorchMetricsBuilder(_MetricBuilder):
    """
    Computes specified metrics over multiple batches
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        metrics: List[MetricName] = DEFAULT_METRICS,
        top_k: Optional[List[int]] = DEFAULT_KS,
        item_count: Optional[int] = None,
    ) -> None:
        """
        :param metrics: (list[MetricName]): Names of metrics to calculate.
            Default: `["map", "ndcg", "recall"]`.
        :param top_k: (list): Consider the highest k scores in the ranking.
            Default: `[1, 5, 10, 20]`.
        :param item_count: (optional, int): the total number of items in the dataset.
            You can omit this parameter if you don't need to calculate the Coverage metric.
        """
        self._mr = _MetricRequirements.from_metrics(
            set(metrics),
            sorted(set(top_k)),
        )
        if self._mr.need_ndcg:
            self._ndcg_weights: torch.Tensor
            self._ndcg_idcgs: torch.Tensor
            self._reserve_ndcg_constants()
        if self._mr.need_map:
            self._map_weights: torch.Tensor
            self._reserve_map_constants()
        self._item_count = item_count
        self._coverage_helper = _CoverageHelper(top_k=self._mr.top_k, item_count=item_count)
        self.reset()

    @property
    def max_k(self) -> int:
        """
        Maximum K for calculating metrics.
        """
        return max(self._mr.top_k)

    @property
    def item_count(self) -> int:
        """
        The number of items in the training dataset.
        """
        assert self._item_count
        return self._item_count

    @item_count.setter
    def item_count(self, value: int) -> None:
        self._item_count = value

    def reset(self) -> None:
        """
        Reload the metric counter
        """
        self._metric_sum = np.zeros(len(self._mr.metric_names), dtype=np.float64)
        self._prediction_counter = 0
        if self._mr.need_coverage:
            self._coverage_helper.item_count = self.item_count
            self._coverage_helper.reset()

    def _ensure_constants_on_device(self, device: torch.device) -> None:
        if self._mr.need_ndcg:
            self._ndcg_weights = self._ndcg_weights.to(device)
            self._ndcg_idcgs = self._ndcg_idcgs.to(device)
        if self._mr.need_map:
            self._map_weights = self._map_weights.to(device)

    def add_prediction(
        self,
        predictions: torch.LongTensor,
        ground_truth: torch.LongTensor,
        train: Optional[torch.LongTensor] = None,
    ) -> None:
        """
        Add a batch with predictions, ground truth and train set to calculate the metrics.

        :param predictions: (torch.LongTensor): A batch with the same number of recommendations for each user.
        :param ground_truth: (torch.LongTensor): A batch corresponding to the test set for each user.
            If users have a test set of different sizes then you need to do the padding using -1.
        :param train: (optional, int): A batch corresponding to the train set for each user.
            If users have a train set of different sizes then you need to do the padding using -2.
            You can omit this parameter if you don't need to calculate the unseen metrics.
        """
        self._ensure_constants_on_device(predictions.device)
        metrics_sum = np.array(self._compute_metrics_sum(predictions, ground_truth, train), dtype=np.float64)
        if self._mr.need_coverage:
            self._coverage_helper.add_prediction(predictions)
            assert train is not None
            self._coverage_helper.add_train(train)
        self._prediction_counter += len(predictions)
        self._metric_sum += metrics_sum

    def get_metrics(self) -> Mapping[str, float]:
        """
        Getting calculated metrics.
        """
        assert self._prediction_counter > 0
        metrics = self._metric_sum / self._prediction_counter
        result = dict(zip(self._mr.metric_names, metrics))
        if self._mr.need_coverage:
            result.update(self._coverage_helper.get_metrics())
        return result

    def _compute_recall(self, hits: torch.Tensor, ground_truth_count: torch.Tensor) -> float:
        recall = hits.sum(1) / ground_truth_count
        return recall.sum().item()

    def _compute_precision(self, hits: torch.Tensor, k: int) -> float:
        precision = hits.sum(1) / k
        return precision.sum().item()

    def _compute_ndcg(self, hits: torch.Tensor, ground_truth_count_at_k: torch.Tensor, k: int) -> float:
        dcg = (hits * self._ndcg_weights[:k]).sum(1)
        idcg = self._ndcg_idcgs[ground_truth_count_at_k]
        ndcg = dcg / idcg
        return ndcg.sum().item()

    def _compute_map(self, hits: torch.Tensor, ground_truth_count_at_k: torch.Tensor, k: int) -> float:
        hits_cumsum = hits * hits.cumsum(1)
        hits_cumsum_discounted = (hits_cumsum * self._map_weights[:k]).sum(1)
        average_precision = hits_cumsum_discounted / ground_truth_count_at_k
        return average_precision.sum().item()

    def _compute_mrr(self, hits: torch.Tensor) -> float:
        indexed_hits = hits * torch.arange(hits.shape[1], 0, -1, device=hits.device)
        vals, indices = torch.max(indexed_hits, dim=1)
        fixed_indices = indices.masked_fill(vals == 0, -2) + 1
        mrr = (1 / fixed_indices).clamp(min=0)
        return mrr.sum().item()

    def _compute_novelty(self, train_hits: torch.Tensor, k: int) -> float:
        novelty = (~train_hits).sum(1) / k
        return novelty.sum().item()

    def _compute_metrics_sum(
        self, predictions: torch.LongTensor, ground_truth: torch.LongTensor, train: Optional[torch.LongTensor]
    ) -> List[float]:
        result: List[float] = []

        # Getting a tensor of the same size as predictions
        # The tensor contains information about whether the item from the prediction is present in the test set
        item_hits = (predictions.unsqueeze(1) == ground_truth.unsqueeze(-1)).any(dim=1)
        if self._mr.need_novelty:
            assert train is not None
            train_hits = (predictions.unsqueeze(1) == train.unsqueeze(-1)).any(dim=1)

        hits = item_hits[:, : self.max_k].float()

        # must be > 0, otherwise it may lead to NaNs in metrics
        gt_count = (ground_truth >= 0).sum(1).clamp(min=1)

        for k in self._mr.top_k:
            hits_at_k = hits[:, :k]
            gt_count_at_k = gt_count.clamp(max=k)

            if self._mr.need_recall:
                result.append(self._compute_recall(hits_at_k, gt_count))

            if self._mr.need_precision:
                result.append(self._compute_precision(hits_at_k, k))

            if self._mr.need_ndcg:
                result.append(self._compute_ndcg(hits_at_k, gt_count_at_k, k))

            if self._mr.need_map:
                result.append(self._compute_map(hits_at_k, gt_count_at_k, k))

            if self._mr.need_mrr:
                result.append(self._compute_mrr(hits_at_k))

            if self._mr.need_novelty:
                train_hits_at_k = train_hits[:, :k]
                result.append(self._compute_novelty(train_hits_at_k, k))

        return result

    def _reserve_ndcg_constants(self) -> None:
        position = torch.arange(2, 2 + self.max_k).float()
        self._ndcg_weights = 1 / torch.log2(position)
        self._ndcg_idcgs = torch.cat(
            [
                torch.tensor([0], dtype=torch.float32),
                self._ndcg_weights.cumsum(dim=0),
            ]
        )

    def _reserve_map_constants(self) -> None:
        self._map_weights = 1.0 / torch.arange(1, 1 + self.max_k).float()


def metrics_to_df(metrics: Mapping[str, float]) -> PandasDataFrame:
    """
    Converting metrics to Pandas DataFrame
    """
    metrics_df = PandasDataFrame(metrics.items(), columns=["m", "v"])

    metric_name_and_k = metrics_df["m"].str.split("@", expand=True)
    metrics_df["metric"] = metric_name_and_k[0]
    metrics_df["k"] = [int(k) for k in metric_name_and_k[1]]

    pivoted_metrics = metrics_df.pivot(index="metric", columns="k", values="v")
    pivoted_metrics.index.name = None

    pivoted_metrics.sort_index(axis=0, inplace=True)
    pivoted_metrics.sort_index(axis=1, inplace=True)

    return pivoted_metrics
