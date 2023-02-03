import logging
import pprint
from typing import List, cast, Sequence

from pyspark.ml import Estimator, Transformer
from pyspark.sql.dataframe import DataFrame

from replay.experiment import Experiment
from replay.metrics import MAP, NDCG, HitRate
from replay.models import Recommender
from replay.spark_ml_rec.spark_base_rec import SparkBaseRec, SparkBaseRecModelParams
from replay.spark_ml_rec.spark_rec import SparkRecModel
from replay.splitters import Splitter


logger = logging.getLogger(__name__)


class SparkTrainTestSplitterAndEvaluator(Estimator, SparkBaseRecModelParams):
    def __init__(self,
                 splitter: Splitter,
                 models: List[SparkBaseRec],
                 bucketize: bool = True,
                 metrics_k: Sequence[int] = (5, 10, 25),
                 choose_best_by_metric: str = "NDCG",
                 choose_best_by_metric_k: int = 10):
        super().__init__()
        self._splitter = splitter
        self._models = models
        self._bucketize = bucketize
        self._choose_best_by_metric = choose_best_by_metric
        self._choose_best_by_metric_k = choose_best_by_metric_k
        self._metrics_k = metrics_k

    def _fit(self, log: DataFrame) -> Transformer:
        train, test = self._splitter.split(log)

        e = Experiment(
            test,
            {
                MAP(): self._metrics_k,
                NDCG(): self._metrics_k,
                HitRate(): self._metrics_k,
            },
        )

        self._rec_models = dict()
        for model in self._models:
            rec_model: SparkRecModel = model.fit(train)
            # single interface with many parameters
            # the wrappers decide for themselves what to call
            recs = rec_model.transform(train, params={
                SparkBaseRecModelParams.numRecommendations: self.getNumRecommendations(),
                SparkBaseRecModelParams.filterSeenItems: self.getFilterSeenItems()
            })

            e.add_result(rec_model.name, recs)
            self._rec_models[rec_model.name] = rec_model
            self._log_metrics(rec_model.name, e)

        return self._rec_models[e.best_result(metric=self._choose_best_by_metric, k=self._choose_best_by_metric_k)]

    def _log_metrics(self, model_name: str, e: Experiment):
        metrics = dict()
        for k in self._metrics_k:
            metrics["NDCG.{}".format(k)] = e.results.at[
                model_name, "NDCG@{}".format(k)
            ]
            metrics["MAP.{}".format(k)] = e.results.at[
                model_name, "MAP@{}".format(k)
            ]
            metrics["HitRate.{}".format(k)] = e.results.at[
                model_name, "HitRate@{}".format(k)
            ]

        print("Metrics: ")
        pprint.pprint(metrics)
