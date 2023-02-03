import logging
import pprint
from typing import List

from pyspark.ml import Estimator, Transformer
from pyspark.sql.dataframe import DataFrame

from replay.experiment import Experiment
from replay.metrics import MAP, NDCG, HitRate
from replay.spark_ml_rec.spark_base_rec import SparkBaseRec, SparkBaseRecModelParams
from replay.spark_ml_rec.spark_rec import SparkRecModel
from replay.splitters import Splitter


logger = logging.getLogger(__name__)


class SparkTrainTestSplitterAndEvaluator(Estimator, SparkBaseRecModelParams):
    def __init__(self,
                 splitter: Splitter,
                 models: List[SparkBaseRec],
                 bucketize: bool = True,
                 metric: str = "NDCG",
                 metric_k: int = 10):
        super().__init__()
        self._splitter = splitter
        self._models = models
        self._bucketize = bucketize
        self._metric = metric
        self._metric_k = metric_k
        self._list_metrics = [5, 10, 25]

    def _fit(self, log: DataFrame) -> Transformer:
        train, test = self._splitter.split(log)

        e = Experiment(
            test,
            {
                MAP(): self._list_metrics,
                NDCG(): self._list_metrics,
                HitRate(): self._list_metrics,
            },
        )

        self._rec_models = dict()
        for model in self._models:
            rec_model: SparkRecModel = model.fit(train)
            # single interface with many parameters
            # the wrappers decide for themselves what to call
            recs = rec_model.transform(test, params={
                SparkBaseRecModelParams.numRecommendations: self.getNumRecommendations(),
                SparkBaseRecModelParams.filterSeenItems: self.getFilterSeenItems()
            })

            e.add_result(rec_model.name, recs)
            self._rec_models[rec_model.name] = rec_model
            self._log_metrics(rec_model.name, e)

        return self._rec_models[e.best_result(metric=self._metric, k=self._metric_k)]

    def _log_metrics(self, model_name: str, e: Experiment):
        metrics = dict()
        for k in self._list_metrics:
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
