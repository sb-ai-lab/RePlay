from typing import List

from pyspark.ml import Estimator, Transformer
from pyspark.sql.dataframe import DataFrame

from replay.experiment import Experiment
from replay.metrics import MAP, NDCG, HitRate
from replay.spark_ml_rec.spark_base_rec import SparkBaseRec, SparkBaseRecModelParams
from replay.spark_ml_rec.spark_rec import SparkRecModel
from replay.splitters import Splitter


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

    def _fit(self, log: DataFrame) -> Transformer:
        train, test = self._splitter.split(log)

        list_metrics = [5, 10, 25, 50, 100, 1000]
        e = Experiment(
            test,
            {
                MAP(): list_metrics,
                NDCG(): list_metrics,
                HitRate(): list_metrics,
            },
        )

        self._rec_models = dict()
        for model in self._models:
            rec_model: SparkRecModel = model.fit(train)
            # single intreface with many parameters
            # the wrappers decide for themselves what to call
            recs = rec_model.transform(test, params={
                SparkBaseRecModelParams.numRecommendations: self.getNumRecommendations(),
                SparkBaseRecModelParams.filterSeenItems: self.getFilterSeenItems()
            })

            e.add_result(rec_model.name, recs)
            self._rec_models[rec_model.name] = rec_model

        return self._rec_models[e.best_result(metric=self._metric, k=self._metric_k)]
