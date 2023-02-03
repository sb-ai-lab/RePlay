import logging
import pprint
from typing import List, Sequence, Optional, Union

from pyspark.ml import Estimator, Transformer, PipelineModel, Pipeline
from pyspark.sql.dataframe import DataFrame

from replay.data_preparator import JoinBasedIndexerEstimator, JoinBasedIndexerTransformer
from replay.experiment import Experiment
from replay.metrics import MAP, NDCG, HitRate
from replay._spark_ml_rec.spark_base_rec import SparkBaseRec, SparkBaseRecModelParams, SparkUserItemFeaturesModelParams
from replay._spark_ml_rec.spark_rec import SparkRecModel
from replay.splitters import Splitter

logger = logging.getLogger(__name__)


class SparkTrainTestSplitterAndEvaluator(Estimator, SparkBaseRecModelParams):
    def __init__(self,
                 indexer: Union[JoinBasedIndexerEstimator, JoinBasedIndexerTransformer],
                 splitter: Splitter,
                 models: List[SparkBaseRec],
                 bucketize: bool = True,
                 metrics_k: Sequence[int] = (5, 10, 25),
                 choose_best_by_metric: str = "NDCG",
                 choose_best_by_metric_k: int = 10,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None):
        super().__init__()
        self._indexer = indexer
        self._splitter = splitter
        self._models = models
        self._bucketize = bucketize
        self._choose_best_by_metric = choose_best_by_metric
        self._choose_best_by_metric_k = choose_best_by_metric_k
        self._metrics_k = metrics_k
        self._user_features = user_features
        self._item_features = item_features

    def _fit(self, log: DataFrame) -> Transformer:
        indexer = Pipeline(stages=[self._indexer]).fit(log)
        log = indexer.transform(log)

        user_features = indexer.transform(self._user_features) if self._user_features is not None else None
        item_features = indexer.transform(self._item_features) if self._item_features is not None else None

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
            paramMap = {}

            if isinstance(model, SparkUserItemFeaturesModelParams):
                paramMap[model.userFeatures] = user_features
                paramMap[model.itemFeatures] = item_features

            rec_model: SparkRecModel = model.fit(train, paramMap)
            # single interface with many parameters
            # the wrappers decide for themselves what to call
            recs = rec_model.transform(train, params={
                rec_model.numRecommendations: self.getNumRecommendations(),
                rec_model.filterSeenItems: self.getFilterSeenItems()
            })

            e.add_result(rec_model.name, recs)
            self._rec_models[rec_model.name] = rec_model
            self._log_metrics(rec_model.name, e)

        return PipelineModel(stages=[
            indexer,
            self._rec_models[e.best_result(metric=self._choose_best_by_metric, k=self._choose_best_by_metric_k)]
        ])

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

        print(f"Metrics for {model_name}: ")
        pprint.pprint(metrics)
