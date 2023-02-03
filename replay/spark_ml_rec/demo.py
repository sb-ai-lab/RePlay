import functools

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import SQLTransformer, StringIndexer
from rs_datasets import MovieLens

from replay.data_preparator import DataPreparator, JoinBasedIndexerEstimator
from replay.models import ClusterRec, PopRec
from replay.session_handler import get_spark_session
from replay.spark_ml_rec.spark_base_rec import SparkUserItemFeaturesModelParams, \
    SparkBaseRecModel
from replay.spark_ml_rec.spark_rec import SparkRec
from replay.spark_ml_rec.spark_user_rec import SparkUserRec
from replay.spark_ml_rec.splitter import SparkTrainTestSplitterAndEvaluator
from replay.splitters import UserSplitter
from replay.utils import convert2spark

HNSWLIB_PARAMS = {
    "space": "ip",
    "M": 100,
    "efS": 2000,
    "efC": 2000,
    "post": 0,
    "index_path": "/tmp/hnswlib_index_{spark_app_id}",
    "build_index_on": "executor"
}
METRICS_K = [5, 10, 25]

spark = get_spark_session()

ds = MovieLens('100k')

# prepare log
log = convert2spark(ds.ratings).withColumnRenamed("rating", "relevance")

# prepare user_features
user_features = convert2spark(ds.users)
user_features = Pipeline(stages=[
    StringIndexer(
        inputCols=["age", "occupation", "zip_code"],
        outputCols=["age_idx", "occupation_idx", "zip_code_idx"]
    ),
    SQLTransformer(
        statement="SELECT user_id, INT(gender) as gender, age_idx, occupation_idx, zip_code_idx  "
                  "FROM __THIS__"
    )
]).fit(user_features).transform(user_features).cache()

user_features.count()

indexer = JoinBasedIndexerEstimator(user_col="user_id", item_col="item_id").fit(log)

model = Pipeline(stages=[
    DataPreparator(columns_mapping={"user_id": "user_id", "item_id": "item_id", "relevance": "relevance"}),
    SparkTrainTestSplitterAndEvaluator(
        indexer=indexer,
        splitter=UserSplitter(item_test_size=0.2, shuffle=True, drop_cold_users=True, drop_cold_items=True, seed=42),
        models=[
            SparkRec(model=PopRec()),
            # SparkRec(model=Word2VecRec()),
            # SparkRec(model=ALSWrap(hnswlib_params=HNSWLIB_PARAMS)),
            SparkUserRec(model=ClusterRec(), transient_user_features=True)
        ],
        metrics_k=METRICS_K,
        user_features=user_features
    )]).fit(log)

rec_model_path = "/tmp/rec_pipe"
model.write().overwrite().save(rec_model_path)
rec_model = PipelineModel.load(rec_model_path)


models = [
    model
    for stage in rec_model.stages if isinstance(stage, PipelineModel)
    for model in stage.stages if isinstance(model, SparkBaseRecModel)
]

dicts = [{
    model.filterSeenItems: True,
    model.numRecommendations: 100,
    **({model.userFeatures: indexer.transform(user_features)} if isinstance(model, SparkUserItemFeaturesModelParams) else dict())
} for model in models]

recs = rec_model.transform(log, params=functools.reduce(lambda agg, d: agg.update(d), dicts))

recs.write.mode('overwrite').format('noop').save()
