from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import SparkSession
from rs_datasets import MovieLens

from replay.data_preparator import DataPreparator
from replay.models import Word2VecRec, ALSWrap, ClusterRec
from replay.spark_ml_rec.data_preparator import SparkIndexer
from replay.spark_ml_rec.spark_base_rec import SparkBaseRecModelParams
from replay.spark_ml_rec.spark_rec import SparkRec
from replay.spark_ml_rec.spark_user_rec import SparkUserRec, SparkUserRecModelParams
from replay.spark_ml_rec.splitter import SparkTrainTestSplitterAndEvaluator
from replay.splitters import UserSplitter, DateSplitter
from replay.utils import convert2spark

spark = SparkSession.builder.getOrCreate()

ds = MovieLens('100k')

log = convert2spark(ds.ratings)
user_features = convert2spark(ds.users)

log_train, log_test = DateSplitter(test_start=0.2, date_col="timestamp")

pipe = Pipeline([
    DataPreparator(columns_mapping={"user_id": "user", "item_id": "item_id", "relevance": "rel"}),
    SQLTransformer(statement="SELECT user_id AS user_idx, item_id AS item_idx, relevance, timestamp FROM __THIS__"),
    SparkIndexer(),
    SparkTrainTestSplitterAndEvaluator(
        splitter=UserSplitter(item_test_size=0.2, shuffle=True, drop_cold_users=True, drop_cold_items=True, seed=42),
        models=[
            SparkRec(model=Word2VecRec()),
            SparkRec(model=ALSWrap()),
            SparkUserRec(model=ClusterRec(), user_features=user_features, transient_user_features=True)
        ]
    )
])

model: PipelineModel = pipe.fit(log_train)

# TODO: extract final params and extract study from optimize

rec_model_path = "/tmp/rec_pipe"
model.write().save(rec_model_path)
rec_model = PipelineModel.read().load(rec_model_path)

recs = rec_model.transform(log_test, params={
    SparkUserRecModelParams.userFeatures: user_features,
    SparkBaseRecModelParams.numRecommendations: 100,
    SparkBaseRecModelParams.filterSeenItems: True
})
recs.write.parquet("recs.parquet", mode='overwrite')
