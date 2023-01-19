from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession
from rs_datasets import MovieLens

from replay.data_preparator import DataPreparator
from replay.models import Word2VecRec, ALSWrap, ClusterRec
from replay.spark_ml_rec.data_preparator import SparkIndexer
from replay.spark_ml_rec.spark_base_rec import SparkBaseRecModelParams
from replay.spark_ml_rec.spark_rec import SparkRec
from replay.spark_ml_rec.spark_user_rec import SparkUserRec, SparkUserRecModelParams
from replay.spark_ml_rec.splitter import SparkTrainTestSplitter
from replay.splitters import UserSplitter, DateSplitter
from replay.utils import convert2spark

spark = SparkSession.builder.getOrCreate()

ds = MovieLens('100k')

log = convert2spark(ds.ratings)
user_features = convert2spark(ds.users)

log_train, log_test = DateSplitter(test_start=0.2, date_col="timestamp")

pipe = Pipeline([
    # TODO: set column mapping
    DataPreparator(),
    SparkIndexer(),
    # TODO: optimize, save_all_models
    SparkTrainTestSplitter(
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
