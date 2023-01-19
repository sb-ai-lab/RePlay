from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from rs_datasets import MovieLens

from replay.data_preparator import DataPreparator
from replay.models import Word2VecRec, ALSWrap, ClusterRec
from replay.spark_ml_rec.data_preparator import SparkIndexer
from replay.spark_ml_rec.spark_rec import SparkRec
from replay.spark_ml_rec.spark_user_rec import SparkUserRec
from replay.spark_ml_rec.splitter import SparkTrainTestSplitter
from replay.splitters import UserSplitter
from replay.utils import convert2spark

spark = SparkSession.builder.getOrCreate()

ds = MovieLens('100k')

log = convert2spark(ds.ratings)
user_features = convert2spark(ds.users)

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
            SparkUserRec(model=ClusterRec(), user_features=user_features)
        ]
    )
])

model = pipe.fit(log)

# TODO: extract final params and extract study from optimize

# TODO: save model
# model.save()

# TODO: load model

# TODO: send other params via pipeline ?
recs = model.transform(log, params={"userFeatures": user_features, "k": 100})
recs.write.parquet("recs.parquet", mode='overwrite')
