from rs_datasets import MovieLens

from replay.metrics import MAP, MRR, NDCG, Coverage, HitRate, Surprisal
from replay.metrics.experiment import Experiment
from replay.experimental.models.dt4rec.dt4rec import DT4Rec
from replay.experimental.preprocessing.data_preparator import DataPreparator, Indexer
from replay.splitters import TimeSplitter
from replay.utils import PYSPARK_AVAILABLE

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf


K = 10
K_list_metrics = [1, 5, 15]


df = MovieLens("1m").ratings

preparator = DataPreparator()
log = preparator.transform(
    columns_mapping={
        "user_id": "user_id",
        "item_id": "item_id",
        "relevance": "rating",
        "timestamp": "timestamp",
    },
    data=df,
)
indexer = Indexer(user_col="user_id", item_col="item_id")
indexer.fit(users=log.select("user_id"), items=log.select("item_id"))

# will consider ratings >= 3 as positive feedback.
# A positive feedback is treated with relevance = 1
only_positives_log = log.filter(sf.col("relevance") >= 3).withColumn("relevance", sf.lit(1.0))

indexed_log = indexer.transform(only_positives_log)

date_splitter = TimeSplitter(
    time_threshold=0.2, drop_cold_items=True, drop_cold_users=True, query_column="user_idx", item_column="item_idx"
)
train, test = date_splitter.split(indexed_log)


item_num = train.toPandas()["item_idx"].max() + 1
user_num = train.toPandas()["user_idx"].max() + 1

experiment = Experiment(
    {MAP(K), NDCG(K), HitRate(K_list_metrics), Coverage(K), Surprisal(K), MRR(K)},
    test,
    train,
    query_column="user_idx",
    item_column="item_idx",
    rating_column="relevance",
)

rec_sys = DT4Rec(item_num, user_num, use_cuda=True)
rec_sys.fit(train)
pred = rec_sys.predict(log=train, k=K, users=test.select("user_idx").distinct())

name = "DT4Rec"
experiment.add_result(name, pred)
experiment.results.sort_values(f"NDCG@{K}", ascending=False).to_csv("results.csv")
