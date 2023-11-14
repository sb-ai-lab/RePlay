from rs_datasets import MovieLens

from replay.metrics import MAP, MRR, NDCG, Coverage, HitRate, Surprisal
from replay.metrics.experiment import Experiment
from replay.models import DT4Rec
from replay.preprocessing.data_preparator import DataPreparator, Indexer
from replay.splitters import DateSplitter

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

train_spl = DateSplitter(
    test_start=0.2,
    drop_cold_items=True,
    drop_cold_users=True,
    item_col="item_id",
    user_col="user_id",
)
train, test = train_spl.split(log)
indexer = Indexer(user_col="user_id", item_col="item_id")
indexer.fit(users=train.select("user_id"), items=train.select("item_id"))
train = indexer.transform(train)
test = indexer.transform(test)

item_num = train.toPandas()["item_idx"].max() + 1
user_num = train.toPandas()["user_idx"].max() + 1

experiment = Experiment(
    test,
    {
        MAP(): K,
        NDCG(): K,
        HitRate(): K_list_metrics,
        Coverage(train): K,
        Surprisal(train): K,
        MRR(): K,
    },
)

rec_sys = DT4Rec(item_num, user_num, use_cuda=False)
rec_sys.fit(train)
pred = rec_sys.predict(log=train, k=K, users=test.select("user_idx").distinct())

name = "DT4Rec"
experiment.add_result(name, pred)
experiment.results.sort_values(f"NDCG@{K}", ascending=False).to_csv("results.csv")
