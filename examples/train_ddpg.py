import json
import logging
import time
import warnings
from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint

from optuna.exceptions import ExperimentalWarning
from pyspark.sql import functions as sf
from rs_datasets import MovieLens

from replay.experimental.models import DDPG
from replay.metrics import MAP, MRR, NDCG, Coverage, Experiment, HitRate, Surprisal
from replay.preprocessing import DataPreparator, Indexer
from replay.splitters import TimeSplitter
from replay.utils import State

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ExperimentalWarning)

TOP_K = 10
TOP_K_LIST_METRICS = [1, 5, 10]


def fit_predict_add_res(
    name,
    model: DDPG,
    experiment,
    train,
    test,
    top_k,
    folder_path,
    suffix="",
    load=False,
):
    """
    Run fit_predict for the `model`, measure time on fit_predict and evaluate metrics
    """
    start_time = time.time()

    logs = {"log": train}
    predict_params = {"k": top_k, "users": test.select("user_idx").distinct()}

    predict_params.update(logs)

    if not load:
        model.fit(**logs)
    else:
        model._load_model(name)

    fit_time = time.time() - start_time

    pred = model.predict(**predict_params)
    pred.cache()
    predict_time = time.time() - start_time - fit_time

    experiment.add_result(name + suffix, pred)
    metric_time = time.time() - start_time - fit_time - predict_time
    experiment.results.loc[name + suffix, "fit_time"] = fit_time
    experiment.results.loc[name + suffix, "predict_time"] = predict_time
    experiment.results.loc[name + suffix, "metric_time"] = metric_time
    experiment.results.loc[name + suffix, "full_time"] = (
        fit_time + predict_time + metric_time
    )
    pred.unpersist()
    print(
        experiment.results[
            [
                "NDCG@{}".format(top_k),
                "MRR@{}".format(top_k),
                "Coverage@{}".format(top_k),
                "fit_time",
            ]
        ].sort_values("NDCG@{}".format(top_k), ascending=False)
    )
    experiment.results.to_csv(folder_path / f"{name}.csv")

    if not load:
        model._save_model(folder_path / f"{name}.pt")


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=False)
    parser.add_argument(
        "--n_trials", dest="n_trials", required=False, default=10, type=int
    )
    parser.add_argument("--no-opt", dest="is_opt", action="store_false")
    parser.add_argument("--name", dest="name", required=False, default="DDPG")
    args = parser.parse_args()
    config_path = Path(args.config_path)
    folder_path = config_path.parent
    with open(config_path) as config_file:
        config = json.load(config_file)

    print("DDPG PARAMS")
    pprint(config["ddpg_params"])
    print("\n")
    print("SEARCH SPACE")
    pprint(config["search_space"])

    spark = State().session

    # set loggers levels
    # TODO add arg for that
    spark.sparkContext.setLogLevel("OFF")
    logger = logging.getLogger("replay")
    logger.setLevel(logging.CRITICAL)

    # prepare data
    data = MovieLens("1m")

    preparator = DataPreparator()
    log = preparator.transform(
        columns_mapping={
            "user_id": "user_id",
            "item_id": "item_id",
            "relevance": "rating",
            "timestamp": "timestamp",
        },
        data=data.ratings,
    )

    only_positives_log = log.filter(sf.col("relevance") >= 3).withColumn(
        "relevance", sf.lit(1)
    )

    indexer = Indexer(user_col="user_id", item_col="item_id")
    indexer.fit(users=log.select("user_id"), items=log.select("item_id"))
    log_replay = indexer.transform(df=only_positives_log)

    train_spl = TimeSplitter(
        test_start=0.2,
        drop_cold_items=True,
        drop_cold_users=True,
    )
    train, test = train_spl.split(log_replay)

    opt_train, opt_val = train_spl.split(train)

    # fitting
    model = DDPG(**config["ddpg_params"])
    model._search_space = config["search_space"]

    if args.is_opt:
        best_params = model.optimize(
            opt_train, opt_val, k=TOP_K, budget=args.n_trials
        )

        with open(folder_path / "best_params.json", "w") as f:
            json.dump(best_params, f)

        model.set_params(**best_params)

    e = Experiment(
        test,
        {
            MAP(): TOP_K,
            NDCG(): TOP_K,
            HitRate(): TOP_K_LIST_METRICS,
            Coverage(train): TOP_K,
            Surprisal(train): TOP_K,
            MRR(): TOP_K,
        },
    )

    fit_predict_add_res(args.name, model, e, train, test, TOP_K, folder_path)


if __name__ == "__main__":
    main()
