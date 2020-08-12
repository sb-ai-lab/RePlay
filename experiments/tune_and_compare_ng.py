# type: ignore
# pylint: disable-all
from replay.metrics import *
from replay.models import *
from rs_datasets import MovieLens
from datetime import datetime

from replay.models import PopRec
from replay.splitters import UserSplitter
from replay.experiment import Experiment

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

import nevergrad as ng

seed = 1337
budget = 100
params = dict()
k = [5, 10, 50, 100]
start = datetime.now()
for dataset in ["20m"]:
    print(f"starting {dataset}")
    ml = MovieLens(dataset)
    df = ml.ratings
    df["relevance"] = df.rating
    params[dataset] = dict()

    splitter = UserSplitter(0.2, shuffle=True, drop_cold_items=True, seed=seed)
    train, test = splitter.split(df)
    train, val = splitter.split(train)
    test.toPandas().to_csv("test.csv")
    # def train_als(factors, regularization):
    #     model = AlternatingLeastSquares(factors, regularization)
    #     return train_implicit(model)

    def train_bpr(factors, regularization, learning_rate):
        model = BayesianPersonalizedRanking(
            factors, learning_rate, regularization
        )
        return train_implicit(model)

    def train_implicit(model):
        model = ImplicitWrap(model)
        pred = model.fit_predict(train, k=max(k))
        return -NDCG()(pred, val, max(k))

    e = Experiment(test, {NDCG(): k, HitRate(): k, MRR(): k, Recall(): k,},)

    lr = ng.p.Log(lower=0.0001, upper=1.0)
    reg = ng.p.Log(lower=0.001, upper=1.0)
    factors = ng.p.Scalar(lower=5, upper=300).set_integer_casting()

    parametrization = ng.p.Instrumentation(
        regularization=reg, factors=factors, learning_rate=lr
    )

    optimizer = ng.optimizers.OnePlusOne(
        parametrization=parametrization, budget=budget
    )
    recommendation = optimizer.minimize(train_bpr)
    model = BayesianPersonalizedRanking(**recommendation.kwargs)
    model = ImplicitWrap(model)
    pred = model.fit_predict(train, k=max(k))
    e.add_result("bpr", pred)
    params[dataset]["bpr"] = str(recommendation.kwargs)
    pred.toPandas().to_csv("bpr_pred.csv")
    print("bpr done")

    model = PopRec()
    pred = model.fit_predict(train, max(k))
    e.add_result("pop", pred)
    pred.toPandas().to_csv("pop_pred.csv")
    print("pop done")

    e.results.to_csv(f"implicit_{dataset}.csv")
    print(f"time elapsed: {str(datetime.now() - start)}")

with open("params.txt", "w") as f:
    f.write(str(params))

print(f"finished in: {str(datetime.now() - start)}")
