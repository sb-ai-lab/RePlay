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
budget = 20
params = dict()
k = [5, 10, 50, 100]
start = datetime.now()
for dataset in ["1m", "10m"]:
    print(f"starting {dataset}")
    ml = MovieLens(dataset)
    df = ml.ratings
    df["relevance"] = df.rating
    params[dataset] = dict()

    splitter = UserSplitter(0.2, shuffle=True, drop_cold_items=True, seed=seed)
    train, test = splitter.split(df)
    train, val = splitter.split(train)
    cov = Coverage(df)

    def train_slim(beta, lambda_):
        model = SLIM(beta, lambda_)
        pred = model.fit_predict(train, k=2000)
        n = NDCG()(pred, val, 100)
        c = cov(pred, 2000)
        return -(n * c) / (n + c)

    def train_als(factors, regularization):
        model = AlternatingLeastSquares(factors, regularization)
        return train_implicit(model)

    def train_bpr(factors, regularization, learning_rate):
        model = BayesianPersonalizedRanking(
            factors, learning_rate, regularization
        )
        return train_implicit(model)

    def train_implicit(model):
        model = ImplicitWrap(model)
        pred = model.fit_predict(train, k=max(k))
        return -NDCG()(pred, val, max(k))

    e = Experiment(test, {NDCG(): [100], cov: [100, 2000], Recall(): [100],},)

    lr = ng.p.Log(lower=0.0001, upper=1.0)
    reg = ng.p.Log(lower=0.001, upper=1.0)
    factors = ng.p.Scalar(lower=5, upper=300).set_integer_casting()

    parametrization = ng.p.Instrumentation(
        regularization=reg, factors=factors,
    )

    optimizer = ng.optimizers.OnePlusOne(
        parametrization=parametrization, budget=budget
    )
    recommendation = optimizer.minimize(train_als)
    model = AlternatingLeastSquares(**recommendation.kwargs)
    model = ImplicitWrap(model)
    pred = model.fit_predict(train, k=max(k))
    e.add_result("als", pred)
    params[dataset]["als"] = str(recommendation.kwargs)
    print("als done")

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
    print("bpr done")

    model = PopRec()
    pred = model.fit_predict(train, max(k))
    e.add_result("pop", pred)
    print("pop done")

    e.results.to_csv(f"implicit_{dataset}.csv")
    print(f"time elapsed: {str(datetime.now() - start)}")

with open("params.txt", "w") as f:
    f.write(str(params))

print(f"finished in: {str(datetime.now() - start)}")
