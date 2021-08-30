"""
Stack models
"""
# pylint: disable=invalid-name

import logging
from functools import reduce
from operator import add
from typing import List, Optional

import nevergrad as ng
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from tqdm import tqdm

from replay.constants import BASE_SCHEMA, IDX_SCHEMA
from replay.metrics import NDCG
from replay.models.base_rec import Recommender
from replay.session_handler import State
from replay.splitters import k_folds


class Stack(Recommender):
    """Use base models predictions to get final recommendations"""

    def __init__(
        self,
        models: List[Recommender],
        n_folds: Optional[int] = 5,
        budget: Optional[int] = 30,
        seed: Optional[int] = None,
    ):
        """
        :param models: list of initialized models
        :param n_folds: number of folds used to train stack
        :param budget: number of tries to find best way to stack models
        :param seed: random seed
        """
        self.models = models
        State()
        self.n_folds = n_folds
        self.budget = budget
        self._logger = logging.getLogger("replay")
        self.seed = seed

    # pylint: disable=too-many-locals, invalid-name
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        df = log.withColumnRenamed("user_idx", "user_id").withColumnRenamed(
            "item_idx", "item_id"
        )
        top_train, top_test = self._create_train(df)
        # pylint: disable=attribute-defined-outside-init
        self.top_train = top_train
        self._optimize_weights(top_train, top_test)
        for model in self.models:
            model.fit(df)

    def _optimize_weights(self, top_train, top_test):
        feature_cols = [str(model) for model in self.models]
        optimizer = self._get_optimizer(feature_cols)

        for _ in tqdm(range(optimizer.budget)):
            weights = optimizer.ask()
            ranking = [
                NDCG()(rerank(pred, **weights.kwargs), true, 50)
                for pred, true in zip(top_train, top_test)
            ]
            loss = -np.mean(ranking)
            optimizer.tell(weights, loss)

        # pylint: disable=attribute-defined-outside-init
        self.params = optimizer.provide_recommendation().kwargs
        # pylint: disable=invalid-name
        s = np.array(list(self.params.values()))
        if (s == 1).sum() == 1 and s.sum() == 1:
            name = [name for name in feature_cols if self.params[name] == 1][0]
            self._logger.warning(
                "Could not find combination to improve quality, "
                "%s works best on its own",
                name,
            )

    def _get_optimizer(self, feature_cols):
        coefs = {
            model: ng.p.Scalar(lower=0, upper=1) for model in feature_cols
        }
        parametrization = ng.p.Instrumentation(**coefs)
        optimizer = ng.optimizers.OnePlusOne(
            parametrization=parametrization, budget=self.budget
        )
        base = [
            dict(zip(feature_cols, vals)) for vals in np.eye(len(feature_cols))
        ]
        for one_model in base:
            optimizer.suggest(**one_model)
        return optimizer

    def _create_train(self, df):  # pylint: disable=invalid-name
        top_train = []
        top_test = []
        # pylint: disable=invalid-name
        for i, (train, test) in enumerate(
            k_folds(df, self.n_folds, self.seed)
        ):
            self._logger.info("Processing fold #%d", i)
            test_items = test.select("item_id").distinct()
            train_items = train.select("item_id").distinct()
            items_pos = test_items.join(train_items, on="item_id", how="inner")
            if items_pos.count() == 0:
                self._logger.info(
                    "Bad split, no positive examples, skipping..."
                )
                continue
            n_pos = (
                test.groupBy("user_id")
                .count()
                .agg({"count": "max"})
                .collect()[0][0]
            )
            fold_train = self._fold_predictions(train, n_pos * 2)
            top_train.append(fold_train)
            top_test.append(test)
        return top_train, top_test

    def _fold_predictions(self, train, n_pos):
        fold_train = State().session.createDataFrame(
            data=[], schema=BASE_SCHEMA
        )
        for model in self.models:
            scores = model.fit_predict(train, k=n_pos)
            scores = scores.withColumnRenamed("relevance", str(model))
            fold_train = fold_train.join(
                scores, on=["user_id", "item_id"], how="outer"
            ).fillna(0)
        return fold_train

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        top = State().session.createDataFrame(data=[], schema=IDX_SCHEMA)
        for model in self.models:
            # pylint: disable=protected-access
            scores = model._predict(
                log,
                k,
                users,
                items,
                user_features,
                item_features,
                filter_seen_items,
            )
            scores = scores.withColumn(
                "relevance",
                sf.when(scores["relevance"] < 0, None).otherwise(
                    scores["relevance"]
                ),
            )
            scores = scores.withColumnRenamed("relevance", str(model))
            top = top.join(
                scores, on=["user_idx", "item_idx"], how="outer"
            ).fillna(0)
        feature_cols = [str(model) for model in self.models]
        pred = rerank(top, **self.params)
        pred = pred.drop(*feature_cols)
        return pred


def rerank(df: DataFrame, **kwargs) -> DataFrame:
    """add relevance columns as a linear combination of kwargs"""
    res = df.withColumn(
        "relevance",
        reduce(add, [sf.col(col) * weight for col, weight in kwargs.items()]),
    )
    res = res.orderBy("relevance", ascending=False)
    return res
