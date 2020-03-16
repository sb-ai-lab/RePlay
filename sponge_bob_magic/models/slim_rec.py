"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from scipy.sparse import csc_matrix
from sklearn.linear_model import SGDRegressor

from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.session_handler import State
from sponge_bob_magic.utils import get_top_k_recs


class SlimRec(Recommender):
    """ SLIM Recommender основан на обучении матрицы близости объектов
    :math:`W`.

    Оптимизирует следующий функционал:

    .. math::
        L = \\frac 12||A - A W||^2_F + \\frac \\beta 2 ||W||_F^2+
        \\lambda
        ||W||_1

    :math:`W` -- матрица близости между объектами

    :math:`A` -- матрица взаимодействия пользователей/объектов

    Задачу нахождения матрицы :math:`W` можно разбить на множество
    задач линейной регрессии с ElasticNet регуляризацией. Таким образом,
    для каждой строки матрицы :math:`W` необходимо оптимизировать следующий
    функционал

    .. math::
        l = \\frac 12||a_j - A w_j||^2_2 + \\frac \\beta 2 ||w_j||_2^2+
        \\lambda ||w_j||_1

    Чтобы решение было не тривиальным, его ищут с ограничением :math:`w_{jj}=0`,
    кроме этого :math:`w_{ij}\\ge 0`
    """

    user_indexer: StringIndexer
    item_indexer: StringIndexer
    similarity: DataFrame

    def __init__(
            self,
            beta: float = 0.0,
            lambda_: float = 1.0,
            tolerance: float = 1e-6):
        if beta < 0 or lambda_ < 0 or (beta == 0 and lambda_ == 0) or \
                tolerance <= 0:
            raise ValueError("Неверно указаны параметры для регуляризации")
        self.beta = beta
        self.lambda_ = lambda_
        self.tolerance = tolerance
        self.spark = State().session

    def get_params(self) -> Dict[str, object]:
        return {"lambda": self.lambda_,
                "beta": self.beta}

    def _pre_fit(self,
                 log: DataFrame,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None) -> None:
        self.user_indexer = StringIndexer(
            inputCol="user_id", outputCol="user_idx").fit(log)
        self.item_indexer = StringIndexer(
            inputCol="item_id", outputCol="item_idx").fit(log)

    def _fit(self,
             log: DataFrame,
             user_features: Optional[DataFrame] = None,
             item_features: Optional[DataFrame] = None) -> None:
        logging.debug("Построение модели SLIM")

        alpha = self.beta + self.lambda_
        l1_ratio = self.lambda_ / alpha

        log_indexed = self.user_indexer.transform(log)
        log_indexed = self.item_indexer.transform(log_indexed)
        pandas_log = log_indexed.select(
            "user_idx", "item_idx", "relevance").collect()
        pandas_log = pd.DataFrame.from_records(pandas_log,
                                               columns=["user_idx",
                                                        "item_idx",
                                                        "relevance"])
        interactions_matrix = csc_matrix(
            (
                pandas_log.relevance,
                (pandas_log.user_idx, pandas_log.item_idx)
            ),
            shape=(
                len(self.user_indexer.labels),
                len(self.item_indexer.labels)
            )
        )
        similarity = (self.spark
                      .createDataFrame(range(len(self.item_indexer.labels)),
                                       st.StringType())
                      .withColumnRenamed("value", "item_id_one"))

        sgd = SGDRegressor(penalty='elasticnet',
                           alpha=alpha,
                           l1_ratio=l1_ratio,
                           fit_intercept=False)

        @sf.pandas_udf("item_id_one long, item_id_two long, relevance double",
                       sf.PandasUDFType.GROUPED_MAP)
        def slim_row(pandas_df):
            """
            Построчное обучение матрицы близости объектов, стохастическим
            градиентным спуском
            :param pandas_df: pd.Dataframe
            :return: pd.Dataframe
            """
            idx = int(pandas_df["item_id_one"][0])
            column = interactions_matrix[idx]
            column_arr = column.toarray().reshape(-1)
            interactions_matrix.data[
                interactions_matrix.indptr[idx]:
                interactions_matrix.indptr[idx + 1]] = 0

            sgd.fit(interactions_matrix.T, column_arr)
            interactions_matrix[idx] = column
            good_idx = np.argwhere(sgd.coef_ > self.tolerance).reshape(-1)
            good_values = sgd.coef_[good_idx]
            similarity_row = {'item_id_one': idx,
                              'item_id_two': good_idx,
                              'relevance': good_values}
            return pd.DataFrame(data=similarity_row)

        self.similarity = (similarity.groupby("item_id_one")
                           .apply(slim_row)).cache()

    def _predict(self,
                 log: DataFrame,
                 k: int,
                 users: DataFrame = None,
                 items: DataFrame = None,
                 context: str = None,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:
        recs = (
            log
            .join(
                users,
                how="inner",
                on="user_id"
            )
            .join(
                self.similarity,
                how="left",
                on=sf.col("item_id") == sf.col("item_id_one")
            )
            .groupby("user_id", "item_id_two")
            .agg(sf.sum("similarity").alias("relevance"))
            .withColumnRenamed("item_id_two", "item_id")
            .cache()
        )

        if filter_seen_items:
            recs = self._filter_seen_recs(recs, log)

        recs = get_top_k_recs(recs, k)
        recs = recs.filter(sf.col("relevance") > 0.0)

        return recs
