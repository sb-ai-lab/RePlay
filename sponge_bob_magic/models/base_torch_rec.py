import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Union

from ignite import engine
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql import types as st
import torch
from torch import nn

from sponge_bob_magic.converter import convert, get_type
from sponge_bob_magic.models import Recommender
from sponge_bob_magic.session_handler import State


class TorchRecommender(Recommender, ABC):
    """ Базовый класс-рекомендатель для нейросетевой модели. """

    def _predict(self,
                 log: DataFrame,
                 k: int,
                 users: Optional[DataFrame] = None,
                 items: Optional[DataFrame] = None,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:
        items_pd = (self.item_indexer.transform(items)
                    .toPandas()["item_idx"].values)
        items_count = self.items_count
        model = self.model.cpu()
        agg_fn = self._predict_by_user

        @sf.pandas_udf(
            st.StructType([
                st.StructField("user_idx", st.LongType(), True),
                st.StructField("item_idx", st.LongType(), True),
                st.StructField("relevance", st.FloatType(), True)
            ]),
            sf.PandasUDFType.GROUPED_MAP
        )
        def grouped_map(pandas_df):
            return agg_fn(
                pandas_df, model, items_pd, k, items_count
            )[["user_idx", "item_idx", "relevance"]]

        self.logger.debug("Предсказание модели")
        recs = (
            self.item_indexer.transform(
                self.user_indexer.transform(
                    users.join(log, how="left", on="user_id")
                )
            )
        )
        recs = (
            recs.selectExpr(
                "CAST(user_idx AS INT) AS user_idx",
                "CAST(item_idx AS INT) AS item_idx")
            .groupby("user_idx")
            .apply(grouped_map)
        )
        recs = self.inv_item_indexer.transform(
            self.inv_user_indexer.transform(recs)
        ).drop("item_idx", "user_idx")

        recs = self.min_max_scale_column(recs, "relevance")
        return recs

    @staticmethod
    @abstractmethod
    def _predict_by_user(
            pandas_df: pd.DataFrame,
            model: nn.Module,
            items_np: np.array,
            k: int,
            item_count: int
    ) -> pd.DataFrame:
        """
        Расчёт значения метрики для каждого пользователя

        :param pandas_df: DataFrame, содержащий индексы просмотренных айтемов
            по каждому пользователю -- pandas-датафрейм вида
            ``[user_idx, item_idx]``
        :param model: обученная модель
        :param items_np: список допустимых для рекомендаций объектов
        :param k: количество рекомендаций
        :param item_count: общее количество объектов в рекомендателе
        :return: DataFrame c рассчитанными релевантностями --
            pandas-датафрейм вида ``[user_idx , item_idx , relevance]``
        """

    @staticmethod
    def min_max_scale_column(dataframe: DataFrame, column: str) -> DataFrame:
        """
        Отнормировать колонку датафрейма.
        Применяет классическую форму нормализации с минимумом и максимумом:
        new_value_i = (value_i - min) / (max - min).

        :param dataframe: спарк-датафрейм
        :param column: имя колонки, которую надо нормализовать
        :return: исходный датафрейм с измененной колонкой
        """
        unlist = sf.udf(lambda x: float(list(x)[0]), st.DoubleType())
        assembler = VectorAssembler(
            inputCols=[column], outputCol=f"{column}_Vect"
        )
        scaler = MinMaxScaler(
            inputCol=f"{column}_Vect", outputCol=f"{column}_Scaled"
        )
        pipeline = Pipeline(stages=[assembler, scaler])
        dataframe = (pipeline
                     .fit(dataframe)
                     .transform(dataframe)
                     .withColumn(column, unlist(f"{column}_Scaled"))
                     .drop(f"{column}_Vect", f"{column}_Scaled"))

        return dataframe

    def load_model(self, path: str) -> None:
        """
        Загрузка весов модели из файла

        :param path: путь к файлу, откуда загружать
        :return:
        """
        self.logger.debug("-- Загрузка модели из файла")
        self.model.load_state_dict(torch.load(path))

    def _create_trainer(self):
        pass

    def _create_evaluator(self):
        pass
