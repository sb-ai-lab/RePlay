"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import pickle
from enum import Enum
from os import path

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, explode, length, lit, split, trim
from pyspark.sql.types import IntegerType, StringType
from scipy.sparse import load_npz

from sponge_bob_magic.sber_datasets.sber_dataset import SberDataset
from sponge_bob_magic.session_handler import State


class WhatToRecommend(Enum):
    """ СберСоветник может рекомендовать покупателей поставщикам, либо наоборот """

    BUYERS = "kt"
    SELLERS = "dt"


# pylint: disable=too-few-public-methods
class SberAdviser(SberDataset):
    """ Парсер выгрузки данных для СберСоветник """

    def __init__(
        self,
        folder: str,
        okato: str = "22",
        what_to_recommend: WhatToRecommend = WhatToRecommend.BUYERS,
        only_positive: bool = True,
    ):
        """
        >>> sber_adviser = SberAdviser("tests/data/sber_adviser")
        >>> sber_adviser.log
        DataFrame[user_id: string, item_id: string, relevance: int]
        >>> sber_adviser.user_features
        DataFrame[user_id: string, wntm_0: double, okved_0: double]

        :param folder: полный путь до папки с данными
        :param okato: код ОКАТО региона. По умолчанию 22 (Нижегородская область)
        :param what_to_recommend: рекомендовать поставщиков или покупателей
        :param only_positive: загружать только положительные (по умолчанию)
                              или ещё и предварительно отобранные отрицательные примеры
        """
        super().__init__(folder)
        self.only_positive = only_positive
        self.filtered_ids = (
            State()
            .session.read.csv(
                path.join(self.folder, "okveds_okato_hash.csv"), header=True
            )
            .where(col("okato") == okato)
            .select("ID")
        ).cache()
        self.filtered_ids_array = (
            self.filtered_ids.toPandas()["ID"].astype(int).to_numpy()
        )
        self.what_to_recommend = what_to_recommend
        self._item_features = self._prepare_item_features()
        self._log = self._load_log()

    def _load_log(self) -> DataFrame:
        positive_log = self._load_log_part(1)
        if self.only_positive:
            log = positive_log
        else:
            log = positive_log.union(self._load_log_part(0))
        return log

    def _load_log_part(self, feedback_part: int) -> DataFrame:
        filename = (
            f"df_recom_{self.what_to_recommend.value}{feedback_part}.csv"
        )
        ids_with_lists = (
            State()
            .session.read.csv(
                path.join(self.folder, self.what_to_recommend.value, filename)
            )
            .toDF("user_id", "item_list")
        )
        user_item_matrix = ids_with_lists.withColumn(
            "raw_id",
            explode(
                split(
                    col("item_list").substr(lit(2), length("item_list") - 2),
                    ",",
                )
            ),
        ).select(col("user_id"), trim(col("raw_id")).alias("item_id"))
        log_with_filtered_ids = (
            user_item_matrix.join(
                self.filtered_ids.select(col("ID").alias("item_id")),
                how="inner",
                on="item_id",
            )
            .join(
                self.filtered_ids.select(col("ID").alias("user_id")),
                how="inner",
                on="user_id",
            )
            .withColumn("relevance", lit(feedback_part))
        ).cache()
        return log_with_filtered_ids

    @property
    def log(self):
        return self._log

    def _get_feature_ids(self, features_type: str) -> np.ndarray:
        with open(
            path.join(
                self.folder, "feats", f"{features_type}_features_ID.pkl"
            ),
            "rb",
        ) as pickle_file:
            some_array = pickle.load(pickle_file)
            if len(some_array.shape) == 2:
                feature_ids = some_array[:, 0]
            else:
                feature_ids = some_array
        return feature_ids

    def _load_features(self, features_type: str) -> DataFrame:
        feature_ids = self._get_feature_ids(features_type)
        features = load_npz(
            path.join(self.folder, "feats", f"{features_type}_features.npz")
        ).astype(np.float32)
        feature_id_column = self.filtered_ids_array[
            np.in1d(self.filtered_ids_array, feature_ids)
        ].reshape(-1, 1)
        feature_value_columns = features[
            np.in1d(feature_ids, self.filtered_ids_array)
        ].todense()
        item_ids_with_feature_columns = pd.DataFrame(
            np.hstack([feature_id_column, feature_value_columns])
        )
        item_feature_matrix = (
            State()
            .session.createDataFrame(item_ids_with_feature_columns)
            .toDF(
                "iid",
                *[f"{features_type}_{i}" for i in range(features.shape[1])],
            )
        )
        return item_feature_matrix.withColumn(
            "item_id", col("iid").astype(IntegerType()).astype(StringType()),
        ).drop("iid")

    def _prepare_item_features(self) -> DataFrame:
        item_features_okved = self._load_features("okved")
        item_features_wntm = self._load_features("wntm")
        return item_features_wntm.join(
            item_features_okved, on="item_id", how="inner"
        ).cache()

    @property
    def item_features(self):
        return self._item_features

    @property
    def user_features(self):
        if self._user_features is None:
            self._user_features = self.item_features.withColumnRenamed(
                "item_id", "user_id"
            )
        return self._user_features
