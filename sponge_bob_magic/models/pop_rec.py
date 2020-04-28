"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

from sponge_bob_magic.models.base_rec import Recommender


class PopRec(Recommender):
    """
    Базовый рекомендатель на основе популярности.

    Популярность объекта определяется как вероятность того,
    что случайно выбранный пользователь взаимодействовал с объектом:

    .. math::
        Popularity(i) = \\dfrac{N_i}{N}

    :math:`N_i` - количество пользователей, у которых было взаимодействие с
    объектом :math:`i`

    :math:`N` - общее количество пользователей,
    независимо от взаимодействия с объектом.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"user_id": [1, 1, 2, 2, 3], "item_id": [1, 2, 3, 3, 3]})
    >>> df
       user_id  item_id
    0        1        1
    1        1        2
    2        2        3
    3        2        3
    4        3        3

    >>> from sponge_bob_magic.converter import convert
    >>> res = PopRec().fit_predict(convert(df), 1)
    >>> res.toPandas().sort_values("user_id", ignore_index=True)
       user_id  item_id  relevance
    0        1        3   0.666667
    1        2        2   0.333333
    2        3        2   0.333333

    >>> res = PopRec().fit_predict(convert(df), 1, filter_seen_items=False)
    >>> res.toPandas().sort_values("user_id", ignore_index=True)
       user_id  item_id  relevance
    0        1        3        0.666667
    1        2        3        0.666667
    2        3        3        0.666667
    """

    item_popularity: DataFrame
    can_predict_cold_users = True

    def get_params(self) -> Dict[str, object]:
        return {}

    def _pre_fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        super()._pre_fit(log, user_features, item_features)
        self.item_popularity = log.groupBy("item_id").agg(
            sf.countDistinct("user_id").alias("user_count")
        )

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:

        self.item_popularity = (
            self.item_popularity.select(
                "item_id",
                (sf.col("user_count") / sf.lit(self.users_count)).alias("relevance"),
            )
        ).cache()

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
        # удаляем ненужные items
        items_pd = (
            self.item_indexer.transform(
                items.join(
                    self.item_popularity.withColumnRenamed("item_id", "item_id_2"),
                    on=sf.col("item_id") == sf.col("item_id_2"),
                    how="inner",
                )
            )
            .drop("item_id_2", "item_id")
            .toPandas()
        )

        @sf.pandas_udf(
            st.StructType(
                [
                    st.StructField("user_id", users.schema["user_id"].dataType, True),
                    st.StructField("user_idx", st.LongType(), True),
                    st.StructField("item_idx", st.LongType(), True),
                    st.StructField("relevance", st.DoubleType(), True),
                ]
            ),
            sf.PandasUDFType.GROUPED_MAP,
        )
        def grouped_map(pandas_df):
            user_idx = pandas_df["user_idx"][0]
            user_id = pandas_df["user_id"][0]
            cnt = pandas_df["cnt"][0]

            items_idx = np.argsort(items_pd["relevance"].values)[-cnt:]

            return pd.DataFrame(
                {
                    "user_id": cnt * [user_id],
                    "user_idx": cnt * [user_idx],
                    "item_idx": items_pd["item_idx"].values[items_idx],
                    "relevance": items_pd["relevance"].values[items_idx],
                }
            )

        model_len = len(items_pd)
        recs = self.user_indexer.transform(
            users.join(log, how="left", on="user_id")
            .select("user_id", "item_id")
            .groupby("user_id")
            .agg(sf.countDistinct("item_id").alias("cnt"))
        )
        recs = self.inv_item_indexer.transform(
            recs.selectExpr(
                "user_id",
                "CAST(user_idx AS INT) AS user_idx",
                f"CAST(LEAST(cnt + {k}, {model_len}) AS INT) AS cnt",
            )
            .groupby("user_id", "user_idx")
            .apply(grouped_map)
        )
        recs = recs.drop("item_idx", "user_idx").select(
            "user_id", "item_id", "relevance"
        )
        return recs
