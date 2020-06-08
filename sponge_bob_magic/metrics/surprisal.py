"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from sponge_bob_magic.constants import AnyDataFrame
from sponge_bob_magic.converter import convert
from sponge_bob_magic.metrics.base_metric import RecOnlyMetric


# pylint: disable=too-few-public-methods
class Surprisal(RecOnlyMetric):
    """
    Показывает насколько редкие предметы выдаются в рекомендациях.
    В качестве оценки редкости используется собственная информация объекта,
    превращающая популярность объекта в его неожиданность.

    .. math::
        \\textit{Self-Information}(j)= -\log_2 \\frac {u_j}{N}

    :math:`u_j` -- количество пользователей, которые взаимодействовали с объектом :math:`j`.
    Для холодных объектов количество взаимодействий считается равным 1,
    то есть их появление в рекомендациях будет считаться крайне неожиданным.

    Чтобы метрику было проще интерпретировать, это значение нормируется.

    Таким образом редкость объекта :math:`j` определяется как

    .. math::
        Surprisal(j)= \\frac {\\textit{Self-Information}(j)}{log_2 N}

    Для списка рекомендаций длины :math:`K` значение метрики определяется как среднее значение редкости.

    .. math::
        Surprisal@K(i) = \\frac {\sum_{j=1}^{K}Surprisal(j)} {K}

    Итоговое значение усредняется по пользователям

    .. math::
        Surprisal@K = \\frac {\sum_{i=1}^{N}Surprisal@K(i)}{N}
    """

    def __init__(
        self, log: AnyDataFrame
    ):  # pylint: disable=super-init-not-called
        """
        Чтобы посчитать метрику, необходимо предрассчитать собственную информацию каждого объекта.

        :param log: датафрейм с логом действий пользователей
        """
        self.log = convert(log)
        n_users = self.log.select("user_id").distinct().count()
        self.item_weights = self.log.groupby("item_id").agg(
            (
                sf.log2(n_users / sf.countDistinct("user_id"))
                / np.log2(n_users)
            ).alias("rec_weight")
        )

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        return pandas_df.assign(
            cum_agg=pandas_df["rec_weight"].cumsum() / pandas_df["k"]
        )

    def _get_enriched_recommendations(
        self, recommendations: DataFrame, ground_truth: DataFrame
    ) -> DataFrame:
        return recommendations.join(
            self.item_weights, on="item_id", how="left"
        ).fillna(1)
