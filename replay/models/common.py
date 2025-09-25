import logging
from typing import Any, Optional

from replay.utils import SparkDataFrame
from replay.utils.spark_utils import cache_temp_view, drop_temp_view


class RecommenderCommons:
    """
    Common methods and attributes of RePlay models for caching, setting parameters and logging
    """

    _logger: Optional[logging.Logger] = None
    cached_dfs: Optional[set] = None
    query_column: str
    item_column: str
    rating_column: str
    timestamp_column: str

    def set_params(self, **params: dict[str, Any]) -> None:
        """
        Set model parameters

        :param params: dictionary param name - param value
        :return:
        """
        for param, value in params.items():
            setattr(self, param, value)
        self._clear_cache()

    def _clear_cache(self):
        """
        Clear spark cache
        """

    def __str__(self):
        return type(self).__name__

    @property
    def logger(self) -> logging.Logger:
        """
        :returns: get library logger
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger

    def _cache_model_temp_view(self, df: SparkDataFrame, df_name: str) -> None:
        """
        Create Spark SQL temporary view for df, cache it and add temp view name to self.cached_dfs.
        Temp view name is : "id_<python object id>_model_<RePlay model name>_<df_name>"
        """
        full_name = f"id_{id(self)}_model_{self!s}_{df_name}"
        cache_temp_view(df, full_name)

        if self.cached_dfs is None:
            self.cached_dfs = set()
        self.cached_dfs.add(full_name)

    def _clear_model_temp_view(self, df_name: str) -> None:
        """
        Uncache and drop Spark SQL temporary view and remove from self.cached_dfs
        Temp view to replace will be constructed as
        "id_<python object id>_model_<RePlay model name>_<df_name>"
        """
        full_name = f"id_{id(self)}_model_{self!s}_{df_name}"
        drop_temp_view(full_name)
        if self.cached_dfs is not None:
            self.cached_dfs.discard(full_name)
