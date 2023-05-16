import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

from pyspark.sql import DataFrame

from replay.ann.entities.hnswlib_param import HnswlibParam
from replay.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.utils import FileInfo

logger = logging.getLogger("replay")


class BaseHnswIndexBuilder(ABC):
    @abstractmethod
    def _build_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: HnswlibParam,
        id_col: Optional[str] = None,
    ) -> Optional[FileInfo]:
        """
        Method that builds index and save it.

        :return: path where the index is stored or None
        """

    def build_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: Union[HnswlibParam, NmslibHnswParam],
        id_col: Optional[str] = None,
    ) -> Optional[FileInfo]:
        """
        Method that builds index and save it. Wraps `_build_index` method.

        :param vectors: DataFrame with vectors to build index. Schema: [{id_col}: int, {features_col}: array<float>]
        :param features_col: name of column in the `vectors` dataframe
        :param params: index params
        :param id_col: name of column in the `vectors` dataframe that contains ids (of vectors)
        :return: path where the index is stored or None
        """
        logger.info("Started building hnsw index")
        index_file = self._build_index(vectors, features_col, params, id_col)
        logger.info("Finished building hnsw index")
        return index_file
