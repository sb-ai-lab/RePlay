from abc import ABC, abstractmethod
from typing import Optional

import hnswlib
from pyspark.sql import DataFrame

from replay.ann.entities.hnswlib_param import HnswlibParam
from replay.utils import FileInfo


class BaseHnswIndexBuilder(ABC):
    @abstractmethod
    def build_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: HnswlibParam,
        id_col: Optional[str] = None,
    ) -> Optional[FileInfo]:
        """

        :return:
        """

    @staticmethod
    def init_index(params: HnswlibParam):
        index = hnswlib.Index(
            space=params.space, dim=params.dim
        )  # pylint: disable=c-extension-no-member

        # Initializing index - the maximum number of elements should be known beforehand
        index.init_index(
            max_elements=params.max_elements,
            ef_construction=params.efC,
            M=params.M,
        )

        return index
