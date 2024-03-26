import logging
from typing import Optional

from replay.models.extensions.ann.index_inferers.base_inferer import IndexInferer
from replay.models.extensions.ann.index_inferers.nmslib_filter_index_inferer import NmslibFilterIndexInferer
from replay.models.extensions.ann.index_inferers.nmslib_index_inferer import NmslibIndexInferer
from replay.utils import SparkDataFrame
from replay.utils.spark_utils import spark_to_pandas

from .base_index_builder import IndexBuilder
from .nmslib_index_builder_mixin import NmslibIndexBuilderMixin

logger = logging.getLogger("replay")


class DriverNmslibIndexBuilder(IndexBuilder):
    """
    Builder that builds nmslib hnsw index on driver.
    """

    def produce_inferer(self, filter_seen_items: bool) -> IndexInferer:
        if filter_seen_items:
            return NmslibFilterIndexInferer(self.index_params, self.index_store)
        else:
            return NmslibIndexInferer(self.index_params, self.index_store)

    def build_index(
        self,
        vectors: SparkDataFrame,
        features_col: str,  # noqa: ARG002
        ids_col: Optional[str] = None,  # noqa: ARG002
    ):
        vectors = spark_to_pandas(vectors, self.allow_collect_to_master)
        NmslibIndexBuilderMixin.build_and_save_index(vectors, self.index_params, self.index_store)
