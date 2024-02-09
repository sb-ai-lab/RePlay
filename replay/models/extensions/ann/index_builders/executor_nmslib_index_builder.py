import logging
from typing import Iterator, Optional

import pandas as pd

from .base_index_builder import IndexBuilder
from .nmslib_index_builder_mixin import NmslibIndexBuilderMixin
from replay.models.extensions.ann.index_inferers.base_inferer import IndexInferer
from replay.models.extensions.ann.index_inferers.nmslib_filter_index_inferer import NmslibFilterIndexInferer
from replay.models.extensions.ann.index_inferers.nmslib_index_inferer import NmslibIndexInferer
from replay.utils import PandasDataFrame, SparkDataFrame

logger = logging.getLogger("replay")


class ExecutorNmslibIndexBuilder(IndexBuilder):
    """
    Builder that build nmslib hnsw index on one executor.
    """

    def produce_inferer(self, filter_seen_items: bool) -> IndexInferer:
        if filter_seen_items:
            return NmslibFilterIndexInferer(
                self.index_params, self.index_store
            )
        else:
            return NmslibIndexInferer(self.index_params, self.index_store)

    def make_build_index_udf(self):
        """
        Method returns udf to build nmslib index.
        :return: `build_index_udf` pandas UDF
        """

        index_params = self.index_params
        index_store = self.index_store

        def build_index_udf(iterator: Iterator[PandasDataFrame]):  # pragma: no cover
            """Builds index on executor and writes it to shared disk or hdfs.

            Args:
                iterator: iterates on dataframes with vectors/features

            """
            # We collect all iterator values into one dataframe,
            # because we cannot guarantee that `pdf` will contain rows
            # with the same `item_idx_two`.
            # And therefore we cannot call the `addDataPointBatch` iteratively
            # (in build_and_save_index).
            pdfs = []
            for pdf in iterator:
                pdfs.append(pdf)

            pdf = pd.concat(pdfs)

            NmslibIndexBuilderMixin.build_and_save_index(
                pdf, index_params, index_store
            )

            yield PandasDataFrame(data={"_success": 1}, index=[0])

        return build_index_udf

    def build_index(
        self,
        vectors: SparkDataFrame,
        features_col: str,
        ids_col: Optional[str] = None,
    ):
        # to execution in one executor
        vectors = vectors.repartition(1)

        # this function will build the index
        build_index_udf = self.make_build_index_udf()

        # Here we perform materialization (`.collect()`) to build the hnsw index.
        vectors.select(
            "similarity", "item_idx_one", "item_idx_two"
        ).mapInPandas(build_index_udf, "_success int").collect()
