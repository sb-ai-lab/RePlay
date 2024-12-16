import logging
from typing import Iterator, Optional

import numpy as np

from replay.models.extensions.ann.index_inferers.base_inferer import IndexInferer
from replay.models.extensions.ann.index_inferers.hnswlib_filter_index_inferer import HnswlibFilterIndexInferer
from replay.models.extensions.ann.index_inferers.hnswlib_index_inferer import HnswlibIndexInferer
from replay.models.extensions.ann.utils import create_hnswlib_index_instance
from replay.utils import PandasDataFrame, SparkDataFrame

from .base_index_builder import IndexBuilder

logger = logging.getLogger("replay")


class ExecutorHnswlibIndexBuilder(IndexBuilder):
    """
    Builder that build hnswlib index on one executor.
    """

    def produce_inferer(self, filter_seen_items: bool) -> IndexInferer:
        if filter_seen_items:
            return HnswlibFilterIndexInferer(self.index_params, self.index_store)
        else:
            return HnswlibIndexInferer(self.index_params, self.index_store)

    def build_index(
        self,
        vectors: SparkDataFrame,
        features_col: str,
        ids_col: Optional[str] = None,
    ):
        # to execution in one executor
        vectors = vectors.repartition(1)

        _index_store = self.index_store
        _index_params = self.index_params

        def build_index_udf(iterator: Iterator[PandasDataFrame]):  # pragma: no cover
            """Builds index on executor and writes it to shared disk or hdfs.

            Args:
                iterator: iterates on dataframes with vectors/features

            """
            index = create_hnswlib_index_instance(_index_params, init=True)

            # pdf is a pandas dataframe that contains ids and features (vectors)
            for pdf in iterator:
                vectors_np = np.squeeze(pdf[features_col].values)
                if ids_col:
                    index.add_items(np.stack(vectors_np), pdf[ids_col].values)
                else:
                    # ids will be from [0, ..., len(vectors_np)]
                    index.add_items(np.stack(vectors_np))

            _index_store.save_to_store(lambda path: index.save_index(path))

            yield PandasDataFrame(data={"_success": 1}, index=[0])

        # Here we perform materialization (`.collect()`) to build the hnsw index.
        cols = [ids_col, features_col] if ids_col else [features_col]

        vectors.select(*cols).mapInPandas(build_index_udf, "_success int").collect()
