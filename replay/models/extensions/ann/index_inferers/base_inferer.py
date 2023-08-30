from abc import ABC, abstractmethod

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.extensions.ann.entities.base_hnsw_param import BaseHnswParam
from replay.models.extensions.ann.index_stores.base_index_store import IndexStore


# pylint: disable=too-few-public-methods
class IndexInferer(ABC):
    """Abstract base class that describes a common interface for index inferers
    and provides common methods for them."""

    # All implementations use the same udf return type.
    udf_return_type = "item_idx array<int>, distance array<double>"

    def __init__(self, index_params: BaseHnswParam, index_store: IndexStore):
        self.index_params = index_params
        self.index_store = index_store

    @abstractmethod
    def infer(
        self, vectors: DataFrame, features_col: str, k: int
    ) -> DataFrame:
        """Infers index"""

    @staticmethod
    def _unpack_infer_struct(inference_result: DataFrame) -> DataFrame:
        """Transforms input dataframe.
        Unpacks and explodes arrays from `neighbours` struct.

        >>>
        >> inference_result.printSchema()
        root
         |-- user_idx: integer (nullable = true)
         |-- neighbours: struct (nullable = true)
         |    |-- item_idx: array (nullable = true)
         |    |    |-- element: integer (containsNull = true)
         |    |-- distance: array (nullable = true)
         |    |    |-- element: double (containsNull = true)
        >> ANNMixin._unpack_infer_struct(inference_result).printSchema()
        root
         |-- user_idx: integer (nullable = true)
         |-- item_idx: integer (nullable = true)
         |-- relevance: double (nullable = true)

        Args:
            inference_result: output of infer_index UDF
        """
        res = inference_result.select(
            "user_idx",
            sf.explode(
                sf.arrays_zip("neighbours.item_idx", "neighbours.distance")
            ).alias("zip_exp"),
        )

        # Fix arrays_zip random behavior.
        # It can return zip_exp.0 or zip_exp.item_idx in different machines.
        fields = res.schema["zip_exp"].jsonValue()["type"]["fields"]
        item_idx_field_name: str = fields[0]["name"]
        distance_field_name: str = fields[1]["name"]

        res = res.select(
            "user_idx",
            sf.col(f"zip_exp.{item_idx_field_name}").alias("item_idx"),
            (sf.lit(-1.0) * sf.col(f"zip_exp.{distance_field_name}")).alias(
                "relevance"
            ),
        )
        return res
