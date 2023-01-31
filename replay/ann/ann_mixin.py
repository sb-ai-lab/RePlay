import uuid
from abc import abstractmethod
from functools import cached_property
from typing import Optional, Dict, Union, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.base_rec import BaseRecommender


class ANNMixin(BaseRecommender):
    @cached_property
    def _spark_index_file_uid(self):
        return uuid.uuid4().hex[-12:]

    @property
    @abstractmethod
    def _use_ann(self) -> bool:
        ...

    @abstractmethod
    def _get_vectors_to_build_ann(self, log: DataFrame) -> DataFrame:
        ...

    @abstractmethod
    def _get_ann_build_params(self, log: DataFrame) -> Dict[str, Any]:
        ...

    def _fit_wrap(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        super()._fit_wrap(log, user_features, item_features)

        if self._use_ann:
            vectors = self._get_vectors_to_build_ann(log)
            ann_params = self._get_ann_build_params(log)
            self._build_ann_index(vectors, **ann_params)

    @abstractmethod
    def _get_vectors_to_infer_ann_inner(
        self, log: DataFrame, users: DataFrame
    ) -> DataFrame:
        ...

    def _get_vectors_to_infer_ann(
        self, log: DataFrame, users: DataFrame, filter_seen_items: bool
    ) -> DataFrame:

        users = self._get_vectors_to_infer_ann_inner(log, users)

        # here we add `seen_item_idxs` to filter the viewed items in UDFs (see infer_index)
        if filter_seen_items:
            user_to_max_items = log.groupBy("user_idx").agg(
                sf.count("item_idx").alias("num_items"),
                sf.collect_set("item_idx").alias("seen_item_idxs"),
            )
            users = users.join(user_to_max_items, on="user_idx")

        return users

    @abstractmethod
    def _get_ann_infer_params(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def _build_ann_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: Dict[str, Union[int, str]],
        dim: int = None,
        num_elements: int = None,
        id_col: Optional[str] = None,
        index_type: str = None,
        items_count: Optional[int] = None,
    ) -> None:
        ...

    @abstractmethod
    def _infer_ann_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: Dict[str, Union[int, str]],
        k: int,
        filter_seen_items: bool,
        index_dim: str = None,
        index_type: str = None,
        log: DataFrame = None,
    ) -> DataFrame:
        ...

    def _inner_predict_wrap(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:

        if self._use_ann:
            vectors = self._get_vectors_to_infer_ann(
                log, users, filter_seen_items
            )
            ann_params = self._get_ann_infer_params()
            return self._infer_ann_index(
                vectors,
                k=k,
                filter_seen_items=filter_seen_items,
                log=log,
                **ann_params,
            )
        else:
            return self._predict(
                log,
                k,
                users,
                items,
                user_features,
                item_features,
                filter_seen_items,
            )

    def _unpack_infer_struct(self, inference_result: DataFrame) -> DataFrame:
        """Transforms input dataframe.
        Unpacks and explodes arrays from `neighbours` struct.

        >>> inference_result.printSchema()
        root
         |-- user_idx: integer (nullable = true)
         |-- neighbours: struct (nullable = true)
         |    |-- item_idx: array (nullable = true)
         |    |    |-- element: integer (containsNull = true)
         |    |-- distance: array (nullable = true)
         |    |    |-- element: double (containsNull = true)
        >>> self._unpack_infer_struct(inference_result).printSchema()
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

        # Fix arrays_zip random behavior. It can return zip_exp.0 or zip_exp.item_idx in different machines
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

    def _filter_seen(
        self, recs: DataFrame, log: DataFrame, k: int, users: DataFrame
    ):
        """
        Overridden _filter_seen method from base class.
        Filtering is not needed for ann methods, because the data is already filtered in udf.
        """
        if self._use_ann:
            return recs

        return super()._filter_seen(recs, log, k, users)
