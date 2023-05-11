import uuid
from abc import abstractmethod
from cached_property import cached_property
from typing import Optional, Dict, Union, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.base_rec import BaseRecommender


class ANNMixin(BaseRecommender):
    """
    This class overrides the `_fit_wrap` and `_inner_predict_wrap` methods of the base class,
    adding an index construction in the `_fit_wrap` step
    and an index inference in the `_inner_predict_wrap` step.
    """

    @cached_property
    def _spark_index_file_uid(self) -> str:  # pylint: disable=no-self-use
        """
        Cached property that returns the uuid for the index file name.
        The unique name is needed to store the index file in `SparkFiles`
        without conflicts with other index files.
        """
        return uuid.uuid4().hex[-12:]

    @property
    @abstractmethod
    def _use_ann(self) -> bool:
        """
        Property that determines whether the ANN (index) is used.
        If `True`, then the index will be built (at the `fit` stage)
        and index will be inferred (at the `predict` stage).
        """

    @abstractmethod
    def _get_vectors_to_build_ann(self, log: DataFrame) -> DataFrame:
        """Implementations of this method must return a dataframe with item vectors.
        Item vectors from this method are used to build the index.

        Args:
            log: DataFrame with interactions

        Returns: DataFrame[item_idx int, vector array<double>] or DataFrame[vector array<double>].
        Column names in dataframe can be anything.
        """

    @abstractmethod
    def _get_ann_build_params(self, log: DataFrame) -> Dict[str, Any]:
        """Implementation of this method must return dictionary
        with arguments for `_build_ann_index` method.

        Args:
            log: DataFrame with interactions

        Returns: Dictionary with arguments to build index. For example: {
            "id_col": "item_idx",
            "features_col": "item_factors",
            "params": self._hnswlib_params,
            "dim": 100,
            "num_elements": 10_000,
        }

        """

    def _fit_wrap(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """Wrapper extends `_fit_wrap`, adds construction of ANN index by flag.

        Args:
            log: historical log of interactions
                ``[user_idx, item_idx, timestamp, relevance]``
            user_features: user features
                ``[user_idx, timestamp]`` + feature columns
            item_features: item features
                ``[item_idx, timestamp]`` + feature columns

        """
        super()._fit_wrap(log, user_features, item_features)

        if self._use_ann:
            vectors = self._get_vectors_to_build_ann(log)
            ann_params = self._get_ann_build_params(log)
            self._build_ann_index(vectors, **ann_params)

    @abstractmethod
    def _get_vectors_to_infer_ann_inner(
        self, log: DataFrame, users: DataFrame
    ) -> DataFrame:
        """Implementations of this method must return a dataframe with user vectors.
        User vectors from this method are used to infer the index.

        Args:
            log: DataFrame with interactions
            users: DataFrame with users

        Returns: DataFrame[user_idx int, vector array<double>] or DataFrame[vector array<double>].
        Vector column name in dataframe can be anything.
        """

    def _get_vectors_to_infer_ann(
        self, log: DataFrame, users: DataFrame, filter_seen_items: bool
    ) -> DataFrame:
        """This method wraps `_get_vectors_to_infer_ann_inner`
        and adds seen items to dataframe with user vectors by flag.

        Args:
            log: DataFrame with interactions
            users: DataFrame with users
            filter_seen_items: flag to remove seen items from recommendations based on ``log``.

        Returns:

        """
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
        """Implementation of this method must return dictionary
        with arguments for `_infer_ann_index` method.

        Returns: Dictionary with arguments to infer index. For example: {
            "features_col": "user_vector",
            "params": self._hnswlib_params,
            "index_dim": self.rank,
        }

        """

    @abstractmethod
    def _build_ann_index(  # pylint: disable=too-many-arguments
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
        """The method implements the construction of the ANN index.

        Args:
            vectors: DataFrame with vectors to build index
            features_col: Name of column from `vectors` dataframe
                that contains vectors to build index
            params: Index params
            dim: length of vectors from `vectors` dataframe
            num_elements: if `index_type` != "sparse" then
                number of elements that will be stored in the index
            id_col: Name of column that contains identifiers of vectors.
                None if `vectors` dataframe have no id column.
            index_type: "sparse" or "dense".
                If None then `index_type`="dense".
            items_count: if `index_type` == "sparse" then
                `items_count` is dimension of sparse matrix, else None

        """

    @abstractmethod
    def _infer_ann_index(  # pylint: disable=too-many-arguments
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
        """The method implements the inference of the ANN index.

        Args:
            vectors: Dataframe that contains vectors to inference
            features_col: Column name from `vectors` dataframe that contains vectors to inference
            params: Index params
            k: Desired number of neighbour vectors for vectors from `vectors` dataframe
            filter_seen_items: flag to filter seen items before output
            index_dim: Dimension of vectors in index
            index_type: "sparse" or "dense".
                If None then `index_type`="dense".
            log: DataFrame with interactions

        Returns: DataFrame[user_idx int, item_idx int, relevance double]

        """

    def _inner_predict_wrap(  # pylint: disable=too-many-arguments
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """Override base `_inner_predict_wrap` and adds ANN inference by condition"""
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

    @staticmethod
    def _unpack_infer_struct(inference_result: DataFrame) -> DataFrame:
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
