"""
Base abstract classes:
- BaseRecommender - the simplest base class
- Recommender - base class for models that fit on interaction log
- HybridRecommender - base class for models that accept user or item features
- UserRecommender - base class that accepts only user features, but not item features
- NeighbourRec - base class that requires log at prediction time
- ItemVectorModel - class for models which provides items' vectors.
    Implements similar items search.
- NonPersonalizedRecommender - base class for non-personalized recommenders
    with popularity statistics
"""

from abc import ABC, abstractmethod
from os.path import join
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy.random import default_rng

from replay.data import get_schema
from replay.experimental.utils.session_handler import State
from replay.models.base_rec import IsSavable, RecommenderCommons
from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame, SparkDataFrame
from replay.utils.spark_utils import (
    convert2spark,
    cosine_similarity,
    filter_cold,
    get_top_k,
    get_top_k_recs,
    get_unique_entities,
    load_pickled_from_parquet,
    return_recs,
    save_picklable_to_parquet,
    vector_dot,
    vector_euclidean_distance_similarity,
)

if PYSPARK_AVAILABLE:
    from pyspark.sql import (
        Window,
        functions as sf,
    )


class BaseRecommender(RecommenderCommons, IsSavable, ABC):
    """Base recommender"""

    model: Any
    can_predict_cold_users: bool = False
    can_predict_cold_items: bool = False
    fit_users: SparkDataFrame
    fit_items: SparkDataFrame
    _num_users: int
    _num_items: int
    _user_dim_size: int
    _item_dim_size: int

    def _fit_wrap(
        self,
        log: SparkDataFrame,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
    ) -> None:
        """
        Wrapper for fit to allow for fewer arguments in a model.

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features
            ``[user_idx, timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx, timestamp]`` + feature columns
        :return:
        """
        self.logger.debug("Starting fit %s", type(self).__name__)
        if user_features is None:
            users = log.select("user_idx").distinct()
        else:
            users = log.select("user_idx").union(user_features.select("user_idx")).distinct()
        if item_features is None:
            items = log.select("item_idx").distinct()
        else:
            items = log.select("item_idx").union(item_features.select("item_idx")).distinct()
        self.fit_users = sf.broadcast(users)
        self.fit_items = sf.broadcast(items)
        self._num_users = self.fit_users.count()
        self._num_items = self.fit_items.count()
        self._user_dim_size = self.fit_users.agg({"user_idx": "max"}).first()[0] + 1
        self._item_dim_size = self.fit_items.agg({"item_idx": "max"}).first()[0] + 1
        self._fit(log, user_features, item_features)

    @abstractmethod
    def _fit(
        self,
        log: SparkDataFrame,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
    ) -> None:
        """
        Inner method where model actually fits.

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features
            ``[user_idx, timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx, timestamp]`` + feature columns
        :return:
        """

    def _filter_seen(self, recs: SparkDataFrame, log: SparkDataFrame, k: int, users: SparkDataFrame):
        """
        Filter seen items (presented in log) out of the users' recommendations.
        For each user return from `k` to `k + number of seen by user` recommendations.
        """
        users_log = log.join(users, on="user_idx")
        self._cache_model_temp_view(users_log, "filter_seen_users_log")
        num_seen = users_log.groupBy("user_idx").agg(sf.count("item_idx").alias("seen_count"))
        self._cache_model_temp_view(num_seen, "filter_seen_num_seen")

        # count maximal number of items seen by users
        max_seen = 0
        if num_seen.count() > 0:
            max_seen = num_seen.select(sf.max("seen_count")).first()[0]

        # crop recommendations to first k + max_seen items for each user
        recs = recs.withColumn(
            "temp_rank",
            sf.row_number().over(Window.partitionBy("user_idx").orderBy(sf.col("relevance").desc())),
        ).filter(sf.col("temp_rank") <= sf.lit(max_seen + k))

        # leave k + number of items seen by user recommendations in recs
        recs = (
            recs.join(num_seen, on="user_idx", how="left")
            .fillna(0)
            .filter(sf.col("temp_rank") <= sf.col("seen_count") + sf.lit(k))
            .drop("temp_rank", "seen_count")
        )

        # filter recommendations presented in interactions log
        recs = recs.join(
            users_log.withColumnRenamed("item_idx", "item")
            .withColumnRenamed("user_idx", "user")
            .select("user", "item"),
            on=(sf.col("user_idx") == sf.col("user")) & (sf.col("item_idx") == sf.col("item")),
            how="anti",
        ).drop("user", "item")

        return recs

    def _filter_log_users_items_dataframes(
        self,
        log: Optional[SparkDataFrame],
        k: int,
        users: Optional[Union[SparkDataFrame, Iterable]] = None,
        items: Optional[Union[SparkDataFrame, Iterable]] = None,
        user_features: Optional[SparkDataFrame] = None,
    ):
        """
        Returns triplet of filtered `log`, `users`, and `items`.
        Filters out cold entities (users/items) from the `users`/`items` and `log` dataframes
        if the model does not predict cold.
        Filters out duplicates from `users` and `items` dataframes,
        and excludes all columns except `user_idx` and `item_idx`.

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :return: triplet of filtered `log`, `users`, and `items` dataframes.
        """
        self.logger.debug("Starting predict %s", type(self).__name__)
        user_data = users or log or user_features or self.fit_users
        users = get_unique_entities(user_data, "user_idx")
        users, log = self._filter_cold_for_predict(users, log, "user")

        item_data = items or self.fit_items
        items = get_unique_entities(item_data, "item_idx")
        items, log = self._filter_cold_for_predict(items, log, "item")
        num_items = items.count()
        if num_items < k:
            message = f"k = {k} > number of items = {num_items}"
            self.logger.debug(message)
        return log, users, items

    def _predict_wrap(
        self,
        log: Optional[SparkDataFrame],
        k: int,
        users: Optional[Union[SparkDataFrame, Iterable]] = None,
        items: Optional[Union[SparkDataFrame, Iterable]] = None,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[SparkDataFrame]:
        """
        Predict wrapper to allow for fewer parameters in models

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        log, users, items = self._filter_log_users_items_dataframes(log, k, users, items)

        recs = self._predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )
        if filter_seen_items and log:
            recs = self._filter_seen(recs=recs, log=log, users=users, k=k)

        recs = get_top_k_recs(recs, k=k).select("user_idx", "item_idx", "relevance")

        output = return_recs(recs, recs_file_path)
        self._clear_model_temp_view("filter_seen_users_log")
        self._clear_model_temp_view("filter_seen_num_seen")
        return output

    def _filter_cold_for_predict(
        self,
        main_df: SparkDataFrame,
        log_df: Optional[SparkDataFrame],
        entity: str,
        suffix: str = "idx",
    ):
        """
        Filter out cold entities (users/items) from the `main_df` and `log_df`
        if the model does not predict cold.
        Warn if cold entities are present in the `main_df`.
        """
        if getattr(self, f"can_predict_cold_{entity}s"):
            return main_df, log_df

        fit_entities = getattr(self, f"fit_{entity}s")

        num_new, main_df = filter_cold(main_df, fit_entities, col_name=f"{entity}_{suffix}")
        if num_new > 0:
            self.logger.info(
                "%s model can't predict cold %ss, they will be ignored",
                self,
                entity,
            )
        _, log_df = filter_cold(log_df, fit_entities, col_name=f"{entity}_{suffix}")
        return main_df, log_df

    @abstractmethod
    def _predict(
        self,
        log: SparkDataFrame,
        k: int,
        users: SparkDataFrame,
        items: SparkDataFrame,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        """
        Inner method where model actually predicts.

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :return: recommendation dataframe
            ``[user_idx, item_idx, relevance]``
        """

    def _predict_proba(
        self,
        log: SparkDataFrame,
        k: int,
        users: SparkDataFrame,
        items: SparkDataFrame,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
        filter_seen_items: bool = True,
    ) -> np.ndarray:
        """
        Inner method where model actually predicts probability estimates.

        Mainly used in ```OBPOfflinePolicyLearner```.

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, rating]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``rating`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :return: distribution over items for each user with shape
            ``(n_users, n_items, k)``
            where we have probability for each user to choose item at fixed position(top-k).
        """

        n_users = users.select("user_idx").count()
        n_items = items.select("item_idx").count()

        recs = self._predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )

        recs = get_top_k_recs(recs, k=k).select("user_idx", "item_idx", "relevance")

        cols = [f"k{i}" for i in range(k)]

        recs_items = (
            recs.groupBy("user_idx")
            .agg(sf.collect_list("item_idx").alias("item_idx"))
            .select([sf.col("item_idx")[i].alias(cols[i]) for i in range(k)])
        )

        action_dist = np.zeros(shape=(n_users, n_items, k))

        for i in range(k):
            action_dist[
                np.arange(n_users),
                recs_items.select(cols[i]).toPandas()[cols[i]].to_numpy(),
                np.ones(n_users, dtype=int) * i,
            ] += 1

        return action_dist

    def _get_fit_counts(self, entity: str) -> int:
        if not hasattr(self, f"_num_{entity}s"):
            setattr(
                self,
                f"_num_{entity}s",
                getattr(self, f"fit_{entity}s").count(),
            )
        return getattr(self, f"_num_{entity}s")

    @property
    def users_count(self) -> int:
        """
        :returns: number of users the model was trained on
        """
        return self._get_fit_counts("user")

    @property
    def items_count(self) -> int:
        """
        :returns: number of items the model was trained on
        """
        return self._get_fit_counts("item")

    def _get_fit_dims(self, entity: str) -> int:
        if not hasattr(self, f"_{entity}_dim_size"):
            setattr(
                self,
                f"_{entity}_dim_size",
                getattr(self, f"fit_{entity}s").agg({f"{entity}_idx": "max"}).first()[0] + 1,
            )
        return getattr(self, f"_{entity}_dim_size")

    @property
    def _user_dim(self) -> int:
        """
        :returns: dimension of users matrix (maximal user idx + 1)
        """
        return self._get_fit_dims("user")

    @property
    def _item_dim(self) -> int:
        """
        :returns: dimension of items matrix (maximal item idx + 1)
        """
        return self._get_fit_dims("item")

    def _fit_predict(
        self,
        log: SparkDataFrame,
        k: int,
        users: Optional[Union[SparkDataFrame, Iterable]] = None,
        items: Optional[Union[SparkDataFrame, Iterable]] = None,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[SparkDataFrame]:
        self._fit_wrap(log, user_features, item_features)
        return self._predict_wrap(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def _predict_pairs_wrap(
        self,
        pairs: SparkDataFrame,
        log: Optional[SparkDataFrame] = None,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[SparkDataFrame]:
        """
        This method
        1) converts data to spark
        2) removes cold users and items if model does not predict them
        3) calls inner _predict_pairs method of a model

        :param pairs: user-item pairs to get relevance for,
            dataframe containing``[user_idx, item_idx]``.
        :param log: train data
            ``[user_idx, item_idx, timestamp, relevance]``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        log, user_features, item_features, pairs = [
            convert2spark(df) for df in [log, user_features, item_features, pairs]
        ]
        if sorted(pairs.columns) != ["item_idx", "user_idx"]:
            msg = "pairs must be a dataframe with columns strictly [user_idx, item_idx]"
            raise ValueError(msg)
        pairs, log = self._filter_cold_for_predict(pairs, log, "user")
        pairs, log = self._filter_cold_for_predict(pairs, log, "item")

        pred = self._predict_pairs(
            pairs=pairs,
            log=log,
            user_features=user_features,
            item_features=item_features,
        )

        if k:
            pred = get_top_k(
                dataframe=pred,
                partition_by_col=sf.col("user_idx"),
                order_by_col=[
                    sf.col("relevance").desc(),
                ],
                k=k,
            )

        if recs_file_path is not None:
            pred.write.parquet(path=recs_file_path, mode="overwrite")
            return None

        pred.cache().count()
        return pred

    def _predict_pairs(
        self,
        pairs: SparkDataFrame,
        log: Optional[SparkDataFrame] = None,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
    ) -> SparkDataFrame:
        """
        Fallback method to use in case ``_predict_pairs`` is not implemented.
        Simply joins ``predict`` with given ``pairs``.
        :param pairs: user-item pairs to get relevance for,
            dataframe containing``[user_idx, item_idx]``.
        :param log: train data
            ``[user_idx, item_idx, timestamp, relevance]``.
        """
        message = (
            "native predict_pairs is not implemented for this model. "
            "Falling back to usual predict method and filtering the results."
        )
        self.logger.warning(message)

        users = pairs.select("user_idx").distinct()
        items = pairs.select("item_idx").distinct()
        k = items.count()
        pred = self._predict(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=False,
        )

        pred = pred.join(
            pairs.select("user_idx", "item_idx"),
            on=["user_idx", "item_idx"],
            how="inner",
        )
        return pred

    def _get_features_wrap(
        self, ids: SparkDataFrame, features: Optional[SparkDataFrame]
    ) -> Optional[Tuple[SparkDataFrame, int]]:
        if "user_idx" not in ids.columns and "item_idx" not in ids.columns:
            msg = "user_idx or item_idx missing"
            raise ValueError(msg)
        vectors, rank = self._get_features(ids, features)
        return vectors, rank

    def _get_features(
        self, ids: SparkDataFrame, features: Optional[SparkDataFrame]  # noqa: ARG002
    ) -> Tuple[Optional[SparkDataFrame], Optional[int]]:
        """
        Get embeddings from model

        :param ids: id ids to get embeddings for Spark DataFrame containing user_idx or item_idx
        :param features: user or item features
        :return: DataFrame with biases and embeddings, and vector size
        """

        self.logger.info(
            "get_features method is not defined for the model %s. Features will not be returned.",
            str(self),
        )
        return None, None

    def _get_nearest_items_wrap(
        self,
        items: Union[SparkDataFrame, Iterable],
        k: int,
        metric: Optional[str] = "cosine_similarity",
        candidates: Optional[Union[SparkDataFrame, Iterable]] = None,
    ) -> Optional[SparkDataFrame]:
        """
        Convert indexes and leave top-k nearest items for each item in `items`.
        """
        items = get_unique_entities(items, "item_idx")
        if candidates is not None:
            candidates = get_unique_entities(candidates, "item_idx")

        nearest_items_to_filter = self._get_nearest_items(
            items=items,
            metric=metric,
            candidates=candidates,
        )

        rel_col_name = metric if metric is not None else "similarity"
        nearest_items = get_top_k(
            dataframe=nearest_items_to_filter,
            partition_by_col=sf.col("item_idx_one"),
            order_by_col=[
                sf.col(rel_col_name).desc(),
                sf.col("item_idx_two").desc(),
            ],
            k=k,
        )

        nearest_items = nearest_items.withColumnRenamed("item_idx_two", "neighbour_item_idx")
        nearest_items = nearest_items.withColumnRenamed("item_idx_one", "item_idx")
        return nearest_items

    def _get_nearest_items(
        self,
        items: SparkDataFrame,  # noqa: ARG002
        metric: Optional[str] = None,  # noqa: ARG002
        candidates: Optional[SparkDataFrame] = None,  # noqa: ARG002
    ) -> Optional[SparkDataFrame]:
        msg = f"item-to-item prediction is not implemented for {self}"
        raise NotImplementedError(msg)

    def _save_model(self, path: str):
        pass

    def _load_model(self, path: str):
        pass


class ItemVectorModel(BaseRecommender):
    """Parent for models generating items' vector representations"""

    can_predict_item_to_item: bool = True
    item_to_item_metrics: List[str] = [
        "euclidean_distance_sim",
        "cosine_similarity",
        "dot_product",
    ]

    @abstractmethod
    def _get_item_vectors(self) -> SparkDataFrame:
        """
        Return dataframe with items' vectors as a
            spark dataframe with columns ``[item_idx, item_vector]``
        """

    def get_nearest_items(
        self,
        items: Union[SparkDataFrame, Iterable],
        k: int,
        metric: Optional[str] = "cosine_similarity",
        candidates: Optional[Union[SparkDataFrame, Iterable]] = None,
    ) -> Optional[SparkDataFrame]:
        """
        Get k most similar items be the `metric` for each of the `items`.

        :param items: spark dataframe or list of item ids to find neighbors
        :param k: number of neighbors
        :param metric: 'euclidean_distance_sim', 'cosine_similarity', 'dot_product'
        :param candidates: spark dataframe or list of items
            to consider as similar, e.g. popular/new items. If None,
            all items presented during model training are used.
        :return: dataframe with the most similar items,
            where bigger value means greater similarity.
            spark-dataframe with columns ``[item_idx, neighbour_item_idx, similarity]``
        """
        if metric not in self.item_to_item_metrics:
            msg = f"Select one of the valid distance metrics: {self.item_to_item_metrics}"
            raise ValueError(msg)

        return self._get_nearest_items_wrap(
            items=items,
            k=k,
            metric=metric,
            candidates=candidates,
        )

    def _get_nearest_items(
        self,
        items: SparkDataFrame,
        metric: str = "cosine_similarity",
        candidates: Optional[SparkDataFrame] = None,
    ) -> SparkDataFrame:
        """
        Return distance metric value for all available close items filtered by `candidates`.

        :param items: ids to find neighbours, spark dataframe with column ``item_idx``
        :param metric: 'euclidean_distance_sim' calculated as 1/(1 + euclidean_distance),
            'cosine_similarity', 'dot_product'
        :param candidates: items among which we are looking for similar,
            e.g. popular/new items. If None, all items presented during model training are used.
        :return: dataframe with neighbours,
            spark-dataframe with columns ``[item_idx_one, item_idx_two, similarity]``
        """
        dist_function = cosine_similarity
        if metric == "euclidean_distance_sim":
            dist_function = vector_euclidean_distance_similarity
        elif metric == "dot_product":
            dist_function = vector_dot

        items_vectors = self._get_item_vectors()
        left_part = (
            items_vectors.withColumnRenamed("item_idx", "item_idx_one")
            .withColumnRenamed("item_vector", "item_vector_one")
            .join(
                items.select(sf.col("item_idx").alias("item_idx_one")),
                on="item_idx_one",
            )
        )

        right_part = items_vectors.withColumnRenamed("item_idx", "item_idx_two").withColumnRenamed(
            "item_vector", "item_vector_two"
        )

        if candidates is not None:
            right_part = right_part.join(
                candidates.withColumnRenamed("item_idx", "item_idx_two"),
                on="item_idx_two",
            )

        joined_factors = left_part.join(right_part, on=sf.col("item_idx_one") != sf.col("item_idx_two"))

        joined_factors = joined_factors.withColumn(
            metric,
            dist_function(sf.col("item_vector_one"), sf.col("item_vector_two")),
        )

        similarity_matrix = joined_factors.select("item_idx_one", "item_idx_two", metric)

        return similarity_matrix


class HybridRecommender(BaseRecommender, ABC):
    """Base class for models that can use extra features"""

    def fit(
        self,
        log: SparkDataFrame,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
    ) -> None:
        """
        Fit a recommendation model

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features
            ``[user_idx, timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx, timestamp]`` + feature columns
        :return:
        """
        self._fit_wrap(
            log=log,
            user_features=user_features,
            item_features=item_features,
        )

    def predict(
        self,
        log: SparkDataFrame,
        k: int,
        users: Optional[Union[SparkDataFrame, Iterable]] = None,
        items: Optional[Union[SparkDataFrame, Iterable]] = None,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[SparkDataFrame]:
        """
        Get recommendations

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided

        """
        return self._predict_wrap(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def fit_predict(
        self,
        log: SparkDataFrame,
        k: int,
        users: Optional[Union[SparkDataFrame, Iterable]] = None,
        items: Optional[Union[SparkDataFrame, Iterable]] = None,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[SparkDataFrame]:
        """
        Fit model and get recommendations

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._fit_predict(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def predict_pairs(
        self,
        pairs: SparkDataFrame,
        log: Optional[SparkDataFrame] = None,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[SparkDataFrame]:
        """
        Get recommendations for specific user-item ``pairs``.
        If a model can't produce recommendation
        for specific pair it is removed from the resulting dataframe.

        :param pairs: dataframe with pairs to calculate relevance for, ``[user_idx, item_idx]``.
        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :param k: top-k items for each user from pairs.
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._predict_pairs_wrap(
            pairs=pairs,
            log=log,
            user_features=user_features,
            item_features=item_features,
            recs_file_path=recs_file_path,
            k=k,
        )

    def get_features(
        self, ids: SparkDataFrame, features: Optional[SparkDataFrame]
    ) -> Optional[Tuple[SparkDataFrame, int]]:
        """
        Returns user or item feature vectors as a Column with type ArrayType
        :param ids: Spark DataFrame with unique ids
        :param features: Spark DataFrame with features for provided ids
        :return: feature vectors
            If a model does not have a vector for some ids they are not present in the final result.
        """
        return self._get_features_wrap(ids, features)


class Recommender(BaseRecommender, ABC):
    """Usual recommender class for models without features."""

    def fit(self, log: SparkDataFrame) -> None:
        """
        Fit a recommendation model

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :return:
        """
        self._fit_wrap(
            log=log,
            user_features=None,
            item_features=None,
        )

    def predict(
        self,
        log: SparkDataFrame,
        k: int,
        users: Optional[Union[SparkDataFrame, Iterable]] = None,
        items: Optional[Union[SparkDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[SparkDataFrame]:
        """
        Get recommendations

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._predict_wrap(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=None,
            item_features=None,
            filter_seen_items=filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def predict_pairs(
        self,
        pairs: SparkDataFrame,
        log: Optional[SparkDataFrame] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[SparkDataFrame]:
        """
        Get recommendations for specific user-item ``pairs``.
        If a model can't produce recommendation
        for specific pair it is removed from the resulting dataframe.

        :param pairs: dataframe with pairs to calculate relevance for, ``[user_idx, item_idx]``.
        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :param k: top-k items for each user from pairs.
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._predict_pairs_wrap(
            pairs=pairs,
            log=log,
            recs_file_path=recs_file_path,
            k=k,
        )

    def fit_predict(
        self,
        log: SparkDataFrame,
        k: int,
        users: Optional[Union[SparkDataFrame, Iterable]] = None,
        items: Optional[Union[SparkDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[SparkDataFrame]:
        """
        Fit model and get recommendations

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._fit_predict(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=None,
            item_features=None,
            filter_seen_items=filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def get_features(self, ids: SparkDataFrame) -> Optional[Tuple[SparkDataFrame, int]]:
        """
        Returns user or item feature vectors as a Column with type ArrayType

        :param ids: Spark DataFrame with unique ids
        :return: feature vectors.
            If a model does not have a vector for some ids they are not present in the final result.
        """
        return self._get_features_wrap(ids, None)


class UserRecommender(BaseRecommender, ABC):
    """Base class for models that use user features
    but not item features. ``log`` is not required for this class."""

    def fit(
        self,
        log: SparkDataFrame,
        user_features: SparkDataFrame,
    ) -> None:
        """
        Finds user clusters and calculates item similarity in that clusters.

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features
            ``[user_idx, timestamp]`` + feature columns
        :return:
        """
        self._fit_wrap(log=log, user_features=user_features)

    def predict(
        self,
        user_features: SparkDataFrame,
        k: int,
        log: Optional[SparkDataFrame] = None,
        users: Optional[Union[SparkDataFrame, Iterable]] = None,
        items: Optional[Union[SparkDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[SparkDataFrame]:
        """
        Get recommendations

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._predict_wrap(
            log=log,
            user_features=user_features,
            k=k,
            filter_seen_items=filter_seen_items,
            users=users,
            items=items,
            recs_file_path=recs_file_path,
        )

    def predict_pairs(
        self,
        pairs: SparkDataFrame,
        user_features: SparkDataFrame,
        log: Optional[SparkDataFrame] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[SparkDataFrame]:
        """
        Get recommendations for specific user-item ``pairs``.
        If a model can't produce recommendation
        for specific pair it is removed from the resulting dataframe.

        :param pairs: dataframe with pairs to calculate relevance for, ``[user_idx, item_idx]``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :param k: top-k items for each user from pairs.
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._predict_pairs_wrap(
            pairs=pairs,
            log=log,
            user_features=user_features,
            recs_file_path=recs_file_path,
            k=k,
        )


class NonPersonalizedRecommender(Recommender, ABC):
    """Base class for non-personalized recommenders with popularity statistics."""

    can_predict_cold_users = True
    can_predict_cold_items = True
    item_popularity: SparkDataFrame
    add_cold_items: bool
    cold_weight: float
    sample: bool
    fill: float
    seed: Optional[int] = None

    def __init__(self, add_cold_items: bool, cold_weight: float):
        self.add_cold_items = add_cold_items
        if 0 < cold_weight <= 1:
            self.cold_weight = cold_weight
        else:
            msg = "`cold_weight` value should be in interval (0, 1]"
            raise ValueError(msg)

    @property
    def _dataframes(self):
        return {"item_popularity": self.item_popularity}

    def _save_model(self, path: str):
        save_picklable_to_parquet(self.fill, join(path, "params.dump"))

    def _load_model(self, path: str):
        self.fill = load_pickled_from_parquet(join(path, "params.dump"))

    def _clear_cache(self):
        if hasattr(self, "item_popularity"):
            self.item_popularity.unpersist()

    @staticmethod
    def _calc_fill(item_popularity: SparkDataFrame, weight: float) -> float:
        """
        Calculating a fill value a the minimal relevance
        calculated during model training multiplied by weight.
        """
        return item_popularity.select(sf.min("relevance")).first()[0] * weight

    @staticmethod
    def _check_relevance(log: SparkDataFrame):
        vals = log.select("relevance").where((sf.col("relevance") != 1) & (sf.col("relevance") != 0))
        if vals.count() > 0:
            msg = "Relevance values in log must be 0 or 1"
            raise ValueError(msg)

    def _get_selected_item_popularity(self, items: SparkDataFrame) -> SparkDataFrame:
        """
        Choose only required item from `item_popularity` dataframe
        for further recommendations generation.
        """
        return self.item_popularity.join(
            items,
            on="item_idx",
            how="right" if self.add_cold_items else "inner",
        ).fillna(value=self.fill, subset=["relevance"])

    @staticmethod
    def _calc_max_hist_len(log: SparkDataFrame, users: SparkDataFrame) -> int:
        max_hist_len = (
            (log.join(users, on="user_idx").groupBy("user_idx").agg(sf.countDistinct("item_idx").alias("items_count")))
            .select(sf.max("items_count"))
            .first()[0]
        )
        # all users have empty history
        if max_hist_len is None:
            max_hist_len = 0

        return max_hist_len

    def _predict_without_sampling(
        self,
        log: SparkDataFrame,
        k: int,
        users: SparkDataFrame,
        items: SparkDataFrame,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        """
        Regular prediction for popularity-based models,
        top-k most relevant items from `items` are chosen for each user
        """
        selected_item_popularity = self._get_selected_item_popularity(items)
        selected_item_popularity = selected_item_popularity.withColumn(
            "rank",
            sf.row_number().over(Window.orderBy(sf.col("relevance").desc(), sf.col("item_idx").desc())),
        )

        if filter_seen_items and log is not None:
            user_to_num_items = (
                log.join(users, on="user_idx").groupBy("user_idx").agg(sf.countDistinct("item_idx").alias("num_items"))
            )
            users = users.join(user_to_num_items, on="user_idx", how="left")
            users = users.fillna(0, "num_items")
            # 'selected_item_popularity' truncation by k + max_seen
            max_seen = users.select(sf.coalesce(sf.max("num_items"), sf.lit(0))).first()[0]
            selected_item_popularity = selected_item_popularity.filter(sf.col("rank") <= k + max_seen)
            return users.join(selected_item_popularity, on=(sf.col("rank") <= k + sf.col("num_items")), how="left")

        return users.crossJoin(selected_item_popularity.filter(sf.col("rank") <= k)).drop("rank")

    def _predict_with_sampling(
        self,
        log: SparkDataFrame,
        k: int,
        users: SparkDataFrame,
        items: SparkDataFrame,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        """
        Randomized prediction for popularity-based models,
        top-k items from `items` are sampled for each user based with
        probability proportional to items' popularity
        """
        selected_item_popularity = self._get_selected_item_popularity(items)
        selected_item_popularity = selected_item_popularity.withColumn(
            "relevance",
            sf.when(sf.col("relevance") == sf.lit(0.0), 0.1**6).otherwise(sf.col("relevance")),
        )

        items_pd = selected_item_popularity.withColumn(
            "probability",
            sf.col("relevance") / selected_item_popularity.select(sf.sum("relevance")).first()[0],
        ).toPandas()

        rec_schema = get_schema(
            query_column="user_idx",
            item_column="item_idx",
            rating_column="relevance",
            has_timestamp=False,
        )
        if items_pd.shape[0] == 0:
            return State().session.createDataFrame([], rec_schema)

        seed = self.seed
        class_name = self.__class__.__name__

        def grouped_map(pandas_df: PandasDataFrame) -> PandasDataFrame:
            user_idx = pandas_df["user_idx"][0]
            cnt = pandas_df["cnt"][0]

            local_rng = default_rng(seed + user_idx) if seed is not None else default_rng()

            items_positions = local_rng.choice(
                np.arange(items_pd.shape[0]),
                size=cnt,
                p=items_pd["probability"].values,
                replace=False,
            )

            # workaround to unify RandomRec and UCB
            if class_name == "RandomRec":
                relevance = 1 / np.arange(1, cnt + 1)
            else:
                relevance = items_pd["probability"].values[items_positions]

            return PandasDataFrame(
                {
                    "user_idx": cnt * [user_idx],
                    "item_idx": items_pd["item_idx"].values[items_positions],
                    "relevance": relevance,
                }
            )

        if log is not None and filter_seen_items:
            recs = (
                log.select("user_idx", "item_idx")
                .distinct()
                .join(users, how="right", on="user_idx")
                .groupby("user_idx")
                .agg(sf.countDistinct("item_idx").alias("cnt"))
                .selectExpr(
                    "user_idx",
                    f"LEAST(cnt + {k}, {items_pd.shape[0]}) AS cnt",
                )
            )
        else:
            recs = users.withColumn("cnt", sf.lit(min(k, items_pd.shape[0])))

        return recs.groupby("user_idx").applyInPandas(grouped_map, rec_schema)

    def _predict(
        self,
        log: SparkDataFrame,
        k: int,
        users: SparkDataFrame,
        items: SparkDataFrame,
        user_features: Optional[SparkDataFrame] = None,  # noqa: ARG002
        item_features: Optional[SparkDataFrame] = None,  # noqa: ARG002
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        if self.sample:
            return self._predict_with_sampling(
                log=log,
                k=k,
                users=users,
                items=items,
                filter_seen_items=filter_seen_items,
            )
        else:
            return self._predict_without_sampling(log, k, users, items, filter_seen_items)

    def _predict_pairs(
        self,
        pairs: SparkDataFrame,
        log: Optional[SparkDataFrame] = None,  # noqa: ARG002
        user_features: Optional[SparkDataFrame] = None,  # noqa: ARG002
        item_features: Optional[SparkDataFrame] = None,  # noqa: ARG002
    ) -> SparkDataFrame:
        return (
            pairs.join(
                self.item_popularity,
                on="item_idx",
                how="left" if self.add_cold_items else "inner",
            )
            .fillna(value=self.fill, subset=["relevance"])
            .select("user_idx", "item_idx", "relevance")
        )
