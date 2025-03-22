import logging
from os.path import join
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import pandas as pd

from replay.data.dataset import Dataset
from replay.utils import PandasDataFrame
from replay.utils.pandas_utils import (
    filter_cold,
    get_top_k,
    get_unique_entities,
    load_pickled_from_parquet,
    return_recs,
    save_picklable_to_parquet,
)


class _PopRecPandas:
    model: int
    items_count: Optional[int] = None

    def __init__(
        self,
        use_rating: bool = False,
        add_cold_items: bool = True,
        cold_weight: float = 0.5,
        sample=False,
        fill=0.0,
        seed=42,
        **kwargs,
    ):
        self.use_rating = use_rating
        self.sample = sample
        self.fill = fill
        self.seed = seed
        self.add_cold_items = add_cold_items
        self.can_predict_cold_queries = True
        self.can_predict_cold_items = True
        self.cold_weight = cold_weight
        self.item_popularity = None
        self.fit_items = None
        self.fit_queries = None
        self._logger = None
        self._search_space = None
        self._objective = None
        self.study = None
        self.criterion = None
        self.other_params = kwargs

    def set_params(self, **params: Dict[str, Any]) -> None:
        """
        Set model parameters

        :param params: dictionary param name - param value
        :return:
        """
        for param, value in params.items():
            setattr(self, param, value)

    @property
    def logger(self) -> logging.Logger:
        """
        :returns: get library logger
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger

    @property
    def _init_args(self):
        return {
            "use_rating": self.use_rating,
            "add_cold_items": self.add_cold_items,
            "cold_weight": self.cold_weight,
        }

    @staticmethod
    def _calc_fill(item_popularity: PandasDataFrame, weight: float, rating_column: str) -> float:
        """
        Calculating a fill value a the minimal rating
        calculated during model training multiplied by weight.
        """
        return item_popularity[rating_column].min() * weight

    def __str__(self):
        return type(self).__name__

    def _get_selected_item_popularity(self, items: PandasDataFrame) -> PandasDataFrame:
        """
        Choose only required item from `item_popularity` dataframe
        for further recommendations generation.
        """
        df = self.item_popularity.merge(items, on=self.item_column, how="right" if self.add_cold_items else "inner")
        df = df.fillna(value=self.fill)
        return df

    def get_features(
        self, ids: PandasDataFrame, features: Optional[PandasDataFrame] = None
    ) -> Optional[Tuple[PandasDataFrame, int]]:
        if self.query_column not in ids.columns and self.item_column not in ids.columns:
            msg = f"{self.query_column} or {self.item_column} missing"
            raise ValueError(msg)
        vectors, rank = self._get_features(ids, features)

        return vectors, rank

    def _get_features(
        self, ids: PandasDataFrame, features: Optional[PandasDataFrame]  # noqa: ARG002
    ) -> Tuple[Optional[PandasDataFrame], Optional[int]]:
        """
        Get embeddings from model

        :param ids: id ids to get embeddings for Spark DataFrame containing user_idx or item_idx
        :param features: query or item features
        :return: SparkDataFrame with biases and embeddings, and vector size
        """

        self.logger.info(
            "get_features method is not defined for the model %s. Features will not be returned.",
            str(self),
        )
        return None, None

    def _get_fit_counts(self, entity: str) -> int:
        num_entities = "_num_queries" if entity == "query" else "_num_items"
        fit_entities = self.fit_queries if entity == "query" else self.fit_items
        if not hasattr(self, num_entities):
            setattr(
                self,
                num_entities,
                fit_entities.count(),
            )
        return getattr(self, num_entities)

    @property
    def queries_count(self) -> int:
        """
        :returns: number of queries the model was trained on
        """
        return self._get_fit_counts("query")

    @property
    def items_count(self) -> int:
        """
        :returns: number of items the model was trained on
        """
        return self._get_fit_counts("item")

    def fit(self, dataset: PandasDataFrame):
        self.query_column = dataset.feature_schema.query_id_column
        self.item_column = dataset.feature_schema.item_id_column
        self.rating_column = dataset.feature_schema.interactions_rating_column
        self.timestamp_column = dataset.feature_schema.interactions_timestamp_column
        if dataset.query_features is None:
            self.fit_queries = dataset.interactions[[self.query_column]].drop_duplicates()
        else:
            self.fit_queries = pd.concat(
                [dataset.interactions[[self.query_column]], dataset.query_features[[self.query_column]]],
                ignore_index=True,
            ).drop_duplicates()

        if dataset.item_features is None:
            self.fit_items = dataset.interactions[[self.item_column]].drop_duplicates()
        else:
            self.fit_items = pd.concat(
                [dataset.interactions[[self.item_column]], dataset.item_features[[self.item_column]]], ignore_index=True
            ).drop_duplicates()

        self._num_queries = self.fit_queries.shape[0]
        self._num_items = self.fit_items.shape[0]
        self._query_dim_size = self.fit_queries.max() + 1
        self._item_dim_size = self.fit_items.max() + 1
        interactions_df = dataset.interactions
        if self.use_rating:
            item_popularity = interactions_df.groupby(self.item_column, as_index=False)[self.rating_column].sum()
        else:
            item_popularity = interactions_df.groupby(self.item_column, as_index=False)[self.query_column].nunique()
            item_popularity.rename(columns={self.query_column: self.rating_column}, inplace=True)
        item_popularity[self.rating_column] = round(item_popularity[self.rating_column] / self.queries_count, 10)
        item_popularity = item_popularity.sort_values([self.item_column, self.rating_column])
        self.item_popularity = item_popularity
        self.fill = self._calc_fill(self.item_popularity, self.cold_weight, self.rating_column)
        return self

    @staticmethod
    def _calc_max_hist_len(dataset: Dataset, queries: PandasDataFrame) -> int:
        query_column = dataset.feature_schema.query_id_column
        item_column = dataset.feature_schema.item_id_column
        merged_df = dataset.interactions.merge(queries, on=query_column)

        grouped_df = merged_df.groupby(query_column)[item_column].nunique()
        max_hist_len = grouped_df.max() if grouped_df is not None and not grouped_df.empty else 0

        return max_hist_len

    def _filter_seen(
        self, recs: PandasDataFrame, interactions: PandasDataFrame, k: int, queries: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter seen items (present in interactions) out of the queries' recommendations.
        For each query return from 'k' to 'k + number of seen by query' recommendations.
        """

        queries_interactions = interactions.merge(queries, on=self.query_column, how="inner")

        num_seen = (
            queries_interactions.groupby(self.query_column)
            .agg({self.item_column: "count"})
            .rename(columns={self.item_column: "seen_count"})
            .reset_index()
        )
        max_seen = num_seen["seen_count"].max() if not num_seen.empty else 0
        # Rank recommendations to first k + max_seen items for each query
        recs = recs.sort_values(by=[self.query_column, self.rating_column], ascending=[True, False])
        recs["temp_rank"] = recs.groupby(self.query_column).cumcount() + 1
        recs = recs[recs["temp_rank"] <= max_seen + k]
        recs = recs.merge(num_seen, on=self.query_column, how="left").fillna({"seen_count": 0})

        # Filter based on ranking
        recs = recs[recs["temp_rank"] <= recs["seen_count"] + k].drop(columns=["temp_rank", "seen_count"])
        # Filter recommendations presented in interactions
        queries_interactions = queries_interactions.rename(
            columns={self.item_column: "item", self.query_column: "query"}
        )
        recs = recs.merge(
            queries_interactions[["query", "item"]],
            left_on=[self.query_column, self.item_column],
            right_on=["query", "item"],
            how="left",
            indicator=True,
        )
        recs = recs[recs["_merge"] == "left_only"].drop(columns=["query", "item", "_merge"])
        return recs

    def _filter_cold_for_predict(
        self,
        main_df: PandasDataFrame,
        interactions_df: Optional[pd.DataFrame],
        entity: str,
    ):
        """
        Filter out cold entities (queries/items) from the `main_df` and `interactions_df`
        if the model does not predict cold.
        Warn if cold entities are present in the `main_df`.
        """
        can_predict_cold = self.can_predict_cold_queries if entity == "query" else self.can_predict_cold_items
        fit_entities = self.fit_queries if entity == "query" else self.fit_items
        column = self.query_column if entity == "query" else self.item_column
        if can_predict_cold:
            return main_df, interactions_df

        num_new, main_df = filter_cold(main_df, fit_entities, col_name=column)  # pragma: no cover
        if num_new > 0:  # pragma: no cover
            self.logger.info(
                "%s model can't predict cold %ss, they will be ignored",
                self,
                entity,
            )
        _, interactions_df = filter_cold(interactions_df, fit_entities, col_name=column)  # pragma: no cover
        main_df.reset_index(inplace=True)  # pragma: no cover
        interactions_df.reset_index(inplace=True)  # pragma: no cover
        return main_df, interactions_df  # pragma: no cover

    def _filter_interactions_queries_items_dataframes(
        self,
        dataset: Optional[Dataset],
        k: int,
        queries: Optional[Union[PandasDataFrame, Iterable]] = None,
        items: Optional[Union[PandasDataFrame, Iterable]] = None,
    ):
        """
        Returns triplet of filtered `dataset`, `queries`, and `items`.
        Filters out cold entities (queries/items) from the `queries`/`items` and `dataset`
        if the model does not predict cold.
        Filters out duplicates from `queries` and `items` dataframes,
        and excludes all columns except `user_idx` and `item_idx`.

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :param k: number of recommendations for each user
        :param queries: queries to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all queries from ``dataset``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``dataset``.
            If it contains new items, ``rating`` for them will be ``0``.
        :return: triplet of filtered `dataset`, `queries`, and `items`.
        """
        self.logger.debug("Starting predict %s", type(self).__name__)
        if dataset is not None:
            query_data = next(
                (
                    df
                    for df in [queries, dataset.interactions, dataset.query_features, self.fit_queries]
                    if df is not None
                    and ((isinstance(df, (list, tuple, set))) or (isinstance(df, PandasDataFrame) and not df.empty))
                ),
                None,
            )
            interactions = dataset.interactions
        else:
            query_data = next(
                (
                    df
                    for df in [queries, self.fit_queries]
                    if df is not None
                    and ((isinstance(df, (list, tuple, set))) or (isinstance(df, PandasDataFrame) and not df.empty))
                ),
                None,
            )
            interactions = None

        queries = get_unique_entities(query_data, self.query_column)
        queries, interactions = self._filter_cold_for_predict(queries, interactions, "query")

        item_data = next(
            (
                df
                for df in [items, self.fit_items]
                if df is not None
                and ((isinstance(df, (list, tuple, set))) or (isinstance(df, PandasDataFrame) and not df.empty))
            ),
            None,
        )
        items = get_unique_entities(item_data, self.item_column)
        items, interactions = self._filter_cold_for_predict(items, interactions, "item")
        num_items = len(items)
        if num_items < k:
            message = f"k = {k} > number of items = {num_items}"
            self.logger.debug(message)

        if dataset is not None:
            dataset = Dataset(
                feature_schema=dataset.feature_schema,
                interactions=interactions,
                query_features=dataset.query_features,
                item_features=dataset.item_features,
                check_consistency=False,
            )
        return dataset, queries, items

    def _predict_without_sampling(
        self,
        dataset: Dataset,
        k: int,
        queries: PandasDataFrame,
        items: PandasDataFrame,
        filter_seen_items: bool = True,
    ) -> PandasDataFrame:
        """
        Regular prediction for popularity-based models,
        top-k most relevant items from `items` are chosen for each query
        """
        selected_item_popularity = self._get_selected_item_popularity(items).sort_values(
            self.item_column, ascending=False
        )
        selected_item_popularity = selected_item_popularity.sort_values(
            by=[self.rating_column, self.item_column], ascending=[False, False]
        ).reset_index(
            drop=True
        )  # TODO: Think about to remove sorting if tests is ok. In other place, like utils too
        selected_item_popularity["rank"] = range(1, len(selected_item_popularity) + 1)

        if filter_seen_items and dataset is not None:
            queries = PandasDataFrame(queries)
            query_to_num_items = (
                dataset.interactions.merge(queries, on=self.query_column)
                .groupby(self.query_column, as_index=False)[self.item_column]
                .nunique()
            ).rename(columns={self.item_column: "num_items"})
            queries = queries.merge(query_to_num_items, on=self.query_column, how="left")
            queries = queries.fillna(0)
            max_seen = queries["num_items"].max()  # noqa: F841
            selected_item_popularity = selected_item_popularity.query("rank <= @k + @max_seen")
            joined = queries.merge(selected_item_popularity, how="cross")
            return joined[joined["rank"] <= (k + joined["num_items"])]
        joined = queries.merge(selected_item_popularity[selected_item_popularity["rank"] <= k], how="cross")
        return joined.drop("rank", axis=1)

    def _predict_with_sampling(
        self,
        dataset: Dataset,
        k: int,
        queries: PandasDataFrame,
        items: PandasDataFrame,
        filter_seen_items: bool = True,
    ) -> PandasDataFrame:  # pragma: no cover
        # TODO: In Other NonPersonolizedRecommeder models we need to use _predict_with sample
        raise NotImplementedError()

    def predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[PandasDataFrame, Iterable]] = None,
        items: Optional[Union[PandasDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[PandasDataFrame]:
        """
        Predict wrapper to allow for fewer parameters in models

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :param k: number of recommendations for each user
        :param queries: queries to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all queries from ``interactions``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``interactions``.
            If it contains new items, ``rating`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``interactions``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, rating]``
            or None if `file_path` is provided
        """
        dataset, queries, items = self._filter_interactions_queries_items_dataframes(dataset, k, queries, items)
        recs = self._predict_without_sampling(dataset, k, queries, items, filter_seen_items)
        if filter_seen_items and dataset is not None:
            recs = self._filter_seen(recs=recs, interactions=dataset.interactions, queries=queries, k=k)
        recs = get_top_k(recs, self.query_column, [(self.rating_column, False), (self.item_column, True)], k)[
            [self.query_column, self.item_column, self.rating_column]
        ].reset_index(drop=True)
        recs = return_recs(recs, recs_file_path)
        return recs

    def fit_predict(
        self,
        dataset: PandasDataFrame,
        k: int,
        queries: PandasDataFrame = None,
        items: PandasDataFrame = None,
        filter_seen_items=True,
        recs_file_path: Optional[str] = None,
    ) -> PandasDataFrame:
        self.fit(dataset)
        return self.predict(dataset, k, queries, items, filter_seen_items, recs_file_path)

    def predict_pairs(
        self,
        pairs: PandasDataFrame,
        dataset: Optional[Dataset] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[PandasDataFrame]:
        if dataset is not None:
            interactions, query_features, item_features = (
                dataset.interactions,
                dataset.query_features,
                dataset.item_features,
            )
            if set(pairs.columns) != {self.item_column, self.query_column}:
                msg = "pairs must be a dataframe with columns strictly [user_idx, item_idx]"
                raise ValueError(msg)
            pairs, interactions = self._filter_cold_for_predict(pairs, interactions, "query")
            pairs, interactions = self._filter_cold_for_predict(pairs, interactions, "item")
            dataset = Dataset(
                feature_schema=dataset.feature_schema,
                interactions=interactions,
                query_features=query_features,
                item_features=item_features,
            )
        if self.item_popularity is None:
            msg = "Model not fitted. Please call fit() first."
            raise ValueError(msg)
        pred = pairs.merge(self.item_popularity, on=self.item_column, how="left" if self.add_cold_items else "inner")
        fill_value = self._calc_fill(self.item_popularity, self.cold_weight, self.rating_column)
        pred[self.rating_column].fillna(fill_value, inplace=True)
        pred = pred[[self.query_column, self.item_column, self.rating_column]]
        if k:
            pred = get_top_k(pred, self.query_column, [(self.rating_column, False), (self.item_column, True)], k)

        if recs_file_path is not None:
            pred.to_parquet(recs_file_path)  # it's overwrite operation
            return None
        return pred

    def _predict_proba(
        self, dataset: PandasDataFrame, k: int, queries, items: PandasDataFrame, filter_seen_items=True
    ) -> PandasDataFrame:  # pragma: no cover
        # TODO: Implement it in NonPersonolizedRecommender, if you need this function in other models
        raise NotImplementedError()

    def _save_model(self, path: str, additional_params=None):
        saved_params = {
            "query_column": self.query_column,
            "item_column": self.item_column,
            "rating_column": self.rating_column,
            "timestamp_column": self.timestamp_column,
        }
        if additional_params is not None:
            saved_params.update(additional_params)
        save_picklable_to_parquet(saved_params, join(path, "params.dump"))
        return saved_params

    def _load_model(self, path: str):
        loaded_params = load_pickled_from_parquet(join(path, "params.dump"))
        for param, value in loaded_params.items():
            setattr(self, param, value)
