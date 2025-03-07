import logging
from typing import Any, Dict, Iterable, Optional, Union

import polars as pl

from replay.data.dataset import Dataset
from replay.utils import PandasDataFrame, PolarsDataFrame

# In this code PandasDataFrame is replaced by pl.DataFrame.
from replay.utils.polars_utils import filter_cold, get_top_k, get_unique_entities, return_recs


class _PopRecPolars:
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
        self._study = None
        self._criterion = None
        self.other_params = kwargs

    def set_params(self, **params: Dict[str, Any]) -> None:
        for param, value in params.items():
            setattr(self, param, value)

    @property
    def logger(self) -> logging.Logger:
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
    def _calc_fill(item_popularity: pl.DataFrame, weight: float, rating_column: str) -> float:
        return item_popularity[rating_column].min() * weight

    def _get_selected_item_popularity(self, items: pl.DataFrame) -> pl.DataFrame:

        df = items.join(self.item_popularity, on=self.item_column, how="left" if self.add_cold_items else "inner")
        df = df.fill_null(self.fill)
        df = df.with_columns(pl.col(self.item_column).cast(pl.Int64).alias(self.item_column))
        return df

    def _get_fit_counts(self, entity: str) -> int:
        num_entities = "_num_queries" if entity == "query" else "_num_items"
        fit_entities = self.fit_queries if entity == "query" else self.fit_items
        if not hasattr(self, num_entities):
            setattr(self, num_entities, fit_entities.height)
        return getattr(self, num_entities)

    @property
    def queries_count(self) -> int:
        return self._get_fit_counts("query")

    def fit(self, dataset: pl.DataFrame):
        self.query_column = dataset.feature_schema.query_id_column
        self.item_column = dataset.feature_schema.item_id_column
        self.rating_column = dataset.feature_schema.interactions_rating_column
        self.timestamp_column = dataset.feature_schema.interactions_timestamp_column

        # Drop duplicates using .unique()
        self.fit_items = dataset.interactions.select(self.item_column).unique()
        self.fit_queries = dataset.interactions.select(self.query_column).unique()
        # TODO: IMPORTANT: rewrite self.fit items like _BaseRecSparkImpl._fit_wrap if-else

        self._num_queries = self.fit_queries.height
        self._num_items = self.fit_items.height
        self._query_dim_size = self.fit_queries.select(pl.col(self.query_column)).max().item() + 1
        self._item_dim_size = self.fit_items.select(pl.col(self.item_column)).max().item() + 1
        interactions_df = dataset.interactions
        print(f"fit {interactions_df.shape[0]=}")
        save_df(interactions_df, "polars_base_predict_wrap_interactions")
        if self.use_rating:
            item_popularity = interactions_df.group_by(self.item_column).agg(
                pl.col(self.rating_column).sum().alias(self.rating_column)
            )
        else:
            item_popularity = interactions_df.group_by(self.item_column).agg(
                pl.col(self.query_column).n_unique().alias(self.rating_column)
            )
        print(f"fit {self.queries_count=}")
        print(f"fit {item_popularity.shape[0]=}")
        save_df(item_popularity, "polars_base_predict_wrap_item_popularity_v1")
        item_popularity = item_popularity.with_columns(
            (pl.col(self.rating_column) / self.queries_count).round(10).alias(self.rating_column)
        )
        item_popularity = item_popularity.sort(self.item_column, self.rating_column)
        print(f"fit {item_popularity.shape[0]=}")
        # print("item_popularity problem:\n", item_popularity[item_popularity[self.item_column].isin([16, 17, 24,25, 110,111, 265, 266, 296, 297])])
        save_df(item_popularity, "polars_base_predict_wrap_item_popularity_v2")
        self.item_popularity = item_popularity
        self.fill = self._calc_fill(self.item_popularity, self.cold_weight, self.rating_column)
        print(f"polars fit {self.fill=}")
        return self

    @staticmethod
    def _calc_max_hist_len(dataset: Dataset, queries: pl.DataFrame) -> int:
        query_column = dataset.feature_schema.query_id_column
        item_column = dataset.feature_schema.item_id_column
        merged_df = dataset.join(queries, on=query_column, how="left")
        grouped = merged_df.group_by(query_column).agg(pl.col(item_column).n_unique().alias("nunique"))
        max_hist_len = grouped.select(pl.col("nunique").max()).item()
        if max_hist_len is None:
            max_hist_len = 0
        return max_hist_len

    def _filter_seen(
        self, recs: pl.DataFrame, interactions: pl.DataFrame, k: int, queries: pl.DataFrame
    ) -> pl.DataFrame:
        queries_interactions = interactions.join(queries, on=self.query_column, how="inner")
        num_seen = queries_interactions.group_by(self.query_column).agg(
            pl.col(self.item_column).count().alias("seen_count")
        )
        max_seen = num_seen["seen_count"].max() if not num_seen.is_empty() else 0
        save_df(num_seen, "polars_base_predict_wrap_num_seen")
        print("\npolars_num_max_seen = ", max_seen)
        # Ranking per query; "ordinal" in 'rank' replicates pandas' method="first"
        recs = recs.with_columns(
            pl.col(self.rating_column).rank("ordinal", descending=True).over(self.query_column).alias("temp_rank")
        )
        recs = recs.filter(pl.col("temp_rank") <= (max_seen + k))
        save_df(recs, "polars_base_predict_wrap_temp_rank")
        recs = recs.join(num_seen, on=self.query_column, how="left").with_columns(pl.col("seen_count").fill_null(0))
        recs = recs.filter(pl.col("temp_rank") <= (pl.col("seen_count") + k)).drop(["temp_rank", "seen_count"])
        save_df(recs, "polars_base_predict_wrap_temp_rank_v2")
        # To filter out recommendations already seen in interactions, perform a left join with an indicator
        queries_interactions = queries_interactions.rename({self.item_column: "item", self.query_column: "query"})
        recs = recs.join(
            queries_interactions.select(["query", "item"]),
            left_on=[self.query_column, self.item_column],
            right_on=["query", "item"],
            how="anti",
        )  # TODO: check there is ok if empty dataset or nulls
        save_df(recs, "polars_base_predict_wrap_temp_rank_v3")
        return recs

    def _filter_cold_for_predict(
        self,
        main_df: pl.DataFrame,
        interactions_df: Optional[pl.DataFrame],
        entity: str,
    ):
        can_predict_cold = self.can_predict_cold_queries if entity == "query" else self.can_predict_cold_items
        fit_entities = self.fit_queries if entity == "query" else self.fit_items
        column = self.query_column if entity == "query" else self.item_column
        if can_predict_cold:
            return main_df, interactions_df

        num_new, main_df = filter_cold(main_df, fit_entities, col_name=column)
        if num_new > 0:
            self.logger.info("%s model can't predict cold %ss, they will be ignored", self, entity)
        _, interactions_df = filter_cold(interactions_df, fit_entities, col_name=column)
        return main_df, interactions_df

    def _filter_interactions_queries_items_dataframes(
        self,
        dataset: Optional[Dataset],
        k: int,
        queries: Optional[Union[pl.DataFrame, Iterable]] = None,
        items: Optional[Union[pl.DataFrame, Iterable]] = None,
    ):
        self.logger.debug("Starting predict %s", type(self).__name__)
        if dataset is not None:
            query_data = next(
                (
                    df
                    for df in [queries, dataset.interactions, dataset.query_features, self.fit_queries]
                    if df is not None and not df.is_empty()
                ),
                None,
            )
            interactions = dataset.interactions
        else:
            query_data = queries or self.fit_queries
            interactions = None

        queries = get_unique_entities(query_data, self.query_column)
        queries, interactions = self._filter_cold_for_predict(queries, interactions, "query")

        item_data = items or self.fit_items
        items = get_unique_entities(item_data, self.item_column)
        items, interactions = self._filter_cold_for_predict(items, interactions, "item")

        num_items = items.height
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

    def get_items_pd(self, items: pl.DataFrame) -> pl.DataFrame:
        selected_item_popularity = self._get_selected_item_popularity(items)
        # Replace row-wise apply with an expression
        selected_item_popularity = selected_item_popularity.with_columns(
            pl.when(pl.col(self.rating_column) == 0.0)
            .then(0.1**6)
            .otherwise(pl.col(self.rating_column))
            .alias(self.rating_column)
        )
        total_rating = selected_item_popularity[self.rating_column].sum()
        selected_item_popularity = selected_item_popularity.with_columns(
            (pl.col(self.rating_column) / total_rating).alias("probability")
        )
        return selected_item_popularity

    def _predict_without_sampling(
        self,  # TODO: дописать predict_with_sample
        dataset: Dataset,
        k: int,
        queries: pl.DataFrame,
        items: pl.DataFrame,
        filter_seen_items: bool = True,
    ) -> pl.DataFrame:
        selected_item_popularity = self._get_selected_item_popularity(items).sort("item_id", descending=True)
        save_df(selected_item_popularity, "polars_base_predict_wrap_selected_item_popularity")
        # Sort by rating and item in descending order and add a row count as rank
        sorted_df = selected_item_popularity.sort(by=[self.rating_column, self.item_column], descending=[True, True])
        sorted_df = sorted_df.with_row_count("rank", offset=1)
        selected_item_popularity = sorted_df
        save_df(selected_item_popularity, "polars_base_predict_wrap_selected_item_popularity_rank")

        if filter_seen_items and dataset is not None:
            print("INTO POLARS IF")
            queries = queries if queries is not None else self.fit_queries

            query_to_num_items = (
                dataset.interactions.join(queries, on=self.query_column, how="inner")
                .group_by(self.query_column)
                .agg(pl.col(self.item_column).n_unique().alias("num_items"))
            )
            save_df(query_to_num_items, "polars_base_predict_wrap_query_to_num_items")

            queries = queries.join(query_to_num_items, on=self.query_column, how="left").with_columns(
                pl.col("num_items").fill_null(0)
            )
            save_df(queries, "polars_base_predict_wrap_queries_v2")

            max_seen = queries.select(pl.col("num_items").max()).item()
            selected_item_popularity = selected_item_popularity.filter(pl.col("rank") <= (k + max_seen))
            # Cross join queries with the selected items
            save_df(selected_item_popularity, "polars_base_predict_wrap_selected_item_popularity_v2")
            joined = queries.join(selected_item_popularity, how="cross")
            joined = joined.filter(pl.col("rank") <= (k + pl.col("num_items")))  # .drop("rank")
            return joined
        print("NOT INTO POLARS IF")
        joined = queries.join(selected_item_popularity, how="cross")
        save_df(joined, "polars_base_predict_wrap_joined")
        return joined.filter(pl.col("rank") <= k).drop("rank")

    # TODO: РЕАЛИЗОВАТЬ predict_with_sampling в NonPersonolizedPolarsImpl

    def predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[pl.DataFrame, Iterable]] = None,
        items: Optional[Union[pl.DataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[pl.DataFrame]:

        dataset, queries, items = self._filter_interactions_queries_items_dataframes(dataset, k, queries, items)
        print(f"polars: {dataset.interactions.shape[0]=}, {queries.shape[0]=}, {items.shape[0]=}")
        save_df(dataset.interactions, "polars_base_predict_wrap_dataset")
        save_df(queries, "polars_base_predict_wrap_queries")
        save_df(items, "polars_base_predict_wrap_items")
        recs = self._predict_without_sampling(dataset, k, queries, items, filter_seen_items)
        print(f"first {recs.shape[0]=}")
        save_df(recs, "polars_base_predict_wrap_recs_1")
        if filter_seen_items and dataset is not None:
            recs = self._filter_seen(recs=recs, interactions=dataset.interactions, queries=queries, k=k)
            print(f"second {recs.shape[0]=}")
            save_df(recs, "polars_base_predict_wrap_recs_2")
        # recs = get_top_k_recs(recs, k=k, query_column=self.query_column, rating_column=self.rating_column).select(
        #    [self.query_column, self.item_column, self.rating_column]
        # )
        recs = get_top_k(recs, self.query_column, [(self.rating_column, False), (self.item_column, False)], k).select(
            self.query_column, self.item_column, self.rating_column
        )
        print(f"third {recs.shape[0]=}")
        save_df(recs, "polars_base_predict_wrap_recs_3")
        recs = return_recs(recs, recs_file_path)
        return recs

    def fit_predict(
        self,
        dataset: pl.DataFrame,
        k: int,
        queries: pl.DataFrame = None,
        items: pl.DataFrame = None,
        filter_seen_items=True,
    ) -> pl.DataFrame:
        self.fit(dataset)
        return self.predict(dataset, k, queries, items, filter_seen_items)

    def predict_pairs(self, pairs: pl.DataFrame, dataset: pl.DataFrame = None) -> pl.DataFrame:  # noqa: ARG002
        if self.item_popularity is None:
            msg = "Model not fitted. Please call fit() first."
            raise ValueError(msg)
        preds = pairs.join(self.item_popularity, on=self.item_column, how="left" if self.add_cold_items else "inner")
        fill_value = self._calc_fill(self.item_popularity, self.cold_weight, self.rating_column)
        preds = preds.with_columns(pl.col(self.rating_column).fill_null(fill_value))
        return preds.select([self.query_column, self.item_column, self.rating_column])

    def predict_proba(
        self, dataset: pl.DataFrame, k: int, queries, items: pl.DataFrame, filter_seen_items=True  # noqa: ARG002
    ) -> pl.DataFrame:
        return NotImplementedError()  # TODO:

    def save_model(self, path: str, additional_params=None):  # noqa: ARG002
        saved_params = {
            "query_column": self.query_column,
            "item_column": self.item_column,
            "rating_column": self.rating_column,
            "timestamp_column": self.timestamp_column,
        }
        if additional_params is not None:
            saved_params.update(additional_params)
            # TODO: saving
        return saved_params


def save_df(df, filename):
    if isinstance(df, PolarsDataFrame):
        df.write_parquet(filename)
        return True
    elif isinstance(df, PandasDataFrame):
        df.to_parquet(filename)
        return True
    return False
