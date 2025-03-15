import logging
from os.path import join
from typing import Any, Dict, Iterable, Optional, Union

import polars as pl

from replay.data.dataset import Dataset
from replay.utils import PolarsDataFrame
from replay.utils.pandas_utils import load_pickled_from_parquet, save_picklable_to_parquet

# In this code PandasDataFrame is replaced by pl.DataFrame.
from replay.utils.polars_utils import (
    filter_cold,
    get_top_k,
    get_unique_entities,
    return_recs,
)


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
    def _calc_fill(item_popularity: PolarsDataFrame, weight: float, rating_column: str) -> float:
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

        if dataset.query_features is None:
            self.fit_queries = dataset.interactions.select(self.query_column).unique()
        else:
            self.fit_queries = pl.concat(
                [dataset.interactions.select(self.query_column), dataset.query_features.select(self.query_column)]
            ).unique()

        # For items
        if dataset.item_features is None:
            self.fit_items = dataset.interactions.select(self.item_column).unique()
        else:
            self.fit_items = pl.concat(
                [dataset.interactions.select(self.item_column), dataset.item_features.select(self.item_column)]
            ).unique()

        self._num_queries = self.fit_queries.height
        self._num_items = self.fit_items.height
        self._query_dim_size = self.fit_queries.select(pl.col(self.query_column)).max().item() + 1
        self._item_dim_size = self.fit_items.select(pl.col(self.item_column)).max().item() + 1
        interactions_df = dataset.interactions
        if self.use_rating:
            item_popularity = interactions_df.group_by(self.item_column).agg(
                pl.col(self.rating_column).sum().alias(self.rating_column)
            )
        else:
            item_popularity = interactions_df.group_by(self.item_column).agg(
                pl.col(self.query_column).n_unique().alias(self.rating_column)
            )
        item_popularity = item_popularity.with_columns(
            (pl.col(self.rating_column) / self.queries_count).round(10).alias(self.rating_column)
        )
        item_popularity = item_popularity.sort(self.item_column, self.rating_column)
        self.item_popularity = item_popularity
        self.fill = self._calc_fill(self.item_popularity, self.cold_weight, self.rating_column)
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
        self, recs: PolarsDataFrame, interactions: PolarsDataFrame, k: int, queries: pl.DataFrame
    ) -> pl.DataFrame:
        queries = queries.with_columns(pl.col(self.query_column).cast(pl.Int64).alias(self.query_column))
        interactions = interactions.with_columns(pl.col(self.query_column).cast(pl.Int64).alias(self.query_column))
        queries_interactions = interactions.join(queries, on=self.query_column, how="inner")
        num_seen = queries_interactions.group_by(self.query_column).agg(
            pl.col(self.item_column).count().alias("seen_count")
        )
        max_seen = num_seen["seen_count"].max() if not num_seen.is_empty() else 0
        # Ranking per query; "ordinal" in 'rank' replicates pandas' method="first"
        recs = recs.with_columns(
            pl.col(self.rating_column).rank("ordinal", descending=True).over(self.query_column).alias("temp_rank")
        )
        recs = recs.filter(pl.col("temp_rank") <= (max_seen + k))
        recs = recs.join(num_seen, on=self.query_column, how="left").with_columns(pl.col("seen_count").fill_null(0))
        recs = recs.filter(pl.col("temp_rank") <= (pl.col("seen_count") + k)).drop(["temp_rank", "seen_count"])
        queries_interactions = queries_interactions.rename({self.item_column: "item", self.query_column: "query"})
        if recs.is_empty():
            recs = recs.with_columns(pl.lit(None).alias(self.item_column)).limit(0)
        queries_interactions = queries_interactions.with_columns(pl.col("item").cast(pl.Int64).alias("item"))
        recs = recs.with_columns(pl.col(self.item_column).cast(pl.Int64).alias(self.item_column))
        recs = recs.join(
            queries_interactions.select(["query", "item"]),
            left_on=[self.query_column, self.item_column],
            right_on=["query", "item"],
            how="anti",
        )
        return recs

    def _filter_cold_for_predict(
        self,
        main_df: PolarsDataFrame,
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
        queries: Optional[Union[PolarsDataFrame, Iterable]] = None,
        items: Optional[Union[PolarsDataFrame, Iterable]] = None,
    ):
        self.logger.debug("Starting predict %s", type(self).__name__)
        if dataset is not None:
            query_data = next(
                (
                    df
                    for df in [queries, dataset.interactions, dataset.query_features, self.fit_queries]
                    if df is not None
                    and (
                        (isinstance(df, (list, tuple, set))) or (isinstance(df, PolarsDataFrame) and not df.is_empty())
                    )
                ),
                None,
            )
            interactions = dataset.interactions
            interactions = interactions.with_columns(pl.col(self.query_column).cast(pl.Int64).alias(self.query_column))
        else:
            query_data = next(
                (
                    df
                    for df in [queries, self.fit_queries]
                    if df is not None
                    and (
                        (isinstance(df, (list, tuple, set))) or (isinstance(df, PolarsDataFrame) and not df.is_empty())
                    )
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
                and ((isinstance(df, (list, tuple, set))) or (isinstance(df, PolarsDataFrame) and not df.is_empty()))
            ),
            None,
        )
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
        self,
        dataset: Dataset,
        k: int,
        queries: PolarsDataFrame,
        items: PolarsDataFrame,
        filter_seen_items: bool = True,
    ) -> pl.DataFrame:
        selected_item_popularity = self._get_selected_item_popularity(items)
        if not selected_item_popularity.is_empty():
            selected_item_popularity = selected_item_popularity.sort(self.item_column, descending=True)
            # Sort by rating and item in descending order and add a row count as rank
            sorted_df = selected_item_popularity.sort(
                by=[self.rating_column, self.item_column], descending=[True, True]
            )
            selected_item_popularity = sorted_df.with_row_count("rank", offset=1)
        else:
            selected_item_popularity = (
                selected_item_popularity.with_columns(
                    pl.lit(None).alias(self.item_column), pl.lit(None).alias(self.rating_column)
                )
                .with_row_count("rank", offset=1)
                .limit(0)
            )

        if filter_seen_items and dataset is not None:
            queries = queries if queries is not None else self.fit_queries
            queries = queries.with_columns(pl.col(self.query_column).cast(pl.Int64).alias(self.query_column))

            query_to_num_items = (
                dataset.interactions.join(queries, on=self.query_column, how="inner")
                .group_by(self.query_column)
                .agg(pl.col(self.item_column).n_unique().alias("num_items"))
            )
            queries = queries.join(query_to_num_items, on=self.query_column, how="left").with_columns(
                pl.col("num_items").fill_null(0)
            )

            max_seen = queries.select(pl.col("num_items").max()).item()
            selected_item_popularity = selected_item_popularity.filter(pl.col("rank") <= (k + max_seen))
            # Cross join queries with the selected items
            joined = queries.join(selected_item_popularity, how="cross")
            joined = joined.filter(pl.col("rank") <= (k + pl.col("num_items")))  # .drop("rank")
            return joined
        joined = queries.join(selected_item_popularity, how="cross")
        return joined.filter(pl.col("rank") <= k).drop("rank")

    def _predict_with_sampling(
        self,
        dataset: Dataset,
        k: int,
        queries: PolarsDataFrame,
        items: PolarsDataFrame,
        filter_seen_items: bool = True,
    ) -> pl.DataFrame:
        # TODO: In Other NonPersonolizedRecommeder models we need to use _predict_with_sample
        raise NotImplementedError()

    def predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[PolarsDataFrame, Iterable]] = None,
        items: Optional[Union[PolarsDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[pl.DataFrame]:

        dataset, queries, items = self._filter_interactions_queries_items_dataframes(dataset, k, queries, items)
        recs = self._predict_without_sampling(dataset, k, queries, items, filter_seen_items)
        if filter_seen_items and dataset is not None:
            recs = self._filter_seen(recs=recs, interactions=dataset.interactions, queries=queries, k=k)
        if not recs.is_empty():
            recs = get_top_k(
                recs, self.query_column, [(self.rating_column, False), (self.item_column, True)], k
            ).select(self.query_column, self.item_column, self.rating_column)
        recs = return_recs(recs, recs_file_path)
        return recs

    def fit_predict(
        self,
        dataset: PolarsDataFrame,
        k: int,
        queries: pl.DataFrame = None,
        items: pl.DataFrame = None,
        filter_seen_items=True,
    ) -> pl.DataFrame:
        self.fit(dataset)
        return self.predict(dataset, k, queries, items, filter_seen_items)

    def predict_pairs(
        self,
        pairs: PolarsDataFrame,
        dataset: Optional[Dataset] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[pl.DataFrame]:
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
        pred = pairs.join(self.item_popularity, on=self.item_column, how="left" if self.add_cold_items else "inner")
        fill_value = self._calc_fill(self.item_popularity, self.cold_weight, self.rating_column)
        pred = pred.with_columns(pl.col(self.rating_column).fill_null(fill_value))
        pred = pred.select([self.query_column, self.item_column, self.rating_column])
        if k:
            pred = get_top_k(pred, self.query_column, [(self.rating_column, False), (self.item_column, True)], k)

        if recs_file_path is not None:
            pred.write_parquet(recs_file_path)  # it's overwrite operation
            return None
        return pred

    def _predict_proba(
        self, dataset: PolarsDataFrame, k: int, queries, items: PolarsDataFrame, filter_seen_items=True
    ) -> pl.DataFrame:
        # TODO: Implement it in NonPersonolizedRecommender, if you need this function in other models
        raise NotImplementedError()

    def save_model(self, path: str, additional_params=None):
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

    def load_model(self, path: str):
        loaded_params = load_pickled_from_parquet(join(path, "params.dump"))
        for param, value in loaded_params.items():
            setattr(self, param, value)
