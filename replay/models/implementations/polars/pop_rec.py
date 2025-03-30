import logging
from os.path import join
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import polars as pl

from replay.data.dataset import Dataset
from replay.utils import PandasDataFrame, PolarsDataFrame
#from replay.utils.pandas_utils import load_pickled_from_parquet, save_picklable_to_parquet
from replay.utils.spark_utils import load_pickled_from_parquet, save_picklable_to_parquet
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
        self.study = None
        self.criterion = None
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
        return item_popularity.select(pl.min(rating_column)).item() * weight

    def __str__(self):
        return type(self).__name__

    def _get_selected_item_popularity(self, items: pl.DataFrame) -> pl.DataFrame:
        print("get_select pandas")
        print(self.item_popularity)
        if self.item_popularity.select(self.item_column).null_count().item() == 0 and not self.item_popularity.is_empty() and not items.is_empty():
            self.item_popularity = self.item_popularity.with_columns(pl.col(self.item_column).round().cast(pl.Int64))
            items = items.with_columns(pl.col(self.item_column).round().cast(pl.Int64))
        else:
            self.item_popularity = self.item_popularity.with_columns(pl.col(self.item_column).cast(pl.Float64))
            items = items.with_columns(pl.col(self.item_column).cast(pl.Float64))
        df = items.join(self.item_popularity, on=self.item_column, how="left" if self.add_cold_items else "inner")
        if not df.is_empty():
            df = df.with_columns(pl.col(self.rating_column).fill_null(self.fill).alias(self.rating_column))
            df = df.with_columns(pl.col(self.item_column).round().cast(pl.Int64).alias(self.item_column))
        return df

    def _get_fit_counts(self, entity: str) -> int:
        num_entities = "_num_queries" if entity == "query" else "_num_items"
        return getattr(self, num_entities)

    def get_features(
        self, ids: PolarsDataFrame, features: Optional[PolarsDataFrame] = None
    ) -> Optional[Tuple[PolarsDataFrame, int]]:
        if self.query_column not in ids.columns and self.item_column not in ids.columns:
            msg = f"{self.query_column} or {self.item_column} missing"
            raise ValueError(msg)
        vectors, rank = self._get_features(ids, features)

        return vectors, rank

    def _get_features(
        self, ids: PolarsDataFrame, features: Optional[PolarsDataFrame]  # noqa: ARG002
    ) -> Tuple[Optional[PolarsDataFrame], Optional[int]]:
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

    @property
    def _dataframes(self):
        return {"item_popularity": self.item_popularity}

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
        print("Polars two dataframes:")
        print(dataset.interactions.select(self.query_column))
        print("Polars two dataframes with casting")
        print(dataset.interactions.select(pl.col(self.query_column)).unique().height)
        print(dataset.interactions.select(pl.col(self.query_column)).unique())
        if dataset.item_features is None:
            self.fit_items = dataset.interactions.select(self.item_column).unique()
        else:
            self.fit_items = pl.concat(
                [dataset.interactions.select(pl.col(self.item_column).cast(pl.Float64).alias(self.item_column)), 
                 dataset.item_features.select(pl.col(self.item_column).cast(pl.Float64).alias(self.item_column))
                 ]
            ).select(pl.col(self.item_column).cast(pl.Int64).alias(self.item_column)).unique()
        print("\nFITTED QUERIES POLARS:")
        print(self.fit_queries)
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
                pl.col(self.query_column).drop_nulls().n_unique().alias(self.rating_column)
            )
        print("FIT POPPULARITY  before Polars")
        print(f"{self.queries_count=}")
        print(item_popularity)
        item_popularity = item_popularity.with_columns(
            (pl.col(self.rating_column) / self.queries_count).round(10).alias(self.rating_column)
        )
        item_popularity = item_popularity.sort(self.item_column, self.rating_column, nulls_last=True)
        if not item_popularity.is_empty():
            self.item_popularity = item_popularity.with_columns(pl.col(self.item_column).round().cast(pl.Int64))
        self.fill = self._calc_fill(self.item_popularity, self.cold_weight, self.rating_column)
        return self

    @staticmethod
    def _calc_max_hist_len(dataset: Dataset, queries: pl.DataFrame) -> int:
        query_column = dataset.feature_schema.query_id_column
        item_column = dataset.feature_schema.item_id_column
        merged_df = dataset.interactions.join(queries, on=query_column)
        grouped = merged_df.group_by(query_column).agg(pl.col(item_column).drop_nulls().n_unique().alias("nunique"))
        max_hist_len = grouped.select(pl.col("nunique").max()).item()
        if max_hist_len is None:
            max_hist_len = 0
        return max_hist_len

    def _filter_seen(
        self, recs: PolarsDataFrame, interactions: PolarsDataFrame, k: int, queries: pl.DataFrame
    ) -> pl.DataFrame:
        if not queries.is_empty() and not interactions.is_empty():
            queries = queries.with_columns(pl.col(self.query_column).round().cast(pl.Int64).alias(self.query_column))
            interactions = interactions.with_columns(pl.col(self.query_column).round().cast(pl.Int64).alias(self.query_column))
            recs = recs.with_columns(pl.col(self.query_column).round().cast(pl.Int64).alias(self.query_column))
        else:
            queries = queries.with_columns(pl.col(self.query_column).cast(pl.Float64).alias(self.query_column))
            interactions = interactions.with_columns(pl.col(self.query_column).cast(pl.Float64).alias(self.query_column))
            recs = recs.with_columns(pl.col(self.query_column).cast(pl.Float64).alias(self.query_column))

        queries_interactions = interactions.join(queries, on=self.query_column, how="inner")
        num_seen = queries_interactions.group_by(self.query_column).agg(
            pl.col(self.item_column).count().alias("seen_count")
        )
        max_seen = num_seen["seen_count"].max() if not num_seen.is_empty() else 0
        # Ranking per query: "ordinal" in 'rank' replicates pandas method="first"
        recs = recs.with_columns(
            pl.col(self.rating_column).rank("ordinal", descending=True).over(self.query_column).alias("temp_rank")
        )
        recs = recs.filter(pl.col("temp_rank") <= (max_seen + k))
        recs = recs.join(num_seen, on=self.query_column, how="left").with_columns(pl.col("seen_count").fill_null(0))
        recs = recs.filter(pl.col("temp_rank") <= (pl.col("seen_count") + k)).drop(["temp_rank", "seen_count"])
        print("RECS POLARS")
        print(recs)
        queries_interactions = queries_interactions.rename({self.item_column: "item", self.query_column: "query"})
        if recs.is_empty():
            recs = recs.with_columns(pl.lit(None).alias(self.item_column)).limit(0)
        if not queries_interactions.is_empty() and not recs.is_empty():
            queries_interactions = queries_interactions.with_columns(pl.col("item").round().cast(pl.Int64).alias("item"))
            recs = recs.with_columns(pl.col(self.item_column).round().cast(pl.Int64).alias(self.item_column))
        else:
            queries_interactions = queries_interactions.with_columns(pl.col("item").cast(pl.Float64).alias("item"))
            recs = recs.with_columns(pl.col(self.item_column).cast(pl.Float64).alias(self.item_column))
            
        recs = recs.join(
            queries_interactions.select(["query", "item"]),
            left_on=[self.query_column, self.item_column],
            right_on=["query", "item"],
            how="anti",
        )
        recs = recs.with_columns(pl.col(self.item_column).round().cast(pl.Int64).alias(self.item_column))
            
        return recs

    def _filter_cold_for_predict(
        self,
        main_df: PolarsDataFrame,
        interactions_df: Optional[pl.DataFrame],
        entity: str,
    ) -> Tuple[PolarsDataFrame, PolarsDataFrame]:
        can_predict_cold = self.can_predict_cold_queries if entity == "query" else self.can_predict_cold_items
        fit_entities = self.fit_queries if entity == "query" else self.fit_items
        column = self.query_column if entity == "query" else self.item_column
        if can_predict_cold:
            return main_df, interactions_df

        num_new, main_df = filter_cold(main_df, fit_entities, col_name=column)  # pragma: no cover
        if num_new > 0:  # pragma: no cover
            self.logger.info("%s model can't predict cold %ss, they will be ignored", self, entity)
        _, interactions_df = filter_cold(interactions_df, fit_entities, col_name=column)  # pragma: no cover
        return main_df, interactions_df  # pragma: no cover

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
            if not interactions.is_empty():
                interactions = interactions.with_columns(pl.col(self.query_column).round().cast(pl.Int64).alias(self.query_column))
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

    def get_items_pd(self, items: pl.DataFrame) -> PandasDataFrame:
        selected_item_popularity = self._get_selected_item_popularity(items)
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
        return selected_item_popularity.to_pandas()

    def _predict_without_sampling(
        self,
        dataset: Dataset,
        k: int,
        queries: PolarsDataFrame,
        items: PolarsDataFrame,
        filter_seen_items: bool = True,
    ) -> pl.DataFrame:
        selected_item_popularity = self._get_selected_item_popularity(items)
        print("polars selected_item_popularity")
        print(selected_item_popularity)
        if not selected_item_popularity.is_empty():
            selected_item_popularity = selected_item_popularity.sort(self.item_column, descending=True, nulls_last=True)
            sorted_df = selected_item_popularity.sort(
                by=[self.rating_column, self.item_column], descending=[True, True],
                nulls_last=True
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
        print("polars selected_item_popularity2")
        print(selected_item_popularity)
        if filter_seen_items and dataset is not None:
            queries = queries if queries is not None else self.fit_queries
            if not queries.is_empty():
                dtype_query = dataset.interactions.select(self.query_column).dtypes[0]
                queries = queries.with_columns(pl.col(self.query_column).round().cast(dtype_query).alias(self.query_column))

            query_to_num_items = (
                dataset.interactions.join(queries, on=self.query_column, how="inner")
                .group_by(pl.col(self.query_column).drop_nulls())
                .agg(pl.col(self.item_column).drop_nulls().n_unique().alias("num_items"))
            )
            queries = queries.join(query_to_num_items, on=self.query_column, how="left").with_columns(
                pl.col("num_items").fill_null(0)
            )
            max_seen = queries.select(pl.col("num_items").max()).item()
            selected_item_popularity = selected_item_popularity.filter(pl.col("rank") <= (k + max_seen))
            joined = queries.join(selected_item_popularity, how="cross")
            joined = joined.filter(pl.col("rank") <= (k + pl.col("num_items")))
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
    ) -> pl.DataFrame:  # pragma: no cover
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
        print("polars_recs")
        print(recs)
        if filter_seen_items and dataset is not None:
            recs = self._filter_seen(recs=recs, interactions=dataset.interactions, queries=queries, k=k)
        print("polars_recs2")
        print(recs)
        if not recs.is_empty():
            recs = get_top_k(
                recs, self.query_column, [(self.rating_column, False), (self.item_column, True)], k
            )
        print("polars_recs3")
        print(recs)
        if not recs.is_empty:
            recs = recs.with_columns(pl.col(self.query_column).round().cast(pl.Int64).alias(self.query_column))
            recs = recs.with_columns(pl.col(self.item_column).round().cast(pl.Int64).alias(self.item_column))
        recs = recs.select(
            self.query_column, 
            self.item_column, 
            self.rating_column
        ).drop_nulls() # is it okey?
        print("polars_recs3.5")
        print(recs)
        recs = return_recs(recs, recs_file_path)
        return recs

    def fit_predict(
        self,
        dataset: PolarsDataFrame,
        k: int,
        queries: pl.DataFrame = None,
        items: pl.DataFrame = None,
        filter_seen_items=True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[pl.DataFrame]:
        self.fit(dataset)
        return self.predict(dataset, k, queries, items, filter_seen_items, recs_file_path)

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
        pairs = pairs.with_columns(pl.col(self.item_column).cast(pl.Int64).alias(self.item_column))
        pred = pairs.join(self.item_popularity, on=self.item_column, how="left" if self.add_cold_items else "inner")
        fill_value = self._calc_fill(self.item_popularity, self.cold_weight, self.rating_column)
        pred = pred.with_columns(pl.col(self.rating_column).fill_null(fill_value))
        pred = pred.select([self.query_column, self.item_column, self.rating_column])
        if k:
            pred = get_top_k(pred, self.query_column, [(self.rating_column, False), (self.item_column, True)], k)
        
        if recs_file_path is not None:
            pred.write_parquet(recs_file_path)  # it's overwrite operation - as expected
            return None
        return pred

    def _predict_proba(
        self, dataset: PolarsDataFrame, k: int, queries, items: PolarsDataFrame, filter_seen_items=True
    ) -> pl.DataFrame:  # pragma: no cover
        # TODO: Implement it in NonPersonolizedRecommender, if you need this function in other models
        raise NotImplementedError()

    def _save_model(  # pragma: no cover
        self, path: str, additional_params=None
    ):  # TODO: Think how to save models like on spark(utils.save)
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

    def _load_model(self, path: str):  # pragma: no cover # TODO: Think how to load models like on spark(utils.save)
        loaded_params = load_pickled_from_parquet(join(path, "params.dump"))
        for param, value in loaded_params.items():
            setattr(self, param, value)
