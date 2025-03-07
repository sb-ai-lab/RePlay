import logging
from typing import Any, Dict, Iterable, Optional, Union

import pandas as pd

from replay.data.dataset import Dataset
from replay.utils import PandasDataFrame, SparkDataFrame
from replay.utils.pandas_utils import filter_cold, get_top_k, get_unique_entities, return_recs


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
        self._study = None
        self._criterion = None
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

    def _get_selected_item_popularity(self, items: PandasDataFrame) -> PandasDataFrame:
        """
        Choose only required item from `item_popularity` dataframe
        for further recommendations generation.
        """
        df = self.item_popularity.merge(items, on=self.item_column, how="right" if self.add_cold_items else "inner")
        df = df.fillna(value=self.fill)
        return df

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

    def fit(self, dataset: PandasDataFrame):
        self.query_column = dataset.feature_schema.query_id_column
        self.item_column = dataset.feature_schema.item_id_column
        self.rating_column = dataset.feature_schema.interactions_rating_column
        self.timestamp_column = dataset.feature_schema.interactions_timestamp_column
        self.fit_items = pd.DataFrame(dataset.interactions[self.item_column].drop_duplicates()) 
        # TODO: IMPORTANT: rewrite self.fit items like _BaseRecSparkImpl._fit_wrap if-else
        self.fit_queries = pd.DataFrame(dataset.interactions[self.query_column].drop_duplicates())
        self._num_queries = self.fit_queries.shape[0]
        self._num_items = self.fit_items.shape[0]
        self._query_dim_size = self.fit_queries.max() + 1
        self._item_dim_size = self.fit_items.max() + 1
        interactions_df = dataset.interactions
        print(f"fit {interactions_df.shape[0]=}")
        save_df(interactions_df, "pandas_base_predict_wrap_interactions")
        if self.use_rating:
            item_popularity = interactions_df.groupby(self.item_column, as_index=False)[self.rating_column].sum()
        else:
            item_popularity = interactions_df.groupby(self.item_column, as_index=False)[self.query_column].nunique()
            item_popularity.rename(columns={self.query_column: self.rating_column}, inplace=True)
        print(f"fit {self.queries_count=}")
        print(f"fit {item_popularity.shape[0]=}")
        save_df(item_popularity, "pandas_base_predict_wrap_item_popularity_v1")
        item_popularity[self.rating_column] = round(item_popularity[self.rating_column] / self.queries_count, 10) # TODO: поменять располжение во всех фреймворках
        item_popularity = item_popularity.sort_values([self.item_column, self.rating_column])
        print(f"fit {item_popularity.shape[0]=}")
        save_df(item_popularity, "pandas_base_predict_wrap_item_popularity_v2")
       
        self.item_popularity = item_popularity
        self.fill = self._calc_fill(self.item_popularity, self.cold_weight, self.rating_column)
        print(f"pandas fit {self.fill=}")
        return self

    @staticmethod
    def _calc_max_hist_len(dataset: Dataset, queries: PandasDataFrame) -> int:
        query_column = dataset.feature_schema.query_id_column
        item_column = dataset.feature_schema.item_id_column
        merged_df = dataset.merge(queries, on=query_column, how="left")

        grouped_df = merged_df.groupby(query_column)[item_column].nunique()
        max_hist_len = grouped_df.max()
        if max_hist_len is None:
            max_hist_len = 0

        return max_hist_len

    def _filter_seen(
        self, recs: pd.DataFrame, interactions: pd.DataFrame, k: int, queries: pd.DataFrame
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
        save_df(num_seen, "pandas_base_predict_wrap_num_seen")
        print("\npandas_num_max_seen = ", max_seen)
        # Rank recommendations to first k + max_seen items for each query
        #old_method: recs["temp_rank"] = recs.groupby(self.query_column)[self.rating_column].rank(method="first", ascending=False)
        recs = recs.sort_values(by=[self.query_column, self.rating_column], ascending=[True, False])
        recs["temp_rank"] = recs.groupby(self.query_column).cumcount() + 1
        recs = recs[recs["temp_rank"] <= max_seen + k]
        save_df(recs, "pandas_base_predict_wrap_temp_rank")
        recs = recs.merge(num_seen, on=self.query_column, how="left").fillna({"seen_count": 0})

        # Filter based on ranking
        recs = recs[recs["temp_rank"] <= recs["seen_count"] + k].drop(columns=["temp_rank", "seen_count"])
        save_df(recs, "pandas_base_predict_wrap_temp_rank_v2")
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
        recs = recs[recs["_merge"] == "left_only"].drop(
            columns=["query", "item", "_merge"]
        )  # TODO: check its all ok with join
        save_df(recs, "pandas_base_predict_wrap_temp_rank_v3")
        return recs

    def _filter_cold_for_predict(
        self,
        main_df: pd.DataFrame,
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

        num_new, main_df = filter_cold(main_df, fit_entities, col_name=column)
        if num_new > 0:
            self.logger.info(
                "%s model can't predict cold %ss, they will be ignored",
                self,
                entity,
            )
        _, interactions_df = filter_cold(interactions_df, fit_entities, col_name=column)
        return main_df, interactions_df

    def _filter_interactions_queries_items_dataframes(
        self,
        dataset: Optional[Dataset],
        k: int,
        queries: Optional[Union[pd.DataFrame, Iterable]] = None,
        items: Optional[Union[pd.DataFrame, Iterable]] = None,
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
                    if df is not None and not df.empty
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

    def get_items_pd(self, items: PandasDataFrame) -> PandasDataFrame:
        """
        Function to calculate normalized popularities(in fact, probabilities)
        of given items. Returns pandas DataFrame.
        """
        selected_item_popularity = self._get_selected_item_popularity(items)
        selected_item_popularity[self.rating_column] = selected_item_popularity.apply(
            lambda row: 0.1**6 if row["rating_column"] == 0.0 else row["rating_column"], axis=1
        )

        total_rating = selected_item_popularity[self.rating_column].sum()

        selected_item_popularity["probability"] = selected_item_popularity[self.rating_column] / total_rating
        return selected_item_popularity

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
        selected_item_popularity = self._get_selected_item_popularity(items).sort_values(self.item_column, ascending=False)
        save_df(selected_item_popularity, "pandas_base_predict_wrap_selected_item_popularity")
        # old method: sorted_df = selected_item_popularity.sort_values(by=[self.rating_column, self.item_column], ascending=False)
        # old method: selected_item_popularity["rank"] = sorted_df.index + 1
        selected_item_popularity = selected_item_popularity.sort_values(
            by=[self.rating_column, self.item_column],
            ascending=[False, False]
        ).reset_index(drop=True)
        selected_item_popularity["rank"] = range(1, len(selected_item_popularity) + 1)
        save_df(selected_item_popularity, "pandas_base_predict_wrap_selected_item_popularity_rank")
        
        if filter_seen_items and dataset is not None:
            print("ВОШЛИ В ИФ")
            queries = PandasDataFrame(queries)
            query_to_num_items = (
                dataset.interactions.merge(queries, on=self.query_column)
                .groupby(self.query_column, as_index=False)[self.item_column]
                .nunique()
            ).rename(columns={self.item_column: "num_items"})
            save_df(query_to_num_items, "pandas_base_predict_wrap_query_to_num_items")
            queries = queries.merge(query_to_num_items, on=self.query_column, how="left")
            queries = queries.fillna(0)
            save_df(queries, "pandas_base_predict_wrap_queries_v2")
            max_seen = queries["num_items"].max()  # noqa: F841
            selected_item_popularity = selected_item_popularity.query("rank <= @k + @max_seen")
            save_df(selected_item_popularity, "pandas_base_predict_wrap_selected_item_popularity_v2")
            joined = queries.merge(selected_item_popularity, how="cross")
            return joined[joined["rank"] <= (k + joined["num_items"])]#.drop("rank", axis=1)#.drop("num_items", axis=1)
        print("НЕ ВОШЛИ В ИФ")
        joined = queries.merge(selected_item_popularity[selected_item_popularity['rank'] <= k], how="cross")
        save_df(joined, "pandas_base_predict_wrap_joined")
        return joined.drop("rank", axis=1)

    # TODO: РЕАЛИЗОВАТЬ predict_with_sampling в NonPersonolizedPandasImplementation

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
        print(f"pandas: {dataset.interactions.shape[0]=}, {queries.shape[0]=}, {items.shape[0]=}")
        save_df(dataset.interactions, "pandas_base_predict_wrap_dataset")
        save_df(queries,"pandas_base_predict_wrap_queries")
        save_df(items,"pandas_base_predict_wrap_items")
        recs = self._predict_without_sampling(dataset, k, queries, items, filter_seen_items)
        print(f"first {recs.shape[0]=}")
        save_df(recs, "pandas_base_predict_wrap_recs_1")
        if filter_seen_items and dataset is not None:
            recs = self._filter_seen(recs=recs, interactions=dataset.interactions, queries=queries, k=k)
            print(f"second {recs.shape[0]=}")
            save_df(recs, "pandas_base_predict_wrap_recs_2")
        recs = get_top_k(recs, self.query_column, [(self.rating_column, False), (self.item_column, False)], k)[[
                self.query_column, self.item_column, self.rating_column
        ]].reset_index(drop=True)
        print(f"third {recs.shape[0]=}")
        save_df(recs, "pandas_base_predict_wrap_recs_3")
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

    def predict_pairs(self, pairs: PandasDataFrame, dataset: PandasDataFrame = None) -> PandasDataFrame:  # noqa: ARG002
        if self.item_popularity is None:
            msg = "Model not fitted. Please call fit() first."
            raise ValueError(msg)
        preds = pairs.merge(self.item_popularity, on=self.item_column, how="left" if self.add_cold_items else "inner")
        preds[self.rating_column].fillna(self._calc_fill(), inplace=True)
        return preds[[self.query_column, self.item_column, self.rating_column]]

    def predict_proba(
        self, dataset: PandasDataFrame, k: int, queries, items: PandasDataFrame, filter_seen_items=True  # noqa: ARG002
    ) -> PandasDataFrame:
        return NotImplementedError()

    def save_model(self, path: str, additional_params=None):  # noqa: ARG002
        saved_params = {
            "query_column": self.query_column,
            "item_column": self.item_column,
            "rating_column": self.rating_column,
            "timestamp_column": self.timestamp_column,
        }
        if additional_params is not None:
            saved_params.update(additional_params)
        # TODO: save_picklable_to_parquet(saved_params, join(path, "params.dump"))
        return saved_params


def get_different_rows(source_df, new_df): # TODO: remove
    """Returns just the rows from the new dataframe that differ from the source dataframe"""
    merged_df = source_df.merge(new_df, indicator=True, how='outer')
    changed_rows_df = merged_df[merged_df['_merge'] == 'right_only']
    return changed_rows_df.drop('_merge', axis=1)

def save_df(df, filename):
    if isinstance(df, SparkDataFrame):
        df.write.mode("overwrite").parquet(filename)
        return True
    elif isinstance(df, PandasDataFrame):
        df.to_parquet(filename)
        return True
    return False