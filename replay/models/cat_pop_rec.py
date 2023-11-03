from os.path import join
from typing import Iterable, Optional, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.base_rec import IsSavable, RecommenderCommons
from replay.utils.spark_utils import (
    get_top_k,
    get_unique_entities,
    filter_cold,
    return_recs,
    save_picklable_to_parquet,
    load_pickled_from_parquet,
)
from replay.data import Dataset


# pylint: disable=too-many-instance-attributes
class CatPopRec(IsSavable, RecommenderCommons):
    """
    CatPopRec generate recommendation for item categories.
    It recommends the most popular items in the category or all sub-categories in the category.
    It is assumed that each item belongs to one or more categories.
    Categories could have a flat structure or be hierarchical (tree structure).
    If the categories are hierarchical, items should belong to the lowest level categories only.
    In this case, the model could generate recommendations not only for the lowest level categories,
    but for all categories, combining the lowest level categories statistics.
    """

    cat_item_popularity: DataFrame
    leaf_cat_mapping: DataFrame
    can_predict_cold_items: bool = False
    fit_items: DataFrame
    query_column: str
    item_column: str
    rating_column: str

    def _generate_mapping(
        self, cat_tree: DataFrame, max_iter: int = 20
    ) -> DataFrame:
        """
        Create DataFrame with mapping [`category`, `leaf_cat`]
        where `leaf_cat` is the lowest level categories of category tree,
        which contain items, not sub-categories.
        :param cat_tree: spark dataframe with columns [`category`, `parent_cat`].
            Contains category mapping `category - parent category`.
            Each category has only one parent.
            If the parent is absent, `parent_cat` value should be None.
        :param max_iter: maximal number of iteration of descend through the category tree
        :return: DataFrame with mapping [`category`, `leaf_cat`]
        """
        current_res = cat_tree.select(
            sf.col("category"), sf.col("category").alias("leaf_cat")
        )

        i = 0
        res_size_growth = current_res.count()
        while i < max_iter and res_size_growth > 0:
            new_res = (
                current_res.join(
                    cat_tree.withColumnRenamed("category", "new_leaf_cat"),
                    on=sf.col("leaf_cat") == sf.col("parent_cat"),
                    how="left",
                )
                # if `leaf_cat` is already found, the join result will be None,
                # use coalesce to get value
                .select(
                    "category",
                    sf.coalesce("new_leaf_cat", "leaf_cat").alias("leaf_cat"),
                )
            ).cache()
            res_size_growth = new_res.count() - current_res.count()
            current_res.unpersist()
            current_res = new_res
            i += 1

        # RunTime error instead of logger.warning?
        if i == max_iter:
            self.logger.warning(
                "Category tree was not fully processed in %s iterations. "
                "Increase the `max_iter` value or check the category tree structure."
                "It must not have loops and each category should have only one parent.",
                max_iter,
            )

        return current_res

    def set_cat_tree(self, cat_tree: DataFrame):
        """
        Set/update category tree `cat_tree` used to generate recommendations.
        :param cat_tree: park dataframe with columns [`category`, `parent_cat`].
            Contains category mapping `category - parent category`.
            Each category has only one parent.
        """
        self._clear_cache()
        self.leaf_cat_mapping = self._generate_mapping(cat_tree)

    def __init__(
        self,
        cat_tree: Optional[DataFrame] = None,
        max_iter: Optional[int] = 20,
    ):
        """
        :param cat_tree: spark dataframe with columns [`category`, `parent_cat`].
            Contains category mapping `category - parent category`.
            Each category has only one parent.
        :param max_iter: maximal number of iteration of descend through the category tree
        """
        self.max_iter = max_iter
        if cat_tree is not None:
            self.leaf_cat_mapping = self._generate_mapping(
                cat_tree, max_iter=max_iter
            )

    @property
    def _init_args(self):
        return {"max_iter": self.max_iter}

    @property
    def _dataframes(self):
        return {
            "cat_item_popularity": self.cat_item_popularity,
            "leaf_cat_mapping": self.leaf_cat_mapping,
        }

    def fit(self, dataset: Dataset) -> None:
        """
        Fit a recommendation model

        :param dataset: historical interactions with query/item features
            ``[user_id, item_id, category, timestamp, rating]``
            where `category` is an item's category.
            The `item_idx`, `category` are mandatory columns.
            If `rating` column is present it is treated as number
            of interactions with the item and the `rating` values are summed.
        """
        self.logger.debug("Starting fit %s", type(self).__name__)
        self.query_column = dataset.feature_schema.query_id_column
        self.item_column = dataset.feature_schema.item_id_column
        self.rating_column = dataset.feature_schema.interactions_rating_column

        self.fit_items = sf.broadcast(dataset.interactions.select(self.item_column).distinct())
        self._fit(dataset=dataset)

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        if self.rating_column is not None:
            self.cat_item_popularity = dataset.interactions.groupBy("category", self.item_column).agg(
                sf.sum(self.rating_column).alias(self.rating_column)
            )
        else:
            self.rating_column = "relevance"
            self.cat_item_popularity = dataset.interactions.groupBy("category", self.item_column).agg(
                sf.count(self.item_column).alias(self.rating_column)
            )

        self.cat_item_popularity.cache()
        self.cat_item_popularity.count()

    def _clear_cache(self):
        if hasattr(self, "cat_item_popularity"):
            self.cat_item_popularity.unpersist()
        if hasattr(self, "leaf_cat_mapping"):
            self.leaf_cat_mapping.unpersist()

    # pylint: disable=arguments-differ
    def predict(
        self,
        categories: Union[DataFrame, Iterable],
        k: int,
        items: Optional[Union[DataFrame, Iterable]] = None,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Get top-k recommendations for each category in `categories`.

        :param categories:  dataframe containing ``[category]`` to predict for or ``array-like``;
        :param k: number of recommendations for each category
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, consider all items.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe will be returned
        :return: cached recommendation dataframe with columns ``[category, item_idx, rating]``
            or None if `file_path` is provided
        """
        return self._predict_wrap(
            categories=categories,
            k=k,
            items=items,
            recs_file_path=recs_file_path,
        )

    def _predict_wrap(
        self,
        categories: Union[DataFrame, Iterable],
        k: int,
        items: Optional[Union[DataFrame, Iterable]] = None,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Predict wrapper to allow for fewer parameters in models

        :param categories:  dataframe containing ``[category]`` to predict for or ``array-like``;
        :param k: number of recommendations for each user
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items presented in `fit` data.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[category, item_idx, rating]``
            or None if `file_path` is provided
        """
        self.logger.debug("Starting predict %s", type(self).__name__)

        categories = get_unique_entities(categories, "category")
        item_data = items or self.fit_items
        items = get_unique_entities(item_data, self.item_column)

        num_new, items = filter_cold(
            items, self.fit_items, col_name=self.item_column
        )
        if num_new > 0:
            self.logger.info(
                "%s model can't predict cold items, they will be ignored",
                self,
            )

        num_items = items.count()
        if num_items < k:
            message = f"k = {k} > number of items = {num_items}"
            self.logger.debug(message)

        recs = self._predict(
            categories=categories,
            items=items,
        )

        recs = get_top_k(
            recs,
            k=k,
            partition_by_col=sf.col("category"),
            order_by_col=[
                sf.col(self.rating_column).desc(),
                sf.col(self.item_column).desc(),
            ],
        ).select("category", self.item_column, self.rating_column)

        return return_recs(recs, recs_file_path)

    def _predict(
        self,
        categories: Union[DataFrame, Iterable],
        items: Optional[Union[DataFrame, Iterable]] = None,
    ) -> DataFrame:
        res = categories.join(self.leaf_cat_mapping, on="category")
        # filter required categories and items of `self.cat_item_popularity`
        unique_leaf_cats = res.select("leaf_cat").distinct()
        unique_leaf_cat_items = (
            self.cat_item_popularity.withColumnRenamed("category", "leaf_cat")
            .join(sf.broadcast(unique_leaf_cats), on="leaf_cat")
            .join(sf.broadcast(items), on=self.item_column)
        )

        # find number of interactions in all leaf categories after filtering
        num_interactions_in_cat = (
            res.join(
                unique_leaf_cat_items.groupBy("leaf_cat").agg(
                    sf.sum(self.rating_column).alias("sum_rating")
                ),
                on="leaf_cat",
            )
            .groupBy("category")
            .agg(sf.sum("sum_rating").alias("sum_rating"))
        )

        # aggregate results for each category: sum up num interactions in leaf categories
        # and calculate popularity as a number of interactions with an item in all leaf categories
        # divided by the number of interactions in all leaf categories of a category
        return (
            unique_leaf_cat_items.join(res, on="leaf_cat")
            .groupBy("category", self.item_column)
            .agg(sf.sum(self.rating_column).alias(self.rating_column))
            .join(num_interactions_in_cat, on="category")
            .withColumn(
                self.rating_column, sf.col(self.rating_column) / sf.col("sum_rating")
            )
        )

    def _save_model(self, path: str):
        save_picklable_to_parquet(
            {
                "query_column": self.query_column,
                "item_column": self.item_column,
                "rating_column": self.rating_column,
            },
            join(path, "params.dump")
        )

    def _load_model(self, path: str):
        loaded_params = load_pickled_from_parquet(join(path, "params.dump"))
        self.query_column = loaded_params.get("query_column")
        self.item_column = loaded_params.get("item_column")
        self.rating_column = loaded_params.get("rating_column")
