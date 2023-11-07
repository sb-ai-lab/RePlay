from os.path import join
from typing import Optional, Dict, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.window import Window

from replay.models.extensions.ann.index_builders.base_index_builder import IndexBuilder
from replay.models.base_neighbour_rec import NeighbourRec
from replay.optimization.optuna_objective import ItemKNNObjective

from replay.data import Dataset
from replay.utils.spark_utils import save_picklable_to_parquet, load_pickled_from_parquet


# pylint: disable=too-many-ancestors
# pylint: disable=too-many-instance-attributes
class ItemKNN(NeighbourRec):
    """Item-based ItemKNN with modified cosine similarity measure."""

    def _get_ann_infer_params(self) -> Dict[str, Any]:
        return {
            "features_col": None,
        }

    all_items: Optional[DataFrame]
    dot_products: Optional[DataFrame]
    item_norms: Optional[DataFrame]
    bm25_k1 = 1.2
    bm25_b = 0.75
    _objective = ItemKNNObjective
    _search_space = {
        "num_neighbours": {"type": "int", "args": [1, 100]},
        "shrink": {"type": "int", "args": [0, 100]},
        "weighting": {"type": "categorical", "args": [None, "tf_idf", "bm25"]}
    }

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_neighbours: int = 10,
        use_rating: bool = False,
        shrink: float = 0.0,
        weighting: str = None,
        index_builder: Optional[IndexBuilder] = None,
    ):
        """
        :param num_neighbours: number of neighbours
        :param use_rating: flag to use rating values as is or to treat them as 1
        :param shrink: term added to the denominator when calculating similarity
        :param weighting: item reweighting type, one of [None, 'tf_idf', 'bm25']
        :param index_builder: `IndexBuilder` instance that adds ANN functionality.
            If not set, then ann will not be used.
        """
        self.shrink = shrink
        self.use_rating = use_rating
        self.num_neighbours = num_neighbours

        valid_weightings = self._search_space["weighting"]["args"]
        if weighting not in valid_weightings:
            raise ValueError(f"weighting must be one of {valid_weightings}")
        self.weighting = weighting
        if isinstance(index_builder, (IndexBuilder, type(None))):
            self.index_builder = index_builder
        elif isinstance(index_builder, dict):
            self.init_builder_from_dict(index_builder)

    @property
    def _init_args(self):
        return {
            "shrink": self.shrink,
            "use_rating": self.use_rating,
            "num_neighbours": self.num_neighbours,
            "weighting": self.weighting,
            "index_builder": self.index_builder.init_meta_as_dict() if self.index_builder else None,
        }

    def _save_model(self, path: str):
        save_picklable_to_parquet(
            {
                "query_column": self.query_column,
                "item_column": self.item_column,
                "rating_column": self.rating_column,
                "timestamp_column": self.timestamp_column,
            },
            join(path, "params.dump")
        )
        if self._use_ann:
            self._save_index(path)

    def _load_model(self, path: str):
        loaded_params = load_pickled_from_parquet(join(path, "params.dump"))
        self.query_column = loaded_params.get("query_column")
        self.item_column = loaded_params.get("item_column")
        self.rating_column = loaded_params.get("rating_column")
        self.timestamp_column = loaded_params.get("timestamp_column")
        if self._use_ann:
            self._load_index(path)

    @staticmethod
    def _shrink(dot_products: DataFrame, shrink: float) -> DataFrame:
        return dot_products.withColumn(
            "similarity",
            sf.col("dot_product")
            / (sf.col("norm1") * sf.col("norm2") + shrink),
        ).select("item_idx_one", "item_idx_two", "similarity")

    def _get_similarity(self, interactions: DataFrame) -> DataFrame:
        """
        Calculate item similarities

        :param interactions: DataFrame with interactions, `[user_id, item_id, rating]`
        :return: similarity matrix `[item_idx_one, item_idx_two, similarity]`
        """
        dot_products = self._get_products(interactions)
        similarity = self._shrink(dot_products, self.shrink)
        return similarity

    def _reweight_interactions(self, interactions: DataFrame):
        """
        Reweight rating according to TD-IDF or BM25 weighting.

        :param interactions: DataFrame with interactions, `[user_id, item_id, rating]`
        :return: interactions `[user_id, item_id, rating]`
        """
        if self.weighting == "bm25":
            interactions = self._get_tf_bm25(interactions)

        idf = self._get_idf(interactions)

        interactions = interactions.join(idf, how="inner", on=self.query_column).withColumn(
            self.rating_column,
            sf.col(self.rating_column) * sf.col("idf"),
        )

        return interactions

    def _get_tf_bm25(self, interactions: DataFrame):
        """
        Adjust rating by BM25 term frequency.

        :param interactions: DataFrame with interactions, `[user_id, item_id, rating]`
        :return: interactions `[user_id, item_id, rating]`
        """
        item_stats = interactions.groupBy(self.item_column).agg(
            sf.count(self.query_column).alias("n_queries_per_item")
        )
        avgdl = item_stats.select(sf.mean("n_queries_per_item")).take(1)[0][0]
        interactions = interactions.join(item_stats, how="inner", on=self.item_column)

        interactions = (
            interactions.withColumn(
                self.rating_column,
                sf.col(self.rating_column) * (self.bm25_k1 + 1) / (
                    sf.col(self.rating_column) + self.bm25_k1 * (
                        1 - self.bm25_b + self.bm25_b * (
                            sf.col("n_queries_per_item") / avgdl
                        )
                    )
                )
            )
            .drop("n_queries_per_item")
        )

        return interactions

    def _get_idf(self, interactions: DataFrame):
        """
        Return inverse document score for interactions reweighting.

        :param interactions: DataFrame with interactions, `[user_id, item_id, rating]`
        :return: idf `[idf]`
        :raises: ValueError if self.weighting not in ["tf_idf", "bm25"]
        """
        df = interactions.groupBy(self.query_column).agg(sf.count(self.item_column).alias("DF"))
        n_items = interactions.select(self.item_column).distinct().count()

        if self.weighting == "tf_idf":
            idf = (
                df.withColumn("idf", sf.log1p(sf.lit(n_items) / sf.col("DF")))
                .drop("DF")
            )
        elif self.weighting == "bm25":
            idf = (
                df.withColumn(
                    "idf",
                    sf.log1p(
                        (sf.lit(n_items) - sf.col("DF") + 0.5)
                        / (sf.col("DF") + 0.5)
                    ),
                )
                .drop("DF")
            )
        else:
            raise ValueError("weighting must be one of ['tf_idf', 'bm25']")

        return idf

    def _get_products(self, interactions: DataFrame) -> DataFrame:
        """
        Calculate item dot products

        :param interactions: DataFrame with interactions, `[user_id, item_id, rating]`
        :return: similarity matrix `[item_idx_one, item_idx_two, norm1, norm2]`
        """
        if self.weighting:
            interactions = self._reweight_interactions(interactions)

        left = interactions.withColumnRenamed(
            self.item_column, "item_idx_one"
        ).withColumnRenamed(self.rating_column, "rel_one")
        right = interactions.withColumnRenamed(
            self.item_column, "item_idx_two"
        ).withColumnRenamed(self.rating_column, "rel_two")

        dot_products = (
            left.join(right, how="inner", on=self.query_column)
            .filter(sf.col("item_idx_one") != sf.col("item_idx_two"))
            .withColumn(self.rating_column, sf.col("rel_one") * sf.col("rel_two"))
            .groupBy("item_idx_one", "item_idx_two")
            .agg(sf.sum(self.rating_column).alias("dot_product"))
        )

        item_norms = (
            interactions.withColumn(self.rating_column, sf.col(self.rating_column) ** 2)
            .groupBy(self.item_column)
            .agg(sf.sum(self.rating_column).alias("square_norm"))
            .select(sf.col(self.item_column), sf.sqrt("square_norm").alias("norm"))
        )
        norm1 = item_norms.withColumnRenamed(
            self.item_column, "item_id1"
        ).withColumnRenamed("norm", "norm1")
        norm2 = item_norms.withColumnRenamed(
            self.item_column, "item_id2"
        ).withColumnRenamed("norm", "norm2")

        dot_products = dot_products.join(
            norm1, how="inner", on=sf.col("item_id1") == sf.col("item_idx_one")
        )
        dot_products = dot_products.join(
            norm2, how="inner", on=sf.col("item_id2") == sf.col("item_idx_two")
        )

        return dot_products

    def _get_k_most_similar(self, similarity_matrix: DataFrame) -> DataFrame:
        """
        Leaves only top-k neighbours for each item

        :param similarity_matrix: dataframe `[item_idx_one, item_idx_two, similarity]`
        :return: cropped similarity matrix
        """
        return (
            similarity_matrix.withColumn(
                "similarity_order",
                sf.row_number().over(
                    Window.partitionBy("item_idx_one").orderBy(
                        sf.col("similarity").desc(),
                        sf.col("item_idx_two").desc(),
                    )
                ),
            )
            .filter(sf.col("similarity_order") <= self.num_neighbours)
            .drop("similarity_order")
        )

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        df = dataset.interactions.select(self.query_column, self.item_column, self.rating_column)
        if not self.use_rating:
            df = df.withColumn(self.rating_column, sf.lit(1))

        similarity_matrix = self._get_similarity(df)
        self.similarity = self._get_k_most_similar(similarity_matrix)
        self.similarity.cache().count()
