# pylint: disable=too-many-lines
import functools
import logging
import os
import pickle
from collections.abc import Iterable
from typing import Dict, Optional, Tuple, List, Union, Any

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame

from replay.constants import AnyDataFrame
from replay.data_preparator import ToNumericFeatureTransformer
from replay.history_based_fp import HistoryBasedFeaturesProcessor
from replay.metrics import Metric, Precision
from replay.models import ALSWrap, RandomRec, PopRec
from replay.models.base_rec import BaseRecommender, HybridRecommender
from replay.scenarios.two_stages.reranker import LamaWrap
from replay.scenarios.two_stages.slama_reranker import SlamaWrap
from replay.session_handler import State
from replay.splitters import Splitter, UserSplitter
from replay.utils import (
    array_mult,
    cache_if_exists,
    get_log_info,
    get_top_k_recs,
    join_or_return,
    join_with_col_renaming,
    unpersist_if_exists, create_folder, save_transformer, do_path_exists, load_transformer, list_folder, JobGroup,
    cache_and_materialize_if_in_debug, JobGroupWithMetrics,
)

logger = logging.getLogger("replay")


# pylint: disable=too-many-locals, too-many-arguments
def get_first_level_model_features(
    model: BaseRecommender,
    pairs: DataFrame,
    user_features: Optional[DataFrame] = None,
    item_features: Optional[DataFrame] = None,
    add_factors_mult: bool = True,
    prefix: str = "",
) -> DataFrame:
    """
    Get user and item embeddings from replay model.
    Can also compute elementwise multiplication between them with ``add_factors_mult`` parameter.
    Zero vectors are returned if a model does not have embeddings for specific users/items.

    :param model: trained model
    :param pairs: user-item pairs to get vectors for `[user_id/user_idx, item_id/item_id]`
    :param user_features: user features `[user_id/user_idx, feature_1, ....]`
    :param item_features: item features `[item_id/item_idx, feature_1, ....]`
    :param add_factors_mult: flag to add elementwise multiplication
    :param prefix: name to add to the columns
    :return: DataFrame
    """
    users = pairs.select("user_idx").distinct()
    items = pairs.select("item_idx").distinct()
    user_factors, user_vector_len = model._get_features_wrap(
        users, user_features
    )
    item_factors, item_vector_len = model._get_features_wrap(
        items, item_features
    )

    pairs_with_features = join_or_return(
        pairs, user_factors, how="left", on="user_idx"
    )
    pairs_with_features = join_or_return(
        pairs_with_features,
        item_factors,
        how="left",
        on="item_idx",
    )

    if user_factors is not None:
        pairs_with_features = pairs_with_features.withColumn(
            "user_factors",
            sf.coalesce(
                sf.col("user_factors"),
                sf.array([sf.lit(0.0)] * user_vector_len),
            ),
        )

    if item_factors is not None:
        pairs_with_features = pairs_with_features.withColumn(
            "item_factors",
            sf.coalesce(
                sf.col("item_factors"),
                sf.array([sf.lit(0.0)] * item_vector_len),
            ),
        )

    if model.__str__() == "LightFMWrap":
        pairs_with_features = (
            pairs_with_features.fillna({"user_bias": 0, "item_bias": 0})
            .withColumnRenamed("user_bias", f"{prefix}_user_bias")
            .withColumnRenamed("item_bias", f"{prefix}_item_bias")
        )

    if (
        add_factors_mult
        and user_factors is not None
        and item_factors is not None
    ):
        pairs_with_features = pairs_with_features.withColumn(
            "factors_mult",
            array_mult(sf.col("item_factors"), sf.col("user_factors")),
        )

    return pairs_with_features


# pylint: disable=too-many-instance-attributes
class TwoStagesScenario(HybridRecommender):
    """
    *train*:

    1) take input ``log`` and split it into first_level_train and second_level_train
       default splitter splits each user's data 50/50
    2) train ``first_stage_models`` on ``first_stage_train``
    3) create negative examples to train second stage model using one of:

       - wrong recommendations from first stage
       - random examples

        use ``num_negatives`` to specify number of negatives per user
    4) augments dataset with features:

       - get 1 level recommendations for positive examples
         from second_level_train and for generated negative examples
       - add user and item features
       - generate statistical and pair features

    5) train ``TabularAutoML`` from LightAutoML

    *inference*:

    1) take ``log``
    2) generate candidates, their number can be specified with ``num_candidates``
    3) add features as in train
    4) get recommendations

    """

    can_predict_cold_users: bool = True
    can_predict_cold_items: bool = True

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        train_splitter: Splitter = UserSplitter(
            item_test_size=0.5, shuffle=True, seed=42
        ),
        first_level_models: Union[
            List[BaseRecommender], BaseRecommender
        ] = ALSWrap(rank=128),
        fallback_model: Optional[BaseRecommender] = PopRec(),
        use_first_level_models_feat: Union[List[bool], bool] = False,
        second_model_type: str = "lama",
        second_model_params: Optional[Union[Dict, str]] = None,
        second_model_config_path: Optional[str] = None,
        num_negatives: int = 100,
        negatives_type: str = "first_level",
        use_generated_features: bool = False,
        user_cat_features_list: Optional[List] = None,
        item_cat_features_list: Optional[List] = None,
        custom_features_processor: HistoryBasedFeaturesProcessor = None,
        seed: int = 123
    ) -> None:
        """
        :param train_splitter: splitter to get ``first_level_train`` and ``second_level_train``.
            Default is random 50% split.
        :param first_level_models: model or a list of models
        :param fallback_model: model used to fill missing recommendations at first level models
        :param use_first_level_models_feat: flag or a list of flags to use
            features created by first level models
        :param second_model_params: TabularAutoML parameters
        :param second_model_config_path: path to config file for TabularAutoML
        :param num_negatives: number of negative examples used during train
        :param negatives_type: negative examples creation strategy,``random``
            or most relevant examples from ``first-level``
        :param use_generated_features: flag to use generated features to train second level
        :param user_cat_features_list: list of user categorical features
        :param item_cat_features_list: list of item categorical features
        :param custom_features_processor: you can pass custom feature processor
        :param seed: random seed

        """
        self.train_splitter = train_splitter
        self.cached_list = []

        self.first_level_models = (
            first_level_models
            if isinstance(first_level_models, Iterable)
            else [first_level_models]
        )

        self.first_level_item_len = 0
        self.first_level_user_len = 0

        self.random_model = RandomRec(seed=seed)
        self.fallback_model = fallback_model
        self.first_level_user_features_transformer = (
            ToNumericFeatureTransformer()
        )
        self.first_level_item_features_transformer = (
            ToNumericFeatureTransformer()
        )

        if isinstance(use_first_level_models_feat, bool):
            self.use_first_level_models_feat = [
                use_first_level_models_feat
            ] * len(self.first_level_models)
        else:
            if len(self.first_level_models) != len(
                use_first_level_models_feat
            ):
                raise ValueError(
                    f"For each model from first_level_models specify "
                    f"flag to use first level features."
                    f"Length of first_level_models is {len(first_level_models)}, "
                    f"Length of use_first_level_models_feat is {len(use_first_level_models_feat)}"
                )

            self.use_first_level_models_feat = use_first_level_models_feat

        if second_model_type == "lama":
            self.second_stage_model = LamaWrap(
                params=second_model_params, config_path=second_model_config_path
            )
        elif second_model_type == "slama":
            self.second_stage_model = SlamaWrap(
                params=second_model_params, config_path=second_model_config_path
            )
        else:
            raise Exception(f"Unsupported second model type: {second_model_type}")

        self.num_negatives = num_negatives
        if negatives_type not in ["random", "first_level"]:
            raise ValueError(
                f"Invalid negatives_type value: {negatives_type}. Use 'random' or 'first_level'"
            )
        self.negatives_type = negatives_type

        self.use_generated_features = use_generated_features
        self.features_processor = (
            custom_features_processor
            if custom_features_processor
            else HistoryBasedFeaturesProcessor(
                user_cat_features_list=user_cat_features_list,
                item_cat_features_list=item_cat_features_list,
            )
        )
        self.seed = seed

        self._job_group_id = ""

    # TO DO: add save/load for scenarios
    @property
    def _init_args(self):
        return {}

    def _save_model(self, path: str):
        from replay.model_handler import save
        spark = State().session
        create_folder(path, exists_ok=True)

        # save features
        if self.first_level_user_features_transformer is not None:
            save_transformer(
                self.first_level_user_features_transformer,
                os.path.join(path, "first_level_user_features_transformer")
            )

        if self.first_level_item_features_transformer is not None:
            save_transformer(
                self.first_level_item_features_transformer,
                os.path.join(path, "first_level_item_features_transformer")
            )

        if self.features_processor is not None:
            save_transformer(self.features_processor, os.path.join(path, "features_processor"))

        # Save first level models
        first_level_models_path = os.path.join(path, "first_level_models")
        create_folder(first_level_models_path)
        for i, model in enumerate(self.first_level_models):
            save(model, os.path.join(first_level_models_path, f"model_{i}"))

        # save auxillary models
        if self.random_model is not None:
            save(self.random_model, os.path.join(path, "random_model"))

        if self.fallback_model is not None:
            save(self.fallback_model, os.path.join(path, "fallback_model"))

        # save second stage model
        if self.second_stage_model is not None:
            save_transformer(self.second_stage_model, os.path.join(path, "second_stage_model"))

        # save general data and settings
        data = {
            "train_splitter": pickle.dumps(self.train_splitter),
            "first_level_item_len": self.first_level_item_len,
            "first_level_user_len": self.first_level_user_len,
            "use_first_level_models_feat": self.use_first_level_models_feat,
            "num_negatives": self.num_negatives,
            "negatives_type": self.negatives_type,
            "use_generated_features": self.use_generated_features,
            "seed": self.seed
        }

        spark.createDataFrame([data]).write.parquet(os.path.join(path, "data.parquet"))

    def _load_model(self, path: str):
        from replay.model_handler import load
        spark = State().session

        # load general data and settings
        data = spark.read.parquet(os.path.join(path, "data.parquet")).first().asDict()

        # load transformers for features
        comp_path = os.path.join(path, "first_level_user_features_transformer")
        first_level_user_features_transformer = load_transformer(comp_path) if do_path_exists(comp_path) else None #TODO: check why this dir exists if user_features=None

        comp_path = os.path.join(path, "first_level_item_features_transformer")
        first_level_item_features_transformer = load_transformer(comp_path) if do_path_exists(comp_path) else None #TODO same

        comp_path = os.path.join(path, "features_processor")
        features_processor = load_transformer(comp_path) if do_path_exists(comp_path) else None # TODO same

        # load first level models
        first_level_models_path = os.path.join(path, "first_level_models")
        if do_path_exists(first_level_models_path):
            model_paths = [
                os.path.join(first_level_models_path, model_path)
                for model_path in list_folder(first_level_models_path)
            ]
            first_level_models = [load(model_path) for model_path in model_paths]
        else:
            first_level_models = None

        # load auxillary models
        comp_path = os.path.join(path, "random_model")
        random_model = load(comp_path) if do_path_exists(comp_path) else None

        comp_path = os.path.join(path, "fallback_model")
        fallback_model = load(comp_path) if do_path_exists(comp_path) else None

        # load second stage model
        comp_path = os.path.join(path, "second_stage_model")
        # second_stage_model = load_transformer(comp_path) if do_path_exists(comp_path) else None # TODO: fix it
        second_stage_model = None

        self.__dict__.update({
            **data,
            "first_level_user_features_transformer": first_level_user_features_transformer,
            "first_level_item_features_transformer": first_level_item_features_transformer,
            "features_processor": features_processor,
            "first_level_models": first_level_models,
            "random_model": random_model,
            "fallback_model": fallback_model,
            "second_stage_model": second_stage_model
        })

    # pylint: disable=too-many-locals
    def _add_features_for_second_level(
        self,
        log_to_add_features: DataFrame,
        log_for_first_level_models: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
    ) -> DataFrame:
        """
        Added features are:
            - relevance from first level models
            - user and item features from first level models
            - dataset features
            - FeatureProcessor features

        :param log_to_add_features: input DataFrame``[user_idx, item_idx, timestamp, relevance]``
        :param log_for_first_level_models: DataFrame``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features``[user_idx]`` + feature columns
        :param item_features: item features``[item_idx]`` + feature columns
        :return: DataFrame
        """
        self.logger.info("Generating features")
        full_second_level_train = log_to_add_features
        first_level_item_features_cached = cache_if_exists(
            self.first_level_item_features_transformer.transform(item_features)
        )
        first_level_user_features_cached = cache_if_exists(
            self.first_level_user_features_transformer.transform(user_features)
        )

        for idx, model in enumerate(self.first_level_models):
            if self.use_first_level_models_feat[idx]:
                features = get_first_level_model_features(
                    model=model,
                    pairs=full_second_level_train.select("user_idx", "item_idx"),
                    user_features=first_level_user_features_cached,
                    item_features=first_level_item_features_cached,
                    prefix=f"m_{idx}",
                )

                with JobGroup(self._job_group_id, f"features_caching_{type(self).__name__}") as job_desc:
                    cache_and_materialize_if_in_debug(features, job_desc)

                full_second_level_train = join_with_col_renaming(
                    left=full_second_level_train,
                    right=features,
                    on_col_name=["user_idx", "item_idx"],
                    how="left",
                )

        unpersist_if_exists(first_level_user_features_cached)
        unpersist_if_exists(first_level_item_features_cached)

        full_second_level_train_cached = full_second_level_train.fillna(0)

        self.logger.info("Adding features from the dataset")
        full_second_level_train = join_or_return(
            full_second_level_train_cached,
            user_features,
            on="user_idx",
            how="left",
        )
        full_second_level_train = join_or_return(
            full_second_level_train,
            item_features,
            on="item_idx",
            how="left",
        )

        if self.use_generated_features:
            with JobGroupWithMetrics(self._job_group_id, "fitting_the_feature_processor"):
                if not self.features_processor.fitted:
                    # PERF - preventing potential losing time on repeated expensive computations
                    full_second_level_train = full_second_level_train.cache()
                    self.cached_list.append(full_second_level_train)

                    self.features_processor.fit(
                        log=log_for_first_level_models,
                        user_features=user_features,
                        item_features=item_features,
                    )

                self.logger.info("Adding generated features")
                full_second_level_train = self.features_processor.transform(
                    log=full_second_level_train
                )

        self.logger.info(
            "Columns at second level: %s",
            " ".join(full_second_level_train.columns),
        )

        return full_second_level_train

    def _split_data(self, log: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Write statistics"""
        first_level_train, second_level_train = self.train_splitter.split(log)
        logger.info("Log info: %s", get_log_info(log))
        logger.info(
            "first_level_train info: %s", get_log_info(first_level_train)
        )
        logger.info(
            "second_level_train info: %s", get_log_info(second_level_train)
        )
        return first_level_train, second_level_train

    @staticmethod
    def _filter_or_return(dataframe, condition):
        if dataframe is None:
            return dataframe
        return dataframe.filter(condition)

    def _predict_with_first_level_model(
        self,
        model: BaseRecommender,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
        log_to_filter: DataFrame,
        prediction_label: str = ''
    ) -> DataFrame:
        """
        Filter users and items using can_predict_cold_items and can_predict_cold_users, and predict
        """
        if not model.can_predict_cold_items:
            log, items, item_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("item_idx") < self.first_level_item_len,
                )
                for df in [log, items, item_features]
            ]
        if not model.can_predict_cold_users:
            log, users, user_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("user_idx") < self.first_level_user_len,
                )
                for df in [log, users, user_features]
            ]

        log_to_filter_cached = join_with_col_renaming(
            left=log_to_filter,
            right=users,
            on_col_name="user_idx",
        ).cache()
        max_positives_to_filter = 0

        with JobGroupWithMetrics(self._job_group_id, "calculating_max_positives_to_filter"):
            if log_to_filter_cached.count() > 0:
                max_positives_to_filter = (
                    log_to_filter_cached.groupBy("user_idx")
                    .agg(sf.count("item_idx").alias("num_positives"))
                    .select(sf.max("num_positives"))
                    .collect()[0][0]
                )

        with JobGroupWithMetrics(__file__, f"{type(model).__name__}._{prediction_label}_predict") as job_desc:
            pred = model._inner_predict_wrap(
                log,
                k=k + max_positives_to_filter,
                users=users,
                items=items,
                user_features=user_features,
                item_features=item_features,
                filter_seen_items=False,
            )
            cache_and_materialize_if_in_debug(pred, job_desc)

        logger.info(f"{type(model).__name__} prediction: {pred}")
        logger.info(f"Length of {type(model).__name__} prediction: {pred.count()}")

        pred = pred.join(
            log_to_filter_cached.select("user_idx", "item_idx"),
            on=["user_idx", "item_idx"],
            how="anti",
        ).drop("user", "item")

        # PERF - preventing potential losing time on repeated expensive computations
        pred = pred.cache()
        self.cached_list.extend([pred, log_to_filter_cached])

        return get_top_k_recs(pred, k)

    def _predict_pairs_with_first_level_model(
        self,
        model: BaseRecommender,
        log: DataFrame,
        pairs: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
    ):
        """
        Get relevance for selected user-item pairs.
        """
        if not model.can_predict_cold_items:
            log, pairs, item_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("item_idx") < self.first_level_item_len,
                )
                for df in [log, pairs, item_features]
            ]
        if not model.can_predict_cold_users:
            log, pairs, user_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("user_idx") < self.first_level_user_len,
                )
                for df in [log, pairs, user_features]
            ]

        return model._predict_pairs(
            pairs=pairs,
            log=log,
            user_features=user_features,
            item_features=item_features,
        )

    # pylint: disable=unused-argument
    def _get_first_level_candidates(
        self,
        model: BaseRecommender,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
        log_to_filter: DataFrame,
        prediction_label: str = ''
    ) -> DataFrame:
        """
        Combining the base model predictions with the fallback model
        predictions.
        """
        passed_arguments = locals()
        passed_arguments.pop("self")

        with JobGroupWithMetrics(self._job_group_id, f"{type(self).__name__}._predict_with_first_level_model"):
            candidates = self._predict_with_first_level_model(**passed_arguments)

            candidates = candidates.cache()
            candidates.write.mode("overwrite").format("noop").save()
            self.cached_list.append(candidates)

        # TODO: temporary commenting
        # if self.fallback_model is not None:
        #     passed_arguments.pop("model")
        #     with JobGroup("fit", "fallback candidates"):
        #         fallback_candidates = self._predict_with_first_level_model(
        #             model=self.fallback_model, **passed_arguments
        #         )
        #         fallback_candidates = fallback_candidates.cache()
        #
        #     with JobGroup("fit", "fallback"):
        #         # TODO: PERF - no cache and repeated computations for candidate and fallback_candidates?
        #         candidates = fallback(
        #             base=candidates,
        #             fill=fallback_candidates,
        #             k=self.num_negatives,
        #         )
        return candidates

    def _combine(
            self,
            log: DataFrame,
            k: int,
            users: DataFrame,
            items: DataFrame,
            user_features: DataFrame,
            item_features: DataFrame,
            log_to_filter: DataFrame,
            mode: str,
            model_names: List[str] = None,
            prediction_label: str = ''
    ) -> DataFrame:

        partial_dfs = []

        for idx, model in enumerate(self.first_level_models):
            with JobGroupWithMetrics(self._job_group_id, f"{type(model).__name__}._predict_with_first_level_model"):
                candidates = self._predict_with_first_level_model(
                    model=model,
                    log=log,
                    k=k,
                    users=users,
                    items=items,
                    user_features=user_features,
                    item_features=item_features,
                    log_to_filter=log_to_filter,
                    prediction_label=prediction_label
                ).withColumnRenamed("relevance", f"rel_{idx}_{model}")

                # we need this caching if mpairs will be counted later in this method
                candidates = candidates.cache()
                candidates.write.mode("overwrite").format("noop").save()
                self.cached_list.append(candidates)

                partial_dfs.append(candidates)

        if mode == 'union':
            required_pairs = (
                functools.reduce(
                    lambda acc, x: acc.unionByName(x),
                    (df.select('user_idx', 'item_idx') for df in partial_dfs)
                ).distinct()
            )
        else:
            # "leading_<model_name>"
            leading_model_name = mode.split('_')[-1]
            required_pairs = (
                partial_dfs[model_names.index(leading_model_name)]
                .select('user_idx', 'item_idx')
                .distinct()
            )

        logger.info("Selecting missing pairs")

        missing_pairs = [
            required_pairs.join(df, on=['user_idx', 'item_idx'], how='anti').select('user_idx', 'item_idx').distinct()
            for df in partial_dfs
        ]

        def get_rel_col(df: DataFrame) -> str:
            logger.info(f"columns: {str(df.columns)}")
            rel_col = [c for c in df.columns if c.startswith('rel_')][0]
            return rel_col

        def make_missing_predictions(model: BaseRecommender, mpairs: DataFrame, partial_df: DataFrame) -> DataFrame:

            mpairs = mpairs.cache()

            if mpairs.count() == 0:
                mpairs.unpersist()
                return partial_df

            current_pred = model._predict_pairs(
                mpairs,
                log=log,
                user_features=user_features,
                item_features=item_features
            ).withColumnRenamed('relevance', get_rel_col(partial_df))

            mpairs.unpersist()
            return partial_df.unionByName(current_pred.select(*partial_df.columns))

        logger.info("Making missing predictions")
        extended_train_dfs = [
            make_missing_predictions(model, mpairs, partial_df)
            for model, mpairs, partial_df in zip(self.first_level_models, missing_pairs, partial_dfs)
        ]

        # we apply left here because some algorithms like itemknn cannot predict beyond their inbuilt top
        combined_df = functools.reduce(
            lambda acc, x: acc.join(x, on=['user_idx', 'item_idx'], how='left'),
            extended_train_dfs
        )

        logger.info("Combination completed.")

        return combined_df

    # pylint: disable=too-many-locals,too-many-statements
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self._job_group_id = "2stage_fit"

        self.cached_list = []

        # 1. Split train data between first and second levels
        self.logger.info("Data splitting")
        first_level_train, second_level_positive = self._split_data(log)

        self.first_level_item_len = (
            first_level_train.select("item_idx").distinct().count()
        )
        self.first_level_user_len = (
            first_level_train.select("user_idx").distinct().count()
        )

        log.cache()
        first_level_train.cache()
        second_level_positive.cache()
        self.cached_list.extend(
            [log, first_level_train, second_level_positive]
        )

        # 2. Transform user and item features if applicable
        if user_features is not None:
            user_features.cache()
            self.cached_list.append(user_features)

        if item_features is not None:
            item_features.cache()
            self.cached_list.append(item_features)

        with JobGroupWithMetrics(self._job_group_id, "item_features_transformer"):
            if not self.first_level_item_features_transformer.fitted:
                self.first_level_item_features_transformer.fit(item_features)

        with JobGroupWithMetrics(self._job_group_id, "user_features_transformer"):
            if not self.first_level_user_features_transformer.fitted:
                self.first_level_user_features_transformer.fit(user_features)

        first_level_item_features = cache_if_exists(
            self.first_level_item_features_transformer.transform(item_features)
        )
        first_level_user_features = cache_if_exists(
            self.first_level_user_features_transformer.transform(user_features)
        )

        first_level_user_features = first_level_user_features.filter(sf.col("user_idx") < self.first_level_user_len) \
            if first_level_user_features is not None else None

        first_level_item_features = first_level_item_features.filter(sf.col("item_idx") < self.first_level_item_len) \
            if first_level_item_features is not None else None

        # 3. Fit first level models
        logger.info(f"first_level_train: {str(first_level_train.columns)}")

        for base_model in [*self.first_level_models, self.random_model, self.fallback_model]:
            with JobGroupWithMetrics(self._job_group_id, f"{type(base_model).__name__}._fit_wrap"):
                base_model._fit_wrap(
                    log=first_level_train,
                    user_features=first_level_user_features,
                    item_features=first_level_item_features,
                )

        # 4. Generate negative examples
        # by making predictions with first level models and combining them into final recommendation lists
        self.logger.info("Generate negative examples")
        negatives_source = (
            self.first_level_models[0]
            if self.negatives_type == "first_level"
            else self.random_model
        )

        with JobGroupWithMetrics(self._job_group_id, f"{type(self).__name__}._combine"):
            first_level_candidates = self._combine(
                    log=first_level_train,
                    k=self.num_negatives,
                    users=log.select("user_idx").distinct(),
                    items=log.select("item_idx").distinct(),
                    user_features=first_level_user_features,
                    item_features=first_level_item_features,
                    log_to_filter=first_level_train,
                    mode="union",
                    prediction_label='1'
            )

            # may be skipped due to join caching in the end
            first_level_candidates = first_level_candidates.cache()
            first_level_candidates.write.mode('overwrite').format('noop').save()
            self.cached_list.append(first_level_candidates)

        logger.info(f"first_level_candidates.columns: {str(first_level_candidates.columns)}")

        unpersist_if_exists(first_level_user_features)
        unpersist_if_exists(first_level_item_features)

        # 5. Create user/ item pairs for the train dataset of the second level (no features except relevance)
        self.logger.info("Creating train dataset for second level")

        second_level_train = (
            first_level_candidates.join(
                second_level_positive.select(
                    "user_idx", "item_idx"
                ).withColumn("target", sf.lit(1)),
                on=["user_idx", "item_idx"],
                how="left",
            ).fillna(0, subset="target")
        ).cache()

        self.cached_list.append(second_level_train)

        # Apply negative sampling to balance postive / negative combination in the resulting train dataset
        neg = second_level_train.filter(second_level_train.target == 0)
        pos = second_level_train.filter(second_level_train.target == 1)
        neg_new = neg.sample(fraction=10 * pos.count() / neg.count())
        second_level_train = pos.union(neg_new)

        with JobGroupWithMetrics(self._job_group_id, "inferring_class_distribution"):
            self.logger.info(
                "Distribution of classes in second-level train dataset:\n %s",
                (
                    second_level_train.groupBy("target")
                    .agg(sf.count(sf.col("target")).alias("count_for_class"))
                    .take(2)
                ),
            )

        # 6. Fill the second level train dataset with user/item features and features of the first level models
        with JobGroupWithMetrics(self._job_group_id, "feature_processor_fit"):
            if not self.features_processor.fitted:
                self.features_processor.fit(
                    log=first_level_train,
                    user_features=user_features,
                    item_features=item_features,
                )

        self.logger.info("Adding features to second-level train dataset")

        with JobGroupWithMetrics(self._job_group_id, "_add_features_for_second_level"):
            second_level_train_to_convert = self._add_features_for_second_level(
                log_to_add_features=second_level_train,
                log_for_first_level_models=first_level_train,
                user_features=user_features,
                item_features=item_features,
            ).cache()

        self.cached_list.append(second_level_train_to_convert)

        # 7. Fit the second level model
        logger.info(f"Fitting {type(self.second_stage_model).__name__} on {second_level_train_to_convert}")
        # second_level_train_to_convert.write.parquet("hdfs://node21.bdcl:9000/tmp/second_level_train_to_convert.parquet")
        with JobGroupWithMetrics(self._job_group_id, f"{type(self.second_stage_model).__name__}_fitting"):
            self.second_stage_model.fit(second_level_train_to_convert)

        for dataframe in self.cached_list:
            unpersist_if_exists(dataframe)

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        self._job_group_id = "2stage_predict"
        self.cached_list = []

        # 1. Transform user and item features if applicable
        logger.debug(msg="Generating candidates to rerank")

        first_level_user_features = cache_if_exists(
            self.first_level_user_features_transformer.transform(user_features)
        )
        first_level_item_features = cache_if_exists(
            self.first_level_item_features_transformer.transform(item_features)
        )

        # 2. Create user/ item pairs for the train dataset of the second level (no features except relevance)
        # by making predictions with first level models and combining them into final recommendation lists
        with JobGroupWithMetrics(self._job_group_id, f"{type(self).__name__}._combine"):
            candidates = self._combine(
                    log=log,
                    k=self.num_negatives,
                    users=users,
                    items=items,
                    user_features=first_level_user_features,
                    item_features=first_level_item_features,
                    log_to_filter=log,
                    mode="union",
                    prediction_label='2'
            )

            # PERF - preventing potential losing time on repeated expensive computations
            # may be removed after testing
            candidates_cached = candidates.cache()
            candidates_cached.write.mode('overwrite').format('noop').save()
            self.cached_list.append(candidates_cached)

        logger.info(f"2candidates.columns: {candidates.columns}")

        unpersist_if_exists(first_level_user_features)
        unpersist_if_exists(first_level_item_features)

        # 3. Fill the second level recommendations dataset with user/item features
        # and features of the first level models

        self.logger.info("Adding features")
        with JobGroupWithMetrics(self._job_group_id, "_add_features_for_second_level"):
            candidates_features = self._add_features_for_second_level(
                log_to_add_features=candidates_cached,
                log_for_first_level_models=log,
                user_features=user_features,
                item_features=item_features,
            )

        logger.info(f"rel_ columns in candidates_features: {[x for x in candidates_features.columns if 'rel_' in x]}")

        # PERF - preventing potential losing time on repeated expensive computations
        candidates_features = candidates_features.cache()
        self.cached_list.extend([candidates_features, candidates_cached])

        with JobGroupWithMetrics(self._job_group_id, "candidates_features info logging"):
            self.logger.info(
                "Generated %s candidates for %s users",
                candidates_features.count(),
                candidates_features.select("user_idx").distinct().count(),
            )

        # 4. Rerank recommendations with the second level model and produce final version of recommendations
        with JobGroupWithMetrics(self._job_group_id, f"{type(self.second_stage_model).__name__}_predict"):
            predictions = self.second_stage_model.predict(data=candidates_features, k=k)

        logger.info(f"predictions.columns: {predictions.columns}")

        return predictions

    def fit_predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        :param log: input DataFrame ``[user_id, item_id, timestamp, relevance]``
        :param k: length of a recommendation list, must be smaller than the number of ``items``
        :param users: users to get recommendations for
        :param items: items to get recommendations for
        :param user_features: user features``[user_id]`` + feature columns
        :param item_features: item features``[item_id]`` + feature columns
        :param filter_seen_items: flag to removed seen items from recommendations
        :return: DataFrame ``[user_id, item_id, relevance]``
        """
        self.fit(log, user_features, item_features)
        return self.predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )

    @staticmethod
    def _optimize_one_model(
        model: BaseRecommender,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_borders: Optional[Dict[str, List[Any]]] = None,
        criterion: Metric = Precision(),
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ):
        params = model.optimize(
            train,
            test,
            user_features,
            item_features,
            param_borders,
            criterion,
            k,
            budget,
            new_study,
        )
        return params

    # pylint: disable=too-many-arguments, too-many-locals
    def optimize(
        self,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_borders: Optional[List[Dict[str, List[Any]]]] = None,
        criterion: Metric = Precision(),
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Optimize first level models with optuna.

        :param train: train DataFrame ``[user_id, item_id, timestamp, relevance]``
        :param test: test DataFrame ``[user_id, item_id, timestamp, relevance]``
        :param user_features: user features ``[user_id , timestamp]`` + feature columns
        :param item_features: item features``[item_id]`` + feature columns
        :param param_borders: list with param grids for first level models and a fallback model.
            Empty dict skips optimization for that model.
            Param grid is a dict ``{param: [low, high]}``.
        :param criterion: metric to optimize
        :param k: length of a recommendation list
        :param budget: number of points to train each model
        :param new_study: keep searching with previous study or start a new study
        :return: list of dicts of parameters
        """
        number_of_models = len(self.first_level_models)
        if self.fallback_model is not None:
            number_of_models += 1
        if number_of_models != len(param_borders):
            raise ValueError(
                "Provide search grid or None for every first level model"
            )

        first_level_user_features_tr = ToNumericFeatureTransformer()
        first_level_user_features = first_level_user_features_tr.fit_transform(
            user_features
        )
        first_level_item_features_tr = ToNumericFeatureTransformer()
        first_level_item_features = first_level_item_features_tr.fit_transform(
            item_features
        )

        first_level_user_features = cache_if_exists(first_level_user_features)
        first_level_item_features = cache_if_exists(first_level_item_features)

        params_found = []
        for i, model in enumerate(self.first_level_models):
            if param_borders[i] is None or (
                isinstance(param_borders[i], dict) and param_borders[i]
            ):
                self.logger.info(
                    "Optimizing first level model number %s, %s",
                    i,
                    model.__str__(),
                )
                params_found.append(
                    self._optimize_one_model(
                        model=model,
                        train=train,
                        test=test,
                        user_features=first_level_user_features,
                        item_features=first_level_item_features,
                        param_borders=param_borders[i],
                        criterion=criterion,
                        k=k,
                        budget=budget,
                        new_study=new_study,
                    )
                )
            else:
                params_found.append(None)

        if self.fallback_model is None or (
            isinstance(param_borders[-1], dict) and not param_borders[-1]
        ):
            return params_found, None

        self.logger.info("Optimizing fallback-model")
        fallback_params = self._optimize_one_model(
            model=self.fallback_model,
            train=train,
            test=test,
            user_features=first_level_user_features,
            item_features=first_level_item_features,
            param_borders=param_borders[-1],
            criterion=criterion,
            new_study=new_study,
        )
        unpersist_if_exists(first_level_item_features)
        unpersist_if_exists(first_level_user_features)
        return params_found, fallback_params

    def _get_nearest_items(self, items: DataFrame, metric: Optional[str] = None,
                           candidates: Optional[DataFrame] = None) -> Optional[DataFrame]:
        raise NotImplementedError("Unsupported method")
