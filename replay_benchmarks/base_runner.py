import logging
import os
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from rs_datasets import MovieLens, Netflix

from replay.data import (
    FeatureHint,
    FeatureInfo,
    FeatureSchema,
    FeatureSource,
    FeatureType,
    Dataset,
)
from replay.preprocessing.filters import MinCountFilter, NumInteractionsFilter
from replay.splitters import TimeSplitter
from replay.utils import DataFrameLike
from replay.data.nn import (
    SequenceTokenizer,
    SequentialDataset,
    TensorFeatureSource,
    TensorSchema,
    TensorFeatureInfo,
)

DATASET_MAPPINGS = {
    "zvuk": {"kaggle": "alexxl/zvuk-dataset", "file": "zvuk-interactions.parquet"},
    "megamarket": {"kaggle": "alexxl/megamarket", "file": "megamarket.parquet"},
}
SUPPORTED_RS_DATASETS = ["movielens", "netflix"]


class BaseRunner(ABC):
    def __init__(self, config):
        self.config = config
        self.model_name = config["model"]["name"]
        self.dataset_name = config["dataset"]["name"]
        self.dataset_cfg = config["dataset"]
        self.model_cfg = config["model"]["params"]
        self.mode = config["mode"]["name"]
        self.item_column = self.dataset_cfg["feature_schema"]["item_column"]
        self.user_column = self.dataset_cfg["feature_schema"]["query_column"]
        self.timestamp_column = self.dataset_cfg["feature_schema"]["timestamp_column"]
        self.tokenizer = None
        self.interactions = None
        self.user_features = None
        self.item_features = None
        self.tensor_schema = self.build_tensor_schema()
        self.setup_environment()

    def load_data(
        self,
    ) -> Tuple[
        DataFrameLike, DataFrameLike, DataFrameLike, DataFrameLike, DataFrameLike
    ]:
        """Load dataset and split into train, validation, and test sets."""
        dataset_name = self.dataset_cfg["name"]
        data_path = self.dataset_cfg["path"]
        interactions_file = os.path.join(data_path, "interactions.parquet")

        if not os.path.exists(interactions_file):
            logging.info(f"Dataset not found at {interactions_file}. Downloading...")
            self._download_dataset(data_path, dataset_name, interactions_file)

        logging.info(f"Dataset is loaded from {interactions_file}")
        interactions = pd.read_parquet(interactions_file)
        self.interactions = self._filter_data(interactions)

        splitter = TimeSplitter(
            time_threshold=self.dataset_cfg["preprocess"]["global_split_ratio"],
            drop_cold_users=True,
            drop_cold_items=True,
            item_column=self.item_column,
            query_column=self.user_column,
            timestamp_column=self.timestamp_column,
        )

        train_events, validation_events, validation_gt, test_events, test_gt = (
            self._split_data(splitter, self.interactions)
        )
        logging.info("Data split into train, validation, and test sets")
        return train_events, validation_events, validation_gt, test_events, test_gt

    def _download_dataset(
        self, data_path: str, dataset_name: str, interactions_file: str
    ):
        """Download dataset from Kaggle or rs_datasets."""
        if dataset_name in DATASET_MAPPINGS:
            self._download_kaggle_dataset(data_path, dataset_name, interactions_file)
        elif any(ds in dataset_name for ds in SUPPORTED_RS_DATASETS):
            self._download_rs_dataset(data_path, dataset_name, interactions_file)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _download_kaggle_dataset(
        self, data_path: str, dataset_name: str, interactions_file: str
    ) -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        """Download dataset from Kaggle."""
        kaggle_info = DATASET_MAPPINGS[dataset_name]
        kaggle_dataset = kaggle_info["kaggle"]
        raw_data_file = os.path.join(data_path, kaggle_info["file"])

        os.environ.setdefault("KAGGLE_USERNAME", "recsysaccelerate")
        os.environ.setdefault("KAGGLE_KEY", "6363e91b656fea576c39e4f55dcc1d00")

        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(kaggle_dataset, path=data_path, unzip=True)
        logging.info(f"Dataset downloaded and extracted to {data_path}")

        interactions = pd.read_parquet(raw_data_file)
        interactions[self.timestamp_column] = interactions[
            self.timestamp_column
        ].astype("int64")
        if dataset_name == "megamarket":
            interactions = interactions[interactions.event == 2] # take only purchase
        interactions.to_parquet(interactions_file)

    def _download_rs_dataset(
        self, data_path: str, dataset_name: str, interactions_file: str
    ) -> None:
        """Download dataset from rs_datasets."""
        if "movielens" in dataset_name:
            version = dataset_name.split("_")[1]
            movielens = MovieLens(version=version, path=data_path)
            interactions = movielens.ratings
            interactions = interactions[interactions[self.dataset_cfg["feature_schema"]["rating_column"]] > self.dataset_cfg["preprocess"]["min_rating"]]
        elif dataset_name == "netflix":
            netflix = Netflix(path=data_path)
            interactions = pd.concat([netflix.train, netflix.test]).fillna(5).reset_index(drop=True)
            interactions = interactions[interactions[self.dataset_cfg["feature_schema"]["rating_column"]] > self.dataset_cfg["preprocess"]["min_rating"]]
            interactions = interactions.sort_values(by=[self.user_column, self.timestamp_column])
            interactions[self.timestamp_column] += interactions.groupby([self.user_column, self.timestamp_column]).cumcount()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        interactions[self.timestamp_column] = interactions[
            self.timestamp_column
        ].astype("int64")
        interactions.to_parquet(interactions_file)

    def _filter_data(self, interactions: pd.DataFrame):
        """Filters raw data based on minimum interaction counts."""

        def log_min_counts(data: pd.DataFrame, message_prefix: str):
            user_min = data.groupby(self.user_column).size().min()
            item_min = data.groupby(self.item_column).size().min()
            logging.info(
                f"{message_prefix} - Min items per user: {user_min}, Min users per item: {item_min}"
            )

        log_min_counts(interactions, "Before filtering")

        interactions_filtered = MinCountFilter(
            num_entries=self.dataset_cfg["preprocess"]["min_users_per_item"],
            groupby_column=self.item_column,
        ).transform(interactions)

        interactions_filtered = MinCountFilter(
            num_entries=self.dataset_cfg["preprocess"]["min_items_per_user"],
            groupby_column=self.user_column,
        ).transform(interactions_filtered)

        log_min_counts(interactions_filtered, "After filtering")

        return interactions_filtered

    def _split_data(
        self, splitter: TimeSplitter, interactions: pd.DataFrame
    ) -> Tuple[
        DataFrameLike, DataFrameLike, DataFrameLike, DataFrameLike, DataFrameLike
    ]:
        """Split data for training, validation, and testing."""
        test_events, test_gt = splitter.split(interactions)
        validation_events, validation_gt = splitter.split(test_events)
        train_events = validation_events

        # Limit number of gt events in val and test only if max_num_test_interactions is not null
        max_test_interactions = self.dataset_cfg["preprocess"]["max_num_test_interactions"]
        logging.info(
                f"Distribution of seq_len in validation:\n{validation_gt.groupby(self.user_column)[self.item_column].agg('count').describe()}."
            )
        logging.info(
                f"Distribution of seq_len in test:\n{test_gt.groupby(self.user_column)[self.item_column].agg('count').describe()}."
            )
        if max_test_interactions is not None:
            
            validation_gt = NumInteractionsFilter(
                num_interactions=max_test_interactions,
                first=True,
                query_column=self.user_column,
                item_column=self.item_column,
                timestamp_column=self.timestamp_column,
            ).transform(validation_gt)
            logging.info(
                f"Distribution of seq_len in validation  after filtering:\n{validation_gt.groupby(self.user_column)[self.item_column].agg('count').describe()}."
            )

            test_gt = NumInteractionsFilter(
                num_interactions=max_test_interactions,
                first=True,
                query_column=self.user_column,
                item_column=self.item_column,
                timestamp_column=self.timestamp_column,
            ).transform(test_gt)
            logging.info(
                f"Distribution of seq_len in test after filtering:\n{test_gt.groupby(self.user_column)[self.item_column].agg('count').describe()}."
            )
        else:
            logging.info("max_num_test_interactions is null. Skipping filtration.")

        return train_events, validation_events, validation_gt, test_events, test_gt

    def prepare_feature_schema(self, is_ground_truth: bool) -> FeatureSchema:
        """Prepare the feature schema based on whether ground truth is needed."""
        base_features = FeatureSchema(
            [
                FeatureInfo(
                    column=self.user_column,
                    feature_hint=FeatureHint.QUERY_ID,
                    feature_type=FeatureType.CATEGORICAL,
                ),
                FeatureInfo(
                    column=self.item_column,
                    feature_hint=FeatureHint.ITEM_ID,
                    feature_type=FeatureType.CATEGORICAL,
                ),
            ]
        )
        if is_ground_truth:
            return base_features

        return base_features + FeatureSchema(
            [
                FeatureInfo(
                    column=self.timestamp_column,
                    feature_type=FeatureType.NUMERICAL,
                    feature_hint=FeatureHint.TIMESTAMP,
                ),
            ]
        )

    def build_tensor_schema(self) -> TensorSchema:
        """Build TensorSchema for the sequential model."""
        embedding_dim = self.model_cfg["embedding_dim"]
        item_feature_name = "item_id_seq"

        return TensorSchema(
            TensorFeatureInfo(
                name=item_feature_name,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[
                    TensorFeatureSource(
                        FeatureSource.INTERACTIONS,
                        self.item_column,
                    )
                ],
                feature_hint=FeatureHint.ITEM_ID,
                embedding_dim=embedding_dim,
            )
        )

    def prepare_datasets(
        self,
        train_events: DataFrameLike,
        validation_events: DataFrameLike,
        validation_gt: DataFrameLike,
        test_events: DataFrameLike,
        test_gt: DataFrameLike,
    ) -> Tuple[Dataset, Dataset, Dataset, Dataset, Dataset]:
        """Prepare Dataset objects for training, validation, and testing."""
        logging.info("Preparing Dataset objects...")
        feature_schema = self.prepare_feature_schema(is_ground_truth=False)
        ground_truth_schema = self.prepare_feature_schema(is_ground_truth=True)

        train_dataset = Dataset(
            feature_schema=feature_schema,
            interactions=train_events,
            query_features=self.user_features,
            item_features=self.item_features,
            check_consistency=True,
            categorical_encoded=False,
        )
        validation_dataset = Dataset(
            feature_schema=feature_schema,
            interactions=validation_events,
            query_features=self.user_features,
            item_features=self.item_features,
            check_consistency=True,
            categorical_encoded=False,
        )
        validation_gt_dataset = Dataset(
            feature_schema=ground_truth_schema,
            interactions=validation_gt,
            check_consistency=True,
            categorical_encoded=False,
        )
        test_dataset = Dataset(
            feature_schema=feature_schema,
            interactions=test_events,
            query_features=self.user_features,
            item_features=self.item_features,
            check_consistency=True,
            categorical_encoded=False,
        )
        test_gt_dataset = Dataset(
            feature_schema=ground_truth_schema,
            interactions=test_gt,
            check_consistency=True,
            categorical_encoded=False,
        )

        return (
            train_dataset,
            validation_dataset,
            validation_gt_dataset,
            test_dataset,
            test_gt_dataset,
        )

    def prepare_seq_datasets(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        validation_gt: Dataset,
        test_dataset: Dataset,
        test_gt: Dataset,
    ) -> Tuple[
        SequentialDataset, SequentialDataset, SequentialDataset, SequentialDataset
    ]:
        """Prepare SequentialDataset objects for training, validation, and testing."""
        logging.info("Preparing SequentialDataset objects...")
        self.tokenizer = self.tokenizer or self._initialize_tokenizer(train_dataset)

        seq_train_dataset = self.tokenizer.transform(train_dataset)
        seq_validation_dataset, seq_validation_gt = self._prepare_sequential_validation(
            validation_dataset, validation_gt
        )
        seq_test_dataset = self._prepare_sequential_test(test_dataset, test_gt)

        return (
            seq_train_dataset,
            seq_validation_dataset,
            seq_validation_gt,
            seq_test_dataset,
        )

    def _initialize_tokenizer(self, train_dataset: Dataset) -> SequenceTokenizer:
        """Initialize and fit the SequenceTokenizer."""
        tokenizer = SequenceTokenizer(
            self.tensor_schema, allow_collect_to_master=True, handle_unknown_rule="drop"
        )
        tokenizer.fit(train_dataset)
        return tokenizer

    def _prepare_sequential_validation(
        self, validation_dataset: Dataset, validation_gt: Dataset
    ) -> Tuple[SequentialDataset, SequentialDataset]:
        """Prepare sequential datasets for validation."""
        seq_validation_dataset = self.tokenizer.transform(validation_dataset)
        seq_validation_gt = self.tokenizer.transform(
            validation_gt, [self.tensor_schema.item_id_feature_name]
        )

        return SequentialDataset.keep_common_query_ids(
            seq_validation_dataset, seq_validation_gt
        )

    def _prepare_sequential_test(
        self, test_dataset: Dataset, test_gt: Dataset
    ) -> SequentialDataset:
        """Prepare sequential dataset for testing."""
        test_query_ids = test_gt.query_ids
        test_query_ids_np = self.tokenizer.query_id_encoder.transform(test_query_ids)[
            self.user_column
        ].values
        return self.tokenizer.transform(test_dataset).filter_by_query_id(
            test_query_ids_np
        )

    def setup_environment(self):
        os.environ["CUDA_DEVICE_ORDER"] = self.config["env"]["CUDA_DEVICE_ORDER"]
        os.environ["OMP_NUM_THREADS"] = self.config["env"]["OMP_NUM_THREADS"]
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config["env"]["CUDA_VISIBLE_DEVICES"]
        os.environ["KAGGLE_USERNAME"] = "recsysaccelerate"
        os.environ["KAGGLE_KEY"] = "6363e91b656fea576c39e4f55dcc1d00"

    @abstractmethod
    def run(self):
        """Run method to be implemented in derived classes."""
        raise NotImplementedError
