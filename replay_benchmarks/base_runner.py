import logging
import os
from abc import ABC, abstractmethod
from typing import Tuple

from replay_benchmarks.preprocessing import DatasetManager
from replay.data import (
    FeatureHint,
    FeatureInfo,
    FeatureSchema,
    FeatureSource,
    FeatureType,
    Dataset,
)
from replay.utils import DataFrameLike
from replay.data.nn import (
    SequenceTokenizer,
    SequentialDataset,
    TensorFeatureSource,
    TensorSchema,
    TensorFeatureInfo,
)


class BaseRunner(ABC):
    def __init__(self, config):
        self.config = config
        self.model_name = config["model"]["name"]
        self.model_save_name = config["model"]["save_name"]
        self.dataset_name = config["dataset"]["name"]
        self.dataset_cfg = config["dataset"]
        self.model_cfg = config["model"]["params"]
        self.mode = config["mode"]["name"]
        self.item_column = self.dataset_cfg["feature_schema"]["item_column"]
        self.user_column = self.dataset_cfg["feature_schema"]["query_column"]
        self.timestamp_column = self.dataset_cfg["feature_schema"]["timestamp_column"]
        self.tokenizer = None
        self.dataset_manager = DatasetManager(config)
        self.tensor_schema = self.build_tensor_schema()
        self.setup_environment()

    def load_data(self):
        """Load preprocessed data splits."""
        splits = self.dataset_manager.load_data()
        return (
            splits["train"],
            splits["validation"],
            splits["validation_gt"],
            splits["test"],
            splits["test_gt"],
        )

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
        embedding_dim = self.model_cfg["training_params"]["embedding_dim"]
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
        train_dataset = Dataset(
            feature_schema=self.prepare_feature_schema(is_ground_truth=False),
            interactions=train_events,
            check_consistency=True,
            categorical_encoded=False,
        )
        validation_dataset = Dataset(
            feature_schema=self.prepare_feature_schema(is_ground_truth=False),
            interactions=validation_events,
            check_consistency=True,
            categorical_encoded=False,
        )
        validation_gt_dataset = Dataset(
            feature_schema=self.prepare_feature_schema(is_ground_truth=True),
            interactions=validation_gt,
            check_consistency=True,
            categorical_encoded=False,
        )
        test_dataset = Dataset(
            feature_schema=self.prepare_feature_schema(is_ground_truth=False),
            interactions=test_events,
            check_consistency=True,
            categorical_encoded=False,
        )
        test_gt_dataset = Dataset(
            feature_schema=self.prepare_feature_schema(is_ground_truth=True),
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
        logging.info("Initializing and fitting SequenceTokenizer.")
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
