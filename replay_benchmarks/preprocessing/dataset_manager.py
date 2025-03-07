import os
import pandas as pd
import logging

from rs_datasets import MovieLens, Netflix

from replay.splitters import TimeSplitter
from replay.preprocessing.filters import MinCountFilter, NumInteractionsFilter

DATASET_MAPPINGS = {
    "zvuk": {"kaggle": "alexxl/zvuk-dataset", "file": "zvuk-interactions.parquet"},
    "megamarket": {"kaggle": "alexxl/megamarket", "file": "megamarket.parquet"},
}
SUPPORTED_RS_DATASETS = ["movielens", "netflix"]


class DatasetManager:
    def __init__(self, config):
        self.config = config
        self.data_path = config["dataset"]["path"]
        self.dataset_name = config["dataset"]["name"]
        self.item_column = config["dataset"]["feature_schema"]["item_column"]
        self.user_column = config["dataset"]["feature_schema"]["query_column"]
        self.timestamp_column = config["dataset"]["feature_schema"]["timestamp_column"]
        self.split_cache_dir = os.path.join(self.data_path, "split_cache")

        os.makedirs(self.split_cache_dir, exist_ok=True)

    def load_data(self):
        split_paths = {
            "train": os.path.join(self.split_cache_dir, "train.parquet"),
            "validation": os.path.join(self.split_cache_dir, "validation.parquet"),
            "validation_gt": os.path.join(self.split_cache_dir, "validation_gt.parquet"),
            "test": os.path.join(self.split_cache_dir, "test.parquet"),
            "test_gt": os.path.join(self.split_cache_dir, "test_gt.parquet"),
        }

        if all(os.path.exists(path) for path in split_paths.values()):
            logging.info("Loading preprocessed splits from cache.")
            return {key: pd.read_parquet(path) for key, path in split_paths.items()}

        logging.info("Preprocessing data as splits are not cached.")
        raw_data = self._load_raw_data()
        filtered_data = self._filter_data(raw_data)
        splits = self._split_data(filtered_data)

        # Save preprocessed splits
        for split_name, split_data in splits.items():
            split_data.to_parquet(split_paths[split_name], index=False)

        return splits

    def _load_raw_data(self):
        interactions_file = os.path.join(self.data_path, "interactions.parquet")
        if not os.path.exists(interactions_file):
            logging.info(f"Dataset not found at {interactions_file}. Downloading...")
            self._download_dataset(self.data_path, self.dataset_name, interactions_file)
        logging.info(f"Loading raw dataset from {interactions_file}")
        return pd.read_parquet(interactions_file)

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
            interactions = interactions[interactions.event == 2]  # take only purchase
        if dataset_name == "zvuk":
            interactions.rename(columns={'track_id': 'item_id'}, inplace=True)
        interactions.to_parquet(interactions_file)

    def _download_rs_dataset(
        self, data_path: str, dataset_name: str, interactions_file: str
    ) -> None:
        """Download dataset from rs_datasets."""
        if "movielens" in dataset_name:
            version = dataset_name.split("_")[1]
            movielens = MovieLens(version=version, path=data_path)
            interactions = movielens.ratings
            interactions = interactions[
                interactions[self.config["dataset"]["feature_schema"]["rating_column"]]
                > self.config["dataset"]["preprocess"]["min_rating"]
            ]
        elif dataset_name == "netflix":
            netflix = Netflix(path=data_path)
            interactions = (
                pd.concat([netflix.train, netflix.test])
                .fillna(5)
                .reset_index(drop=True)
            )
            interactions = interactions[
                interactions[self.config["dataset"]["feature_schema"]["rating_column"]]
                > self.config["dataset"]["preprocess"]["min_rating"]
            ]
            interactions = interactions.sort_values(
                by=[self.user_column, self.timestamp_column]
            )
            interactions[self.timestamp_column] = pd.to_datetime(interactions[self.timestamp_column])
            interactions[self.timestamp_column] += pd.to_timedelta(
                interactions.groupby([self.user_column, self.timestamp_column]).cumcount(), 
                unit="s"
            )
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

        interactions = MinCountFilter(
            num_entries=self.config["dataset"]["preprocess"]["min_users_per_item"],
            groupby_column=self.item_column,
        ).transform(interactions)

        interactions = MinCountFilter(
            num_entries=self.config["dataset"]["preprocess"]["min_items_per_user"],
            groupby_column=self.user_column,
        ).transform(interactions)

        log_min_counts(interactions, "After filtering")

        return interactions

    def _split_data(self, interactions):
        """Split data for training, validation, and testing."""
        splitter = TimeSplitter(
            time_threshold=self.config["dataset"]["preprocess"]["global_split_ratio"],
            drop_cold_users=True,
            drop_cold_items=True,
            item_column=self.item_column,
            query_column=self.user_column,
            timestamp_column=self.timestamp_column,
        )

        test_events, test_gt = splitter.split(interactions)
        validation_events, validation_gt = splitter.split(test_events)
        train_events = validation_events

        test_gt = test_gt[
            test_gt[self.item_column].isin(train_events[self.item_column])
        ]
        test_gt = test_gt[
            test_gt[self.user_column].isin(train_events[self.user_column])
        ]

        # Limit number of gt events in val and test only if max_num_test_interactions is not null
        max_test_interactions = self.config["dataset"]["preprocess"][
            "max_num_test_interactions"
        ]
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

        return {
            "train": train_events,
            "validation": validation_events,
            "validation_gt": validation_gt,
            "test": test_events,
            "test_gt": test_gt,
        }
