from pathlib import Path
from urllib.parse import urlencode
from zipfile import ZipFile
import os
import pandas as pd
import requests
import logging
import json

from rs_datasets import MovieLens, Netflix

from replay.splitters import TimeSplitter, LastNSplitter, ColdUserRandomSplitter
from replay.preprocessing.filters import MinCountFilter

DATASET_MAPPINGS = {
    "zvuk": {"kaggle": "alexxl/zvuk-dataset", "file": "zvuk-interactions.parquet"},
    "megamarket": {"kaggle": "alexxl/megamarket", "file": "megamarket.parquet"},
    "yelp": {
        "kaggle": "yelp-dataset/yelp-dataset",
        "file": "yelp_academic_dataset_review.json",
    },
}
AMAZON_URLS = {
    "games": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Toys_and_Games.csv.gz",
    "beauty": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Beauty_and_Personal_Care.csv.gz",
    "sports": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Sports_and_Outdoors.csv.gz",
}
YANDEX_URLS = {
    "30music": "https://disk.yandex.ru/d/XwRRFVuJG-ECng",
    "gowalla": "https://disk.yandex.ru/d/PaPpcpwkWcHRHg",
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
            "validation_gt": os.path.join(
                self.split_cache_dir, "validation_gt.parquet"
            ),
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
        elif dataset_name in AMAZON_URLS:
            self._download_csv_gz_dataset(dataset_name, interactions_file)
        elif dataset_name in YANDEX_URLS:
            self._load_yandex_dataset(dataset_name)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _load_yandex_dataset(self, dataset_name: str) -> None:
        data_src = Path(os.path.dirname(self.data_path))
        data_src.mkdir(exist_ok=True)

        url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?" + \
                urlencode({"public_key": YANDEX_URLS[dataset_name]})

        download_url = requests.get(url, timeout=10).json()["href"]

        with open("temp.zip", 'wb') as file:
            for i, chunk in enumerate(
                requests.get(download_url, stream=True, timeout=10).iter_content(1024)
            ):
                file.write(chunk)
                if i < 1024:
                    output = f"{round(i, 2)} K"
                elif 1024 <= i < 1024 ** 2:
                    output = f"{round(i / 1024, 2)} M"
                else:
                    output = f"{round(i / 1024 ** 2, 2)} G"
                print(f"Total downloaded - " + output.rjust(9), end="\r")

        with ZipFile("temp.zip") as zfile:
            zfile.extractall(data_src)

        os.remove("temp.zip")

    def _download_csv_gz_dataset(self, dataset_name: str, interactions_file: str):
        """Download and preprocess datasets from CSV GZ format."""
        url = AMAZON_URLS[dataset_name]
        raw_file = os.path.join(self.data_path, f"{dataset_name}.csv.gz")

        if not os.path.exists(raw_file):
            logging.info(f"Downloading {dataset_name} dataset from {url}")
            response = requests.get(url, stream=True, verify=False)
            with open(raw_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            logging.info("Download complete.")

        logging.info(f"Processing {dataset_name} dataset...")
        df = pd.read_csv(raw_file, compression="gzip")

        column_mapping = {
            "reviewerID": self.user_column,
            "parent_asin": self.item_column,
            "unixReviewTime": self.timestamp_column,
            "rating": self.config["dataset"]["feature_schema"]["rating_column"],
        }
        df.rename(columns=column_mapping, inplace=True)

        min_rating = self.config["dataset"]["preprocess"]["min_rating"]
        df = df[
            df[self.config["dataset"]["feature_schema"]["rating_column"]] > min_rating
        ]

        df[self.timestamp_column] = df[self.timestamp_column].astype("int64")
        df.to_parquet(interactions_file)
        logging.info(
            f"{dataset_name} dataset processed and saved at {interactions_file}"
        )

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

        if dataset_name == "yelp":
            prime = []
            for line in open(raw_data_file, "r", encoding="UTF-8"):
                val = json.loads(line)
                prime.append(
                    [val[self.user_column], val["business_id"], val["stars"], val["date"]]
                )
            interactions = pd.DataFrame(
                prime,
                columns=[
                    self.user_column,
                    self.item_column,
                    self.config["dataset"]["feature_schema"]["rating_column"],
                    self.timestamp_column,
                ],
            )
            interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
        else:
            interactions = pd.read_parquet(raw_data_file)
        interactions[self.timestamp_column] = interactions[
            self.timestamp_column
        ].astype("int64")
        if dataset_name == "megamarket":
            interactions = interactions[interactions.event == 2]  # take only purchase
        if dataset_name == "zvuk":
            interactions.rename(columns={"track_id": self.item_column}, inplace=True)
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
            interactions[self.timestamp_column] = pd.to_datetime(
                interactions[self.timestamp_column]
            )
            interactions[self.timestamp_column] += pd.to_timedelta(
                interactions.groupby(
                    [self.user_column, self.timestamp_column]
                ).cumcount(),
                unit="s",
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
        global_splitter = TimeSplitter(
            time_threshold=self.config["dataset"]["preprocess"]["global_split_ratio"],
            drop_cold_users=False,
            drop_cold_items=True,
            item_column=self.item_column,
            query_column=self.user_column,
            timestamp_column=self.timestamp_column,
        )
        val_splitter = ColdUserRandomSplitter(
            test_size=self.config["dataset"]["preprocess"]["val_users_ratio"],
            drop_cold_items=True,
            query_column=self.user_column,
            item_column=self.item_column,
            seed=42,
        )
        loo_splitter = LastNSplitter(
            N=1,
            drop_cold_users=True,
            drop_cold_items=False,
            divide_column=self.user_column,
            query_column=self.user_column,
            item_column=self.item_column,
            strategy="interactions",
            timestamp_column=self.timestamp_column,
        )

        train, raw_test = global_splitter.split(interactions)
        train_events, val = val_splitter.split(train)
        test_users = set(raw_test[self.user_column]) - set(val[self.user_column])
        test_events, test_gt = loo_splitter.split(
            interactions[
                (interactions[self.user_column].isin(test_users))
                & interactions[self.item_column].isin(train_events[self.item_column].unique())
            ]
        )
        validation_events, validation_gt = loo_splitter.split(val)
        test_gt = test_gt[test_gt[self.item_column].isin(train_events[self.item_column])]
        test_gt = test_gt[test_gt[self.user_column].isin(train_events[self.user_column])]

        return {
            "train": train_events,
            "validation": validation_events,
            "validation_gt": validation_gt,
            "test": test_events,
            "test_gt": test_gt,
        }
