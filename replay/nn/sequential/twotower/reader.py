from typing import Protocol

import numpy as np
import pandas as pd
import torch

from replay.data import FeatureSource
from replay.data.nn import TensorSchema


class FeaturesReaderProtocol(Protocol):
    def __getitem__(self, key: str) -> torch.Tensor: ...


class FeaturesReader:
    """
    Prepares a dict of item features values that will be used for training and inference of the Item Tower.
    """

    def __init__(self, schema: TensorSchema, metadata: dict, path: str):
        """
        :param schema: the same tensor schema used in TwoTower model.
        :param metadata: A dictionary of feature names that
            associated with its shape and padding_value.\n
            For details, see the section :ref:`parquet-processing`.
        :param path: path to parquet with dataframe of item features.\n
            **Note:**\n
            1. Dataframe columns must be already encoded.\n
            2. Every feature for item "tower" in `schema` must contain ``feature_sources`` with the names
               of the source features to create correct inverse mapping.
               Also, for each such feature one of the requirements must be met: the ``schema`` for the feature must
               contain ``feature_sources`` with a source of type ``FeatureSource.ITEM_FEATURES``
               or hint type ``FeatureHint.ITEM_ID``.

        """
        item_feature_names = [
            info.feature_source.column
            for name, info in schema.items()
            if info.feature_source.source == FeatureSource.ITEM_FEATURES or name == schema.item_id_feature_name
        ]
        metadata_names = list(metadata.keys())

        if (unique_metadata_names := set(metadata_names)) != (unique_schema_names := set(item_feature_names)):
            extra_metadata_names = unique_metadata_names - unique_schema_names
            if extra_metadata_names:
                msg = (
                    "The metadata contains information about the following columns,"
                    f"which are not described in schema: {extra_metadata_names}."
                )
                raise ValueError(msg)

            extra_schema_names = unique_schema_names - unique_metadata_names
            if extra_schema_names:
                msg = (
                    "The schema contains information about the following columns,"
                    f"which are not described in metadata: {extra_schema_names}."
                )
                raise ValueError(msg)

        features = pd.read_parquet(
            path=path,
            columns=metadata_names,
        )

        def add_padding(row: np.array, max_len: int, padding_value: int):
            return np.concatenate(([padding_value] * (max_len - len(row)), row))

        for k, v in metadata.items():
            if not v:
                continue
            features[k] = features[k].apply(add_padding, args=(v["shape"], v["padding"]))

        inverse_feature_names_mapping = {
            schema[feature].feature_source.column: feature for feature in item_feature_names
        }
        features.rename(columns=inverse_feature_names_mapping, inplace=True)
        features.sort_values(by=schema.item_id_feature_name, inplace=True)
        features.reset_index(drop=True, inplace=True)

        self._features = {}

        for k in features.columns:
            if schema[k].is_list:
                feature = features[k].to_list()
            else:
                feature = features[k].to_numpy(dtype=np.float32 if schema[k].is_num else np.int64)
            feature_tensor = torch.asarray(
                feature,
                dtype=torch.float32 if schema[k].is_num else torch.int64,
            )
            self._features[k] = feature_tensor

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._features[key]
