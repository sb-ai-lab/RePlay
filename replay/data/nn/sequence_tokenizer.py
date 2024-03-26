import pickle
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import polars as pl
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame

from replay.data import Dataset, FeatureHint, FeatureSchema, FeatureSource
from replay.data.dataset_utils import DatasetLabelEncoder
from replay.preprocessing import LabelEncoder
from replay.preprocessing.label_encoder import HandleUnknownStrategies

from .schema import TensorFeatureInfo, TensorFeatureSource, TensorSchema
from .sequential_dataset import PandasSequentialDataset, PolarsSequentialDataset, SequentialDataset
from .utils import ensure_pandas, groupby_sequences

SequenceDataFrameLike = Union[PandasDataFrame, PolarsDataFrame]


class SequenceTokenizer:
    """
    Data tokenizer for transformers
    """

    def __init__(
        self,
        tensor_schema: TensorSchema,
        handle_unknown_rule: HandleUnknownStrategies = "error",
        default_value_rule: Optional[Union[int, str]] = None,
        allow_collect_to_master: bool = False,
    ) -> None:
        """
        :param tensor_schema: tensor schema of tensor features
        :param handle_unknown_rule: handle unknown labels rule for LabelEncoder,
            values are in ('error', 'use_default_value').
            Default: `error`
        :param default_value: Default value that will fill the unknown labels after transform.
            When the parameter handle_unknown is set to ``use_default_value``,
            this parameter is required and will set the encoded value of unknown labels.
            It has to be distinct from the values used to encode any of the labels in fit.
            If ``None``, then keep null.
            If ``int`` value, then fill by that value.
            If ``str`` value, should be \"last\" only, then fill by ``n_classes`` value.
            Default: ``None``.
        :param allow_collect_to_master: Flag allowing spark to make a collection to the master node,
            Default: ``False``.
        """
        self._tensor_schema = tensor_schema
        self._allow_collect_to_master = allow_collect_to_master
        self._encoder = DatasetLabelEncoder(
            handle_unknown_rule=handle_unknown_rule, default_value_rule=default_value_rule
        )
        self._check_tensor_schema(self._tensor_schema)

    def fit(self, dataset: Dataset) -> "SequenceTokenizer":
        """
        :param dataset: input dataset to fit

        :returns: fitted SequenceTokenizer
        """
        self._check_if_tensor_schema_matches_data(dataset, self._tensor_schema)
        self._encoder.fit(dataset)
        return self

    def transform(
        self,
        dataset: Dataset,
        tensor_features_to_keep: Optional[Sequence[str]] = None,
    ) -> SequentialDataset:
        """
        :param dataset: input dataset to transform
        :param tensor_features_to_keep: specified feature names to transform
        :returns: SequentialDataset
        """
        self._check_if_tensor_schema_matches_data(dataset, self._tensor_schema, tensor_features_to_keep)
        return self._transform_unchecked(dataset, tensor_features_to_keep)

    def fit_transform(
        self,
        dataset: Dataset,
    ) -> SequentialDataset:
        """
        :param dataset: input dataset to transform
        :returns: SequentialDataset
        """
        return self.fit(dataset)._transform_unchecked(dataset)

    @property
    def tensor_schema(self) -> TensorSchema:
        """
        :returns: tensor schema
        """
        return self._tensor_schema

    @property
    def query_id_encoder(self) -> LabelEncoder:
        """
        :returns: encoder for query id
        """
        return self._encoder.query_id_encoder

    @property
    def item_id_encoder(self) -> LabelEncoder:
        """
        :returns: encoder for item id
        """
        return self._encoder.item_id_encoder

    @property
    def query_and_item_id_encoder(self) -> LabelEncoder:
        """
        :returns: encoder for query and item id
        """
        return self._encoder.query_and_item_id_encoder

    @property
    def interactions_encoder(self) -> Optional[LabelEncoder]:
        """
        :returns: encoder for interactions
        """
        return self._encoder.interactions_encoder

    @property
    def query_features_encoder(self) -> Optional[LabelEncoder]:
        """
        :returns: encoder for query features
        """
        return self._encoder.query_features_encoder

    @property
    def item_features_encoder(self) -> Optional[LabelEncoder]:
        """
        :returns: encoder for item features
        """
        return self._encoder.item_features_encoder

    def _transform_unchecked(
        self,
        dataset: Dataset,
        tensor_features_to_keep: Optional[Sequence[str]] = None,
    ) -> SequentialDataset:
        schema = self._tensor_schema
        if tensor_features_to_keep is not None:
            schema = schema.subset(tensor_features_to_keep)

        matched_dataset = self._match_features_with_tensor_schema(dataset, schema)

        is_polars = isinstance(dataset.interactions, PolarsDataFrame)
        encoded_dataset = self._encode_dataset(matched_dataset)
        is_polars = isinstance(encoded_dataset.interactions, PolarsDataFrame)
        grouped_interactions, query_features, item_features = self._group_dataset(encoded_dataset)
        sequence_features = self._make_sequence_features(
            schema,
            dataset.feature_schema,
            grouped_interactions,
            query_features,
            item_features,
        )

        assert self._tensor_schema.item_id_feature_name

        dataset_type = PolarsSequentialDataset if is_polars else PandasSequentialDataset

        return dataset_type(
            tensor_schema=schema,
            query_id_column=dataset.feature_schema.query_id_column,
            item_id_column=self._tensor_schema.item_id_feature_name,
            sequences=sequence_features,
        )

    def _encode_dataset(self, dataset: Dataset) -> Dataset:
        encoded_dataset = self._encoder.transform(dataset)
        return encoded_dataset

    def _group_dataset(
        self,
        dataset: Dataset,
    ) -> Tuple[SequenceDataFrameLike, Optional[SequenceDataFrameLike], Optional[SequenceDataFrameLike]]:
        grouped_interactions = groupby_sequences(
            events=dataset.interactions,
            groupby_col=dataset.feature_schema.query_id_column,
            sort_col=dataset.feature_schema.interactions_timestamp_column,
        )

        if isinstance(grouped_interactions, PolarsDataFrame):
            return (
                grouped_interactions.sort(dataset.feature_schema.query_id_column),
                dataset.query_features,
                dataset.item_features,
            )

        # We sort by QUERY_ID to make sure order is deterministic
        grouped_interactions_pd = ensure_pandas(
            grouped_interactions,
            self._allow_collect_to_master,
        )
        grouped_interactions_pd.sort_values(dataset.feature_schema.query_id_column, inplace=True, ignore_index=True)

        query_features_pd: Optional[PandasDataFrame] = None
        item_features_pd: Optional[PandasDataFrame] = None

        if dataset.query_features is not None:
            query_features_pd = ensure_pandas(dataset.query_features, self._allow_collect_to_master)
        if dataset.item_features is not None:
            item_features_pd = ensure_pandas(dataset.item_features, self._allow_collect_to_master)

        return grouped_interactions_pd, query_features_pd, item_features_pd

    def _make_sequence_features(
        self,
        schema: TensorSchema,
        feature_schema: FeatureSchema,
        grouped_interactions: SequenceDataFrameLike,
        query_features: Optional[SequenceDataFrameLike],
        item_features: Optional[SequenceDataFrameLike],
    ) -> SequenceDataFrameLike:
        processor = _SequenceProcessor(
            tensor_schema=schema,
            query_id_column=feature_schema.query_id_column,
            item_id_column=feature_schema.item_id_column,
            grouped_interactions=grouped_interactions,
            query_features=query_features,
            item_features=item_features,
        )

        if isinstance(grouped_interactions, PolarsDataFrame):
            return processor.process_features_polars()

        all_features: Dict[str, Union[np.ndarray, List[np.ndarray]]] = {}
        all_features[feature_schema.query_id_column] = grouped_interactions[feature_schema.query_id_column].values

        for tensor_feature_name in schema:
            all_features[tensor_feature_name] = processor.process_feature(tensor_feature_name)

        return PandasDataFrame(all_features)

    @classmethod
    def _match_features_with_tensor_schema(
        cls,
        dataset: Dataset,
        tensor_schema: TensorSchema,
    ) -> Dataset:
        feature_subset_filter = cls._get_features_filter_from_schema(
            tensor_schema,
            query_id_column=dataset.feature_schema.query_id_column,
            item_id_column=dataset.feature_schema.item_id_column,
        )

        # We need to keep timestamp column in dataset as it's used to sort interactions
        timestamp_column = dataset.feature_schema.interactions_timestamp_column
        if timestamp_column:
            feature_subset_filter.add(timestamp_column)

        subset = dataset.subset(feature_subset_filter)
        return subset

    @classmethod
    def _get_features_filter_from_schema(
        cls,
        tensor_schema: TensorSchema,
        query_id_column: str,
        item_id_column: str,
    ) -> Set[str]:
        # We need only features, which related to tensor schema, otherwise feature should
        # be ignored for efficiency reasons. The code below does feature filtering, and
        # keeps features used as a source in tensor schema.

        # Query and item IDs are always needed
        features_subset: List[str] = [
            query_id_column,
            item_id_column,
        ]

        for tensor_feature in tensor_schema.values():
            source = tensor_feature.feature_source
            assert source is not None

            # Some columns already added to encoder, skip them
            if source.column in features_subset:
                continue

            if isinstance(source.source, FeatureSource):
                features_subset.append(source.column)
            else:
                assert False, "Unknown tensor feature source"

        return set(features_subset)

    @classmethod
    def _check_tensor_schema(cls, tensor_schema: TensorSchema) -> None:
        # Check consistency of sequential features
        for tensor_feature in tensor_schema.all_features:
            feature_sources = tensor_feature.feature_sources
            if not feature_sources:
                msg = "All tensor features must have sources defined"
                raise ValueError(msg)

            source_tables: List[FeatureSource] = [s.source for s in feature_sources]

            unexpected_tables = list(filter(lambda x: not isinstance(x, FeatureSource), source_tables))
            if len(unexpected_tables) > 0:
                msg = f"Found unexpected source tables: {unexpected_tables}"
                raise ValueError(msg)

            if not tensor_feature.is_seq:
                if FeatureSource.INTERACTIONS in source_tables:
                    msg = "Interaction features must be treated as sequential"
                    raise ValueError(msg)

                if FeatureSource.ITEM_FEATURES in source_tables:
                    msg = "Item features must be treated as sequential"
                    raise ValueError(msg)

    @classmethod
    def _check_if_tensor_schema_matches_data(  # noqa: C901
        cls,
        dataset: Dataset,
        tensor_schema: TensorSchema,
        tensor_features_to_keep: Optional[Sequence[str]] = None,
    ) -> None:
        # Check if all source columns specified in tensor schema exist in provided data frames
        sources_for_tensors: List[TensorFeatureSource] = []
        for tensor_feature_name, tensor_feature in tensor_schema.items():
            if tensor_features_to_keep is not None and tensor_feature_name not in tensor_features_to_keep:
                continue

            if tensor_feature.feature_sources:
                sources_for_tensors += tensor_feature.feature_sources

        query_id_column = dataset.feature_schema.query_id_column
        item_id_column = dataset.feature_schema.item_id_column

        interaction_feature_columns = {
            *dataset.feature_schema.interaction_features.columns,
            query_id_column,
            item_id_column,
        }
        query_feature_columns = {*dataset.feature_schema.query_features.columns, query_id_column}
        item_feature_columns = {*dataset.feature_schema.item_features.columns, item_id_column}

        for feature_source in sources_for_tensors:
            assert feature_source is not None
            if feature_source.source == FeatureSource.INTERACTIONS:
                if feature_source.column not in interaction_feature_columns:
                    msg = f"Expected column '{feature_source.column}' in dataset"
                    raise ValueError(msg)
            elif feature_source.source == FeatureSource.QUERY_FEATURES:
                if dataset.query_features is None:
                    msg = f"Expected column '{feature_source.column}', but query features are not specified"
                    raise ValueError(msg)
                if feature_source.column not in query_feature_columns:
                    msg = f"Expected column '{feature_source.column}' in query features data frame"
                    raise ValueError(msg)
            elif feature_source.source == FeatureSource.ITEM_FEATURES:
                if dataset.item_features is None:
                    msg = f"Expected column '{feature_source.column}', but item features are not specified"
                    raise ValueError(msg)
                if feature_source.column not in item_feature_columns:
                    msg = f"Expected column '{feature_source.column}' in item features data frame"
                    raise ValueError(msg)
            else:
                msg = f"Found unexpected table '{feature_source.source}' in tensor schema"
                raise ValueError(msg)

        # Check if user ID and item ID columns are consistent with tensor schema
        if tensor_schema.query_id_feature_name is not None:
            tensor_feature = tensor_schema.query_id_features.item()
            assert tensor_feature.feature_source
            if tensor_feature.feature_source.column != dataset.feature_schema.query_id_column:
                msg = "Tensor schema query ID source colum does not match query ID in data frame"
                raise ValueError(msg)

        if tensor_schema.item_id_feature_name is None:
            msg = "Tensor schema must have item id feature defined"
            raise ValueError(msg)

        tensor_feature = tensor_schema.item_id_features.item()
        assert tensor_feature.feature_source
        if tensor_feature.feature_source.column != dataset.feature_schema.item_id_column:
            msg = "Tensor schema item ID source colum does not match item ID in data frame"
            raise ValueError(msg)

    @classmethod
    def load(cls, path: str) -> "SequenceTokenizer":
        """
        Load tokenizer object from the given path.

        :param path: Path to load the tokenizer.

        :returns: Loaded tokenizer object.
        """
        with open(path, "rb") as file:
            tokenizer = pickle.load(file)

        return tokenizer

    def save(self, path: str) -> None:
        """
        Save the tokenizer to the given path.

        :param path: Path to save the tokenizer.
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)


class _SequenceProcessor:
    """
    Class to process sequences of different categorical and numerical features.

    Processing performs over all features in `tensor_schema`. Each feature processing steps
    depends on feature type (categorical/numerical), feature source (interactions/query features/item features)
    and `grouped_interactions` data format (Pandas/Polars).
    If `grouped_interactions` is `PolarsDataFrame` object, then method `process_features_polars` is called.
    If `grouped_interactions` is `PandasDataFrame` object, then method `process_features` is called,
        with passing all tensor features one by one.
    """

    def __init__(
        self,
        tensor_schema: TensorSchema,
        query_id_column: str,
        item_id_column: str,
        grouped_interactions: SequenceDataFrameLike,
        query_features: Optional[SequenceDataFrameLike] = None,
        item_features: Optional[SequenceDataFrameLike] = None,
    ) -> None:
        self._tensor_schema = tensor_schema
        self._query_id_column = query_id_column
        self._item_id_column = item_id_column
        self._grouped_interactions = grouped_interactions
        self._is_polars = isinstance(grouped_interactions, PolarsDataFrame)
        if not self._is_polars:
            self._query_features = (
                query_features.set_index(self._query_id_column).sort_index() if query_features is not None else None
            )
            self._item_features = (
                item_features.set_index(self._item_id_column).sort_index() if item_features is not None else None
            )
        else:
            self._query_features = query_features
            self._item_features = item_features

    def process_feature(self, tensor_feature_name: str) -> List[np.ndarray]:
        """
        Process each tensor feature for Pandas dataframes.

        :param tensor_feature_name: name of feature to process.

        :returns: values for provided tensor_feature_name column
        """
        tensor_feature = self._tensor_schema[tensor_feature_name]
        if tensor_feature.is_cat:
            return self._process_cat_feature(tensor_feature)
        if tensor_feature.is_num:
            return self._process_num_feature(tensor_feature)
        assert False, "Unknown tensor feature type"

    def process_features_polars(self) -> PolarsDataFrame:
        """
        Process all features in `tensor_schema` for Polars dataframes.
        Each Polars processing step returns DataFrame with query and target column
        to join in result dataframe.

        :returns: processed Polars DataFrame with all features from tensor schema.
        """
        data = self._grouped_interactions.select(self._query_id_column)
        for tensor_feature_name in self._tensor_schema:
            tensor_feature = self._tensor_schema[tensor_feature_name]
            if tensor_feature.is_cat:
                data = data.join(self._process_cat_feature(tensor_feature), on=self._query_id_column, how="left")
            elif tensor_feature.is_num:
                data = data.join(self._process_num_feature(tensor_feature), on=self._query_id_column, how="left")
            else:
                assert False, "Unknown tensor feature type"
        return data

    def _process_cat_feature(self, tensor_feature: TensorFeatureInfo) -> List[np.ndarray]:
        """
        Process categorical tensor feature depends on it source.
        """
        assert tensor_feature.feature_source is not None
        if tensor_feature.feature_source.source == FeatureSource.INTERACTIONS:
            return self._process_cat_interaction_feature(tensor_feature)
        if tensor_feature.feature_source.source == FeatureSource.QUERY_FEATURES:
            return self._process_cat_query_feature(tensor_feature)
        if tensor_feature.feature_source.source == FeatureSource.ITEM_FEATURES:
            return self._process_cat_item_feature(tensor_feature)
        assert False, "Unknown tensor feature source table"

    def _process_num_feature_polars(self, tensor_feature: TensorFeatureInfo) -> PolarsDataFrame:
        def get_sequence(user, source, data):
            if source.source == FeatureSource.INTERACTIONS:
                return np.array(
                    self._grouped_interactions.filter(pl.col(self._query_id_column) == user)[source.column][0],
                    dtype=np.float32,
                ).tolist()
            elif source.source == FeatureSource.ITEM_FEATURES:
                return (
                    pl.DataFrame({self._item_id_column: data})
                    .join(self._item_features, on=self._item_id_column, how="left")
                    .select(source.column)
                    .to_numpy()
                    .reshape(-1)
                    .tolist()
                )
            else:
                assert False, "Unknown tensor feature source table"

        result = (
            self._grouped_interactions.select(self._query_id_column, self._item_id_column).map_rows(
                lambda x: (x[0], [get_sequence(x[0], source, x[1]) for source in tensor_feature.feature_sources])
            )
        ).rename({"column_0": self._query_id_column, "column_1": tensor_feature.name})

        if tensor_feature.feature_hint == FeatureHint.TIMESTAMP:
            reshape_size = -1
        else:
            reshape_size = (-1, len(tensor_feature.feature_sources))

        return pl.DataFrame(
            {
                self._query_id_column: result[self._query_id_column].to_list(),
                tensor_feature.name: [
                    np.array(x).reshape(reshape_size).tolist() for x in result[tensor_feature.name].to_list()
                ],
            }
        )

    def _process_num_feature(self, tensor_feature: TensorFeatureInfo) -> List[np.ndarray]:
        """
        Process numerical feature for all sources.

        :param tensor_feature: tensor feature information.

        :returns: sequences for each source for each query.
            If feature came from item features then gets item features values.
            If feature came from interactions then gets values from interactions.
            The results are combined in one sequence array.
        """
        assert tensor_feature.feature_sources is not None
        assert tensor_feature.is_seq

        if self._is_polars:
            return self._process_num_feature_polars(tensor_feature)

        values: List[np.ndarray] = []
        for pos, item_id_sequence in enumerate(self._grouped_interactions[self._item_id_column]):
            all_features_for_user = []
            for source in tensor_feature.feature_sources:
                if source.source == FeatureSource.ITEM_FEATURES:
                    item_feature = self._item_features[source.column]
                    feature_sequence = item_feature.loc[item_id_sequence].values
                    all_features_for_user.append(feature_sequence)
                elif source.source == FeatureSource.INTERACTIONS:
                    sequence = self._grouped_interactions[source.column][pos]
                    all_features_for_user.append(sequence)
                else:
                    assert False, "Unknown tensor feature source table"
            all_seqs = np.array(all_features_for_user, dtype=np.float32)
            if tensor_feature.feature_hint == FeatureHint.TIMESTAMP:
                all_seqs = all_seqs.reshape(-1)
            else:
                all_seqs = all_seqs.reshape(-1, (len(tensor_feature.feature_sources)))
            values.append(all_seqs)
        return values

    def _process_cat_interaction_feature(self, tensor_feature: TensorFeatureInfo) -> List[np.ndarray]:
        """
        Process categorical interaction feature.

        :param tensor_feature: tensor feature information.

        :returns: tensor feature column as a sequences from `grouped_interactions`.
        """
        assert tensor_feature.is_seq

        source = tensor_feature.feature_source
        assert source is not None

        if self._is_polars:
            return self._grouped_interactions.select(self._query_id_column, source.column).rename(
                {source.column: tensor_feature.name}
            )

        return [np.array(sequence, dtype=np.int64) for sequence in self._grouped_interactions[source.column]]

    def _process_cat_query_feature(self, tensor_feature: TensorFeatureInfo) -> List[np.ndarray]:
        """
        Process categorical feature from query features dataset.

        :param tensor_feature: tensor feature information.

        :returns: sequences with length of item sequence for each query for
            sequential features and one size sequences otherwise.
        """
        assert self._query_features is not None

        source = tensor_feature.feature_source
        assert source is not None

        if self._is_polars:
            if tensor_feature.is_seq:
                lengths = self._grouped_interactions.select(
                    self._query_id_column, pl.col(self._item_id_column).list.len().alias("len")
                )
                result = self._query_features.join(lengths, on=self._query_id_column, how="left")
                repeat_value = "len"
            else:
                result = self._query_features
                repeat_value = 1

            return result.select(self._query_id_column, pl.col(source.column).repeat_by(repeat_value)).rename(
                {source.column: tensor_feature.name}
            )

        query_feature = self._query_features[source.column].values
        if tensor_feature.is_seq:
            return [
                np.full(len(item_id_sequence), query_feature[i], dtype=np.int64)
                for i, item_id_sequence in enumerate(self._grouped_interactions[self._item_id_column])
            ]
        return [np.array([query_feature[i]], dtype=np.int64) for i in range(len(self._grouped_interactions))]

    def _process_cat_item_feature(self, tensor_feature: TensorFeatureInfo) -> List[np.ndarray]:
        """
        Process categorical feature from item features dataset.

        :param tensor_feature: tensor feature information.

        :returns: item features as a sequence for each item in a sequence
            for each query.
        """
        assert tensor_feature.is_seq
        assert self._item_features is not None

        source = tensor_feature.feature_source
        assert source is not None

        if self._is_polars:
            return (
                self._grouped_interactions.select(self._query_id_column, self._item_id_column)
                .map_rows(
                    lambda x: (
                        x[0],
                        pl.DataFrame({self._item_id_column: x[1]})
                        .join(self._item_features, on=self._item_id_column, how="left")
                        .select(source.column)
                        .to_numpy()
                        .reshape(-1)
                        .tolist(),
                    )
                )
                .rename({"column_0": self._query_id_column, "column_1": tensor_feature.name})
            )

        item_feature = self._item_features[source.column]
        values: List[np.ndarray] = []

        for item_id_sequence in self._grouped_interactions[self._item_id_column]:
            feature_sequence = item_feature.loc[item_id_sequence].values
            values.append(np.array(feature_sequence, dtype=np.int64))

        return values
