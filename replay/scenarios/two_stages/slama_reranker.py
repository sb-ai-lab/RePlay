import logging
import pickle
from typing import Optional, Dict, Any, List

import mlflow
from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.functions import expr
from pyspark.sql import functions as sf

from sparklightautoml.automl.presets.tabular_presets import SparkTabularAutoML
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import WrappingSelectingPipelineModel
from sparklightautoml.tasks.base import SparkTask
from pyspark.sql.types import TimestampType, DoubleType, NumericType, DateType, ArrayType, StringType

from replay.scenarios.two_stages.reranker import ReRanker, ReRankerModel, AutoMLParams
from replay.session_handler import State
from replay.utils import get_top_k_recs, log_exec_timer, JobGroup, JobGroupWithMetrics, \
    cache_and_materialize_if_in_debug

import pandas as pd
import numpy as np


logger = logging.getLogger("replay")


def _handle_columns(df: DataFrame, convert_target: bool = False) -> DataFrame:
    def explode_vec(col_name: str, size: int):
        return [sf.col(col_name).getItem(i).alias(f'{col_name}_{i}') for i in range(size)]

    supported_types = (NumericType, TimestampType, DateType, StringType)

    wrong_type_fields = [
        field for field in df.schema.fields
        if not (isinstance(field.dataType, supported_types)
                or (isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType,
                                                                         NumericType)))
    ]
    assert len(
        wrong_type_fields) == 0, f"Fields with wrong types have been found: {wrong_type_fields}. " \
                                 "Only the following types are supported: {supported_types} " \
                                 "and ArrayType with Numeric type of elements"

    array_fields = [field.name for field in df.schema.fields if isinstance(field.dataType, ArrayType)]

    arrays_to_explode = {
        field.name: df.where(sf.col(field.name).isNotNull()).select(sf.size(field.name).alias("size")).first()[
            "size"]
        for field in df.schema.fields if isinstance(field.dataType, ArrayType)
    }

    timestamp_fields = [field.name for field in df.schema.fields if
                        isinstance(field.dataType, TimestampType)]

    if convert_target:
        additional_columns = [sf.col('target').astype('int').alias('target')]
    else:
        additional_columns = []

    df = (
        df
        .select(
            *(c for c in df.columns if c not in timestamp_fields + array_fields + ['target']),
            *(sf.col(c).astype('int').alias(c) for c in timestamp_fields),
            *additional_columns,
            *(c for f, size in arrays_to_explode.items() for c in explode_vec(f, size))
        )
    )

    return df


class SlamaWrap(ReRanker, AutoMLParams):
    """
    LightAutoML TabularPipeline binary classification model wrapper for recommendations re-ranking.
    Read more: https://github.com/sberbank-ai-lab/LightAutoML
    """
    def __init__(
            self,
            automl_params: Optional[Dict] = None,
            config_path: Optional[str] = None,
            input_cols: Optional[List[str]] = None,
            label_col: str = "target",
            prediction_col: str = "relevance",
            num_recommendations: int = 10
    ):
        """
        Initialize LightAutoML TabularPipeline with passed params/configuration file.

        :param automl_params: dict of model parameters
        :param config_path: path to configuration file
        """
        super(SlamaWrap, self).__init__(input_cols, label_col, prediction_col, num_recommendations)

        self.setAutoMLParams(automl_params)
        self.setConfigPath(config_path)

    def _fit(self, dataset: DataFrame):
        """
        Fit the LightAutoML TabularPipeline model with binary classification task.
        Data should include negative and positive user-item pairs.

        :param data: spark dataframe with obligatory ``[user_idx, item_idx, target]``
            columns and features' columns. `Target` column should consist of zeros and ones
            as the model is a binary classification model.
        :param params: dict of parameters to pass to model.fit()
            See LightAutoML TabularPipeline fit_predict parameters.
        """

        if self.transformer is not None:
            raise RuntimeError("The ranker is already fitted")

        data = dataset.drop("user_idx", "item_idx")

        data = self.handle_columns(data, convert_target=True)

        roles = {
            "target": "target",
            "numeric": [field.name for field in data.schema.fields if
                        isinstance(field.dataType, NumericType) and field.name != 'target'],
        }

        automl_params = {
            "roles": roles,
            "verbose": 1,
            **({} if self.getAutoMLParams() is None else self.getAutoMLParams())
        }

        # this part is required to cut the plan of the dataframe because it may be huge
        temp_checkpoint = f"/tmp/{type(self.model).__name__}_transform.parquet"
        data.write.mode("overwrite").parquet(temp_checkpoint)
        data = SparkSession.getActiveSession().read.parquet(temp_checkpoint).cache()
        data.write.mode('overwrite').format('noop').save()

        model = SparkTabularAutoML(
            spark=State().session,
            task=SparkTask("binary"),
            config_path=self.getConfigPath(),
            **automl_params,
        )
        model.fit_predict(data, **automl_params)

        data.unpersist()

        return SlamaWrapModel(
            input_cols=self.getInputCols(),
            label_col=self.getLabelCol(),
            prediction_col=self.getPredictionCol(),
            num_recommendations=self.getNumRecommendations(),
            automl_transformer=model.transformer()
        )


class SlamaWrapModel(ReRankerModel):
    def __init__(
            self,
            input_cols: Optional[List[str]] = None,
            label_col: str = "target",
            prediction_col: str = "relevance",
            num_recommendations: int = 10,
            automl_transformer: Optional[Transformer] = None
    ):
        super(SlamaWrapModel, self).__init__(input_cols, label_col, prediction_col, num_recommendations)
        self._automl_transformer = automl_transformer

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Re-rank data with the model and get top-k recommendations for each user.

        :param data: spark dataframe with obligatory ``[user_idx, item_idx]``
            columns and features' columns
        :param k: number of recommendations for each user
        :return: spark dataframe with top-k recommendations for each user
            the dataframe columns are ``[user_idx, item_idx, relevance]``
        """
        self.logger.info("Starting re-ranking")

        logger.info(f"transformer type: {str(type(self._automl_transformer))}")

        data = _handle_columns(dataset)

        data.write.mode("overwrite").parquet(f"/tmp/{type(self.model).__name__}_transform.parquet")
        data = SparkSession.getActiveSession().read.parquet(f"/tmp/{type(self.model).__name__}_transform.parquet").cache()
        data.write.mode('overwrite').format('noop').save()

        model_name = type(self.model).__name__

        with JobGroupWithMetrics("slama_predict", f"{model_name}.infer_sec"):
            sdf = self._automl_transformer.transform(data)
            logger.info(f"sdf.columns: {sdf.columns}")
            data.unpersist()

            candidates_pred_sdf = sdf.select(
                'user_idx',
                'item_idx',
                vector_to_array('prediction').getItem(1).alias(self.getPredictionCol())
            )

            self.logger.info("Re-ranking is finished")

            # TODO: strange, but the further process would hang without maetrialization
            # probably, it may be related to optimization and lightgbm models
            # need to dig deeper later
            candidates_pred_sdf = candidates_pred_sdf.cache()
            candidates_pred_sdf.write.mode('overwrite').format('noop').save()

        with JobGroupWithMetrics("slama_predict", "top_k_recs_sec"):
            self.logger.info("top-k")
            top_k_recs = get_top_k_recs(
                recs=candidates_pred_sdf, k=self.getNumRecommendations(), id_type="idx"
            )
            cache_and_materialize_if_in_debug(top_k_recs, "slama_predict_top_k_recs_sec")

        return top_k_recs
