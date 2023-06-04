import logging
import os
import pickle
from abc import abstractmethod
from typing import Dict, Optional, List, Any

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from pyspark.ml.base import Model, Estimator, Transformer
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasInputCols, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable, DefaultParamsWriter, DefaultParamsReader, R
from pyspark.sql import DataFrame, SparkSession

from replay.session_handler import State
from replay.utils import (
    convert2spark,
    get_top_k_recs, )


class _ReRankerParams(HasLabelCol, HasPredictionCol, HasInputCols):
    numRecommendations = Param(
        Params._dummy(), "numRecommendations", "Count of recommendations per user.", typeConverter=TypeConverters.toInt
    )

    def getNumRecommendations(self) -> int:
        return self.getOrDefault(self.numRecommendations)

    def setNumRecommendations(self, value: int):
        raise NotImplementedError()


class AutoMLParams(Params):
    automlParams = Param(
        Params._dummy(),
        "automlParams",
        "Dictionary with parameters of automl serialized in string",
        typeConverter=TypeConverters.toString
    )

    configPath = Param(
        Params._dummy(), "configPath", "Path to a configuration file for AutoML", typeConverter=TypeConverters.toString
    )

    def setAutoMLParams(self, value: Dict[str, Any]):
        pickled_automl_params = pickle.dumps(value)
        self.set(self.automlParams, pickled_automl_params)
        return self

    def getAutoMLParams(self) -> Dict[str, Any]:
        pickled_automl_params = self.getOrDefault(self.automlParams)
        return pickle.loads(pickled_automl_params)

    def setConfigPath(self, value: str):
        self.set(self.configPath, value)
        return self

    def getConfigPath(self) -> str:
        return self.getOrDefault(self.configPath)


class PickledAndDefaultParamsWriter(DefaultParamsWriter):
    def __init__(self, instance):
        super(PickledAndDefaultParamsWriter, self).__init__(instance)

    def saveImpl(self, path: str) -> None:
        super().saveImpl(path)

        fields_dict = {
            key: value for key, value in self.instance.__dict__.items()
            if not isinstance(value, (DataFrame, Estimator, Transformer))
        }
        dfs_dict = {
            key: value for key, value in self.instance.__dict__.items()
            if isinstance(value, DataFrame)
        }
        est_tr_dict = {
            key: value for key, value in self.instance.__dict__.items()
            if isinstance(value, (Estimator, Transformer))
        }
        est_tr_metadata = {
            name: ".".join([type(est_or_tr).__module__, type(est_or_tr).__name__])
            for name, est_or_tr in est_tr_dict.items()
        }

        # saving fields of the python instance
        python_fields_dict_df = State().session.createDataFrame([{"data": pickle.dumps(fields_dict)}])
        python_fields_dict_df.write.parquet(os.path.join(path, "python_fields_dict.parquet"))

        # saving info about internal transformers and estimators of the instance
        instance_metadata_df = State().session.createDataFrame([{
            "dfs_metadata": pickle.dumps(list(dfs_dict.keys())), "est_or_tr_metadata": pickle.dumps(est_tr_metadata)
        }])
        instance_metadata_df.write.parquet(os.path.join(path, "instance_metadata.parquet"))

        # saving internal transformers and estimators of the instance
        for name, est_or_tr in est_tr_dict.items():
            est_or_tr.save(os.path.join(path, f"{name}"))

        # saving internal dataframes of the instance
        for name, df in dfs_dict.items():
            df.write.parquet(os.path.join(path, f"{name}.parquet"))


class PickledAndDefaultParamsReader(DefaultParamsReader):
    def __init__(self, cls):
        super(PickledAndDefaultParamsReader, self).__init__(cls)

    def load(self, path: str) -> R:
        instance = super().load(path)

        # reading metadata dataframes
        python_fields_dict_df = State().session.read.parquet(os.path.join(path, "python_fields_dict.parquet"))
        instance_metadata_row = State().session.read.parquet(os.path.join(path, "instance_metadata.parquet")).first()

        # loading metadata
        fields_dict = pickle.loads(python_fields_dict_df.first()["data"])
        dfs_metadata = pickle.loads(instance_metadata_row["dfs_metadata"])
        est_tr_metadata = pickle.loads(instance_metadata_row["est_or_tr_metadata"])

        # setting fields into the instance
        instance.__dict__.update(fields_dict)

        # setting dataframes into the instance
        for name in dfs_metadata:
            df = State().session.read.parquet(os.path.join(path, f"{name}.parquet"))
            instance.__dict__[name] = df

        # setting transformers or estimators into the instance
        for name, clazz in est_tr_metadata.items():
            est_or_tr = DefaultParamsReader.__get_class(clazz).load(os.path.join(path, f"{name}"))
            instance.__dict__[name] = est_or_tr

        return instance


class PickledAndDefaultParamsReadable(DefaultParamsReadable):
    @classmethod
    def read(cls):
        return PickledAndDefaultParamsReader(cls)


class PickledAndDefaultParamsWritable(DefaultParamsWritable):
    def write(self):
        """Returns a DefaultParamsWriter instance for this class."""
        from pyspark.ml.param import Params

        if isinstance(self, Params):
            return PickledAndDefaultParamsWriter(self)
        else:
            raise TypeError("Cannot use PickledAndDefaultParamsWriter with type %s because it does not " +
                            " extend Params.", type(self))


# class ReRanker(AbleToSaveAndLoad):
class ReRanker(Estimator, _ReRankerParams, PickledAndDefaultParamsReadable, PickledAndDefaultParamsWritable):
    """
    Base class for models which re-rank recommendations produced by other models.
    May be used as a part of two-stages recommendation pipeline.
    """

    _logger: Optional[logging.Logger] = None

    def __init__(
            self,
            input_cols: Optional[List[str]] = None,
            label_col: str = "target",
            prediction_col: str = "relevance",
            num_recommendations: int = 10
    ):
        super(ReRanker, self).__init__()
        self.setInputCols(input_cols or [])
        self.setLabelCol(label_col)
        self.setPredictionCol(prediction_col)
        self.setNumRecommendations(num_recommendations)

    def setNumRecommendations(self, value: int):
        self.set(self.numRecommendations, value)
        return self

    @property
    def logger(self) -> logging.Logger:
        """
        :returns: get library logger
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger

    # @abstractmethod
    # def fit(self, data: DataFrame, fit_params: Optional[Dict] = None) -> 'ReRankerModel':
    #     """
    #     Fit the model which re-rank user-item pairs generated outside the models.
    #
    #     :param data: spark dataframe with obligatory ``[user_idx, item_idx, target]``
    #         columns and features' columns
    #     :param fit_params: dict of parameters to pass to model.fit()
    #     """

    # @abstractmethod
    # def predict(self, data, k) -> DataFrame:
    #     """
    #     Re-rank data with the model and get top-k recommendations for each user.
    #
    #     :param data: spark dataframe with obligatory ``[user_idx, item_idx]``
    #         columns and features' columns
    #     :param k: number of recommendations for each user
    #     """


class ReRankerModel(Model, _ReRankerParams, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(
            self,
            input_cols: Optional[List[str]] = None,
            label_col: str = "target",
            prediction_col: str = "relevance",
            num_recommendations: int = 10
    ):
        super(ReRankerModel, self).__init__()
        self.setInputCols(input_cols or [])
        self.setLabelCol(label_col)
        self.setPredictionCol(prediction_col)
        self.setNumRecommendations(num_recommendations)

    def setInputCols(self, value: List[str]):
        self.set(self.inputCols, value)
        return self

    def setLabelCol(self, value: str):
        self.set(self.labelCol, value)
        return self

    def setPredictionCol(self, value: str):
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        self.set(self.predictionCol, value)
        return self

    def setNumRecommendations(self, value: int):
        self.set(self.numRecommendations, value)
        return self

    def predict(self, value):
        """
        Predict label for the given features.
        """
        return self.transform(value)


class LamaWrap(ReRanker):
    """
    LightAutoML TabularPipeline binary classification model wrapper for recommendations re-ranking.
    Read more: https://github.com/sberbank-ai-lab/LightAutoML
    """

    @classmethod
    def load(cls, path: str, spark: Optional[SparkSession] = None):
        spark = spark or cls._get_spark_session()
        row = spark.read.parquet(path).first().asDict()
        model = pickle.loads(row["data"])
        wrap = LamaWrap()
        wrap.model = model
        return wrap

    def save(self, path: str, overwrite: bool = False, spark: Optional[SparkSession] = None):
        spark = spark or self._get_spark_session()
        data = pickle.dumps(self.model)

        spark.createDataFrame([{
            "classname": self.get_classname(),
            "data": data
        }]).write.parquet(path, mode='overwrite' if overwrite else 'error')

    def __init__(
        self,
        params: Optional[Dict] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize LightAutoML TabularPipeline with passed params/configuration file.

        :param params: dict of model parameters
        :param config_path: path to configuration file
        """
        self.model = TabularAutoML(
            task=Task("binary"),
            config_path=config_path,
            **(params if params is not None else {}),
        )

    def fit(self, data: DataFrame, fit_params: Optional[Dict] = None) -> None:
        """
        Fit the LightAutoML TabularPipeline model with binary classification task.
        Data should include negative and positive user-item pairs.

        :param data: spark dataframe with obligatory ``[user_idx, item_idx, target]``
            columns and features' columns. `Target` column should consist of zeros and ones
            as the model is a binary classification model.
        :param fit_params: dict of parameters to pass to model.fit()
            See LightAutoML TabularPipeline fit_predict parameters.
        """

        params = {"roles": {"target": "target"}, "verbose": 1}
        params.update({} if fit_params is None else fit_params)
        data = data.drop("user_idx", "item_idx")
        data_pd = data.toPandas()
        self.model.fit_predict(data_pd, **params)

    def predict(self, data: DataFrame, k: int) -> DataFrame:
        """
        Re-rank data with the model and get top-k recommendations for each user.

        :param data: spark dataframe with obligatory ``[user_idx, item_idx]``
            columns and features' columns
        :param k: number of recommendations for each user
        :return: spark dataframe with top-k recommendations for each user
            the dataframe columns are ``[user_idx, item_idx, relevance]``
        """
        data_pd = data.toPandas()
        candidates_ids = data_pd[["user_idx", "item_idx"]]
        data_pd.drop(columns=["user_idx", "item_idx"], inplace=True)
        self.logger.info("Starting re-ranking")
        candidates_pred = self.model.predict(data_pd)
        candidates_ids.loc[:, "relevance"] = candidates_pred.data[:, 0]
        self.logger.info(
            "%s candidates rated for %s users",
            candidates_ids.shape[0],
            candidates_ids["user_idx"].nunique(),
        )

        self.logger.info("top-k")
        return get_top_k_recs(
            recs=convert2spark(candidates_ids), k=k, id_type="idx"
        )
