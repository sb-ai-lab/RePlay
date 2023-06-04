import os
import pickle

from pyspark.ml import Estimator, Transformer
from pyspark.ml.util import DefaultParamsWriter, DefaultParamsReader, R, DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame

from replay.session_handler import State
from replay.utils import get_full_class_name, get_class_by_class_name


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
            name: get_full_class_name()
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
            est_or_tr = get_class_by_class_name(clazz).load(os.path.join(path, f"{name}"))
            instance.__dict__[name] = est_or_tr

        return instance


class PickledAndDefaultParamsReadable(DefaultParamsReadable):
    @classmethod
    def read(cls):
        return PickledAndDefaultParamsReader(cls)


class PickledAndDefaultParamsWritable(DefaultParamsWritable):
    def write(self):
        """Returns a PickledAndDefaultParamsWriter instance for this class."""
        from pyspark.ml.param import Params

        if isinstance(self, Params):
            return PickledAndDefaultParamsWriter(self)
        else:
            raise TypeError("Cannot use PickledAndDefaultParamsWriter with type %s because it does not " +
                            " extend Params.", type(self))