from abc import ABC

from pyspark.ml import Estimator, Model
from pyspark.ml.util import MLWritable, MLWriter, MLReadable, MLReader, R


class SparkRecModelWriter(MLWriter):
    def saveImpl(self, path: str) -> None:
        raise NotImplementedError()


class SparkRecModelReader(MLReader):
    def load(self, path: str) -> R:
        raise NotImplementedError()


class SparkRecModelWritable(MLWritable):
    def write(self) -> MLWriter:
        return SparkRecModelWriter()


class SparkRecModelReadable(MLReadable):
    @classmethod
    def read(cls) -> SparkRecModelReader:
        return SparkRecModelReader()


class SparkBaseRec(Estimator, ABC):
    ...


class SparkBaseRecModel(Model, SparkRecModelReadable, SparkRecModelWritable, ABC):
    ...
