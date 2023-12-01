from abc import abstractmethod

import numpy as np
from scipy.stats import norm, sem

from replay.utils import PYSPARK_AVAILABLE, SparkDataFrame

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf


class CalculationDescriptor:
    """
    Abstract class for metric aggregation using given distribution by users
    """

    @property
    def __name__(self) -> str:
        return str(self.__class__.__name__)

    @abstractmethod
    def spark(self, distribution: SparkDataFrame):
        """
        Calculation on PySpark
        """

    @abstractmethod
    def cpu(self, distribution: np.array):
        """
        Calculation on cpu
        """


class Mean(CalculationDescriptor):
    """
    Calculation of the average value for a given distribution by users
    """

    def spark(self, distribution: SparkDataFrame):
        column_name = distribution.columns[0]
        return distribution.select(sf.avg(column_name)).first()[0]

    def cpu(self, distribution: np.array):
        return np.mean(distribution)


class PerUser(CalculationDescriptor):
    """
    Returns the distribution as is, for each user
    """

    def spark(self, distribution: SparkDataFrame):
        return distribution

    def cpu(self, distribution: np.array):
        return distribution


class Median(CalculationDescriptor):
    """
    Calculation of the medium value for a given distribution by users
    """

    def spark(self, distribution: SparkDataFrame):
        column_name = distribution.columns[0]
        return distribution.select(
            sf.expr(f"percentile_approx({column_name}, 0.5)")
        ).first()[0]

    def cpu(self, distribution: np.array):
        return np.median(distribution)


class ConfidenceInterval(CalculationDescriptor):
    """
    Calculating the average value of the confidence interval for a given distribution by users.
    To obtain the left boundary of the interval,
    it is necessary to subtract the result of this class from the average value.
    For the right border - add the average with the result of this class.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def spark(self, distribution: SparkDataFrame):
        column_name = distribution.columns[0]
        quantile = norm.ppf((1 + self.alpha) / 2)
        value = (
            distribution.agg(
                sf.stddev(column_name).alias("std"),
                sf.count(column_name).alias("count"),
            )
            .select(
                sf.when(
                    sf.isnan(sf.col("std")) | sf.col("std").isNull(),
                    sf.lit(0.0),
                )
                .otherwise(sf.col("std"))
                .cast("float")
                .alias("std"),
                "count",
            )
            .first()
        )
        return quantile * value["std"] / (value["count"] ** 0.5)

    def cpu(self, distribution: np.array):
        quantile = norm.ppf((1 + self.alpha) / 2)
        return quantile * sem(distribution)
