# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import numpy as np
import pyspark.sql.functions as sf

from replay.models.extensions.ann.index_inferers.utils import get_csr_matrix
from tests.utils import log2, spark


def test_get_csr_matrix(spark, log2):
    grouped_log = log2.groupBy("user_idx").agg(
        sf.collect_list("item_idx").alias("vector_items"),
        sf.collect_list("relevance").alias("vector_ratings"),
    )

    grouped_log = grouped_log.toPandas()

    csr_matrix = get_csr_matrix(
        grouped_log["user_idx"],
        grouped_log["vector_items"],
        grouped_log["vector_ratings"],
    )

    actual_array = csr_matrix.toarray()

    expected_array = np.array(
        [[3.0, 1.0, 2.0, 0.0], [3.0, 0.0, 0.0, 4.0], [0.0, 3.0, 0.0, 0.0]]
    )

    assert np.array_equal(actual_array, expected_array)
