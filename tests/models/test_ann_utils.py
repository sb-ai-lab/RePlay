import numpy as np
import pytest

from replay.models.extensions.ann.index_inferers.utils import get_csr_matrix

pyspark = pytest.importorskip("pyspark")

import pyspark.sql.functions as sf


@pytest.mark.spark
def test_get_csr_matrix(log2):
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

    expected_array = np.array([[3.0, 1.0, 2.0, 0.0], [3.0, 0.0, 0.0, 4.0], [0.0, 3.0, 0.0, 0.0]])

    assert np.array_equal(actual_array, expected_array)
