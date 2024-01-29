# pylint: disable-all
import pytest

from replay.models.extensions.ann.entities.hnswlib_param import HnswlibParam
from replay.models.extensions.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.models.extensions.ann.index_builders.driver_hnswlib_index_builder import DriverHnswlibIndexBuilder
from replay.models.extensions.ann.index_builders.executor_hnswlib_index_builder import ExecutorHnswlibIndexBuilder
from replay.models.extensions.ann.index_builders.executor_nmslib_index_builder import ExecutorNmslibIndexBuilder
from replay.models.extensions.ann.index_stores.shared_disk_index_store import SharedDiskIndexStore
from replay.models.extensions.ann.index_inferers.hnswlib_index_inferer import HnswlibIndexInferer
from replay.models.extensions.ann.index_inferers.hnswlib_filter_index_inferer import HnswlibFilterIndexInferer
from replay.models.extensions.ann.index_inferers.nmslib_index_inferer import NmslibIndexInferer
from replay.models.extensions.ann.index_inferers.nmslib_filter_index_inferer import NmslibFilterIndexInferer
from replay.utils import PYSPARK_AVAILABLE
from tests.utils import spark


if PYSPARK_AVAILABLE:
    from replay.models.extensions.ann.index_stores.spark_files_index_store import SparkFilesIndexStore


@pytest.fixture
def vectors(spark):
    return spark.createDataFrame(
        data=[
            [0, [0.1, 0.1, 0.1, 0.1]],
            [1, [0.2, 0.1, 0.1, 0.1]],
            [2, [0.1, 0.3, 0.1, 0.1]],
            [3, [0.1, 0.1, 0.4, 0.1]],
            [4, [0.1, 0.1, 0.1, 0.5]],
        ],
        schema=["item_idx", "features"],
    )


@pytest.fixture
def hnsw_driver():
    return DriverHnswlibIndexBuilder(
        index_params=HnswlibParam(
            space="ip",
            m=100,
            ef_c=2000,
            post=0,
            ef_s=2000,
        ),
        index_store=SparkFilesIndexStore()
    )


@pytest.fixture
def hnsw_executor(tmp_path):
    return ExecutorHnswlibIndexBuilder(
        index_params=HnswlibParam(
            space="ip",
            m=100,
            ef_c=2000,
            post=0,
            ef_s=2000,
        ),
        index_store=SharedDiskIndexStore(
            warehouse_dir=str(tmp_path), index_dir="hnswlib_index"
        ),
    )


@pytest.fixture
def nms_executor(tmp_path):
    return ExecutorNmslibIndexBuilder(
        index_params=NmslibHnswParam(
            space="negdotprod_sparse",
            m=10,
            ef_s=200,
            ef_c=200,
            post=0,
        ),
        index_store=SharedDiskIndexStore(
            warehouse_dir=str(tmp_path), index_dir="nmslib_hnsw_index"
        ),
    )


@pytest.mark.spark
@pytest.mark.parametrize(
    "driver", ["hnsw_driver", "hnsw_executor"]
)
def test_hnsw_index_builder(driver, vectors, request):
    driver = request.getfixturevalue(driver)
    inferer = driver.produce_inferer(filter_seen_items=False)
    assert isinstance(inferer, HnswlibIndexInferer)

    inferer = driver.produce_inferer(filter_seen_items=True)
    assert isinstance(inferer, HnswlibFilterIndexInferer)

    driver.index_params.dim = 4
    driver.index_params.max_elements = 5

    driver.build_index(vectors, features_col="features", ids_col="item_idx")
    driver.build_index(vectors, features_col="features", ids_col=None)


@pytest.mark.spark
def test_nms_index_builder(nms_executor):
    inferer = nms_executor.produce_inferer(filter_seen_items=False)
    assert isinstance(inferer, NmslibIndexInferer)

    inferer = nms_executor.produce_inferer(filter_seen_items=True)
    assert isinstance(inferer, NmslibFilterIndexInferer)
