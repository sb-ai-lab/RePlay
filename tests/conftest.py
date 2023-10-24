import pytest

from replay.utils import get_spark_session


@pytest.fixture(scope="session")
def spark_session():
    return get_spark_session(1, 1)
