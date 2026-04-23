import sys
import types

import pytest

from replay.utils import spark_compat


@pytest.mark.core
def test_get_spark_major_version_without_pyspark(monkeypatch):
    monkeypatch.setattr(spark_compat, "PYSPARK_AVAILABLE", False)
    assert spark_compat.get_spark_major_version() is None


@pytest.mark.core
def test_get_spark_major_version_with_unparsable_version(monkeypatch):
    fake_pyspark = types.ModuleType("pyspark")
    fake_pyspark.__version__ = "dev"
    monkeypatch.setitem(sys.modules, "pyspark", fake_pyspark)
    monkeypatch.setattr(spark_compat, "PYSPARK_AVAILABLE", True)

    assert spark_compat.get_spark_major_version() is None
