import pytest


@pytest.mark.core
def test_import_nn_module():
    with pytest.raises(ImportError):
        from replay import nn