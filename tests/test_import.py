import importlib

import pytest


@pytest.mark.core
@pytest.mark.skipif(importlib.util.find_spec("torch"), reason="PyTorch is installed.")
def test_import_raises_when_torch_missing():
    with pytest.raises(ImportError):
        from replay import nn  # noqa: F401
