import builtins

import pytest


@pytest.mark.core
def test_import_nn_raises_if_torch_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, args, *kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError
        return real_import(name, args, *kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        from replay import nn  # noqa: F401
