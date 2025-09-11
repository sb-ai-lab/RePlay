import re

import pytest


@pytest.mark.torch
def test_fail_on_importing_compiled_models_without_deps():
    pattern = re.escape("OpenVINO prerequisites not found")
    with pytest.raises(ImportError, match=pattern):
        import replay.models.nn.sequential.compiled  # noqa: F401
