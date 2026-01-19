import pytest
import torch

from replay.nn.utils import create_activation


@pytest.mark.parametrize(
    "activation_name, expected_type", [("gelu", torch.nn.GELU), ("relu", torch.nn.ReLU), ("sigmoid", torch.nn.Sigmoid)]
)
def test_create_activation(activation_name, expected_type):
    activation = create_activation(activation_name)
    assert isinstance(activation, expected_type)


def test_create_activation_wrong_name():
    with pytest.raises(ValueError):
        create_activation("wrong-name")
