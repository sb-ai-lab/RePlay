
from replay.data.nn.parquet.info.distributed_info import DEFAULT_DISTRIBUTED_INFO as dist_info


def test_non_distributed(mocker) -> None:
    mocker.patch("torch.distributed.is_available", return_value=False)

    assert not dist_info.is_distributed
    assert tuple(dist_info) == (0, 1)


def test_distributed(mocker):
    mocker.patch("torch.distributed.is_available", return_value=True)
    mocker.patch("torch.distributed.is_initialized", return_value=True)
    mocker.patch("torch.distributed.get_rank", return_value=1)
    mocker.patch("torch.distributed.get_world_size", return_value=4)

    assert dist_info.is_distributed
    assert tuple(dist_info) == (1, 4)
