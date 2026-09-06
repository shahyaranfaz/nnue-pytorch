import pytest

from config import TrainingConfig, distributed_batch_sizes


def test_four_node_global_batch_is_split_across_world():
    assert distributed_batch_sizes(16384, 1, 4, 4) == (4, 4096)


def test_torchrun_world_size_must_match_declared_layout():
    with pytest.raises(ValueError, match="WORLD_SIZE=1 does not match"):
        distributed_batch_sizes(16384, 1, 4, 1)


def test_global_batch_must_divide_world_size():
    with pytest.raises(ValueError, match="must be divisible"):
        distributed_batch_sizes(16386, 1, 4, 4)


def test_num_nodes_must_be_positive():
    with pytest.raises(ValueError, match="num_nodes"):
        TrainingConfig(datasets=("data.binpack",), num_nodes=0)
