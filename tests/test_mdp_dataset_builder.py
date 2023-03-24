# pylint: disable-all
import numpy as np
from pytest import approx
from pyspark.sql import DataFrame

from replay.constants import LOG_SCHEMA
from replay.mdp_dataset_builder import MdpDatasetBuilder
from tests.utils import spark, log


def test_mdp_dataset_builder(log: DataFrame):
    """Test MDP dataset preparation is correct."""
    mdp_dataset = MdpDatasetBuilder(top_k=1, action_randomization_scale=1e-9).build(log)

    # we test only users {0, 1} as for the rest log has non-deterministic order (see log dates)
    gt_observations = np.array([
        [0, 0], [0, 2], [0, 1],
        [1, 3], [1, 0],
    ])
    gt_actions = np.array([
        4, 3, 2,
        3, 4
    ])
    gt_rewards = np.array([
        1, 0, 0,
        0, 1,
    ])
    gt_terminals = np.array([
        0, 0, 1,
        0, 1,
    ])
    n = 5

    # as we do not guarantee and require that MDP preparation should keep ints as ints
    # and keeps floats exactly the same, we use approx for all equality checks
    assert mdp_dataset.observations[:n] == approx(gt_observations)
    # larger approx to take into account action randomization noise added to the MDP actions
    assert mdp_dataset.actions[:n].flatten() == approx(gt_actions, abs=1e-2)
    assert mdp_dataset.rewards[:n] == approx(gt_rewards)
    assert mdp_dataset.terminals[:n] == approx(gt_terminals)
