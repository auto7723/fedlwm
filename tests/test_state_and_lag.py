import numpy as np

from fedlwm.dynamics import StructuredDynamics
from fedlwm.fixed_lag import FixedLagInformationSmoother
from fedlwm.state import WorldState
from fedlwm.statistics import InformationIncrement


def test_state_projection_preserves_constraints():
    state = WorldState.initialize(3, 2, 2, 2, 1e-3)
    state.mass[0] = [-1.0, 4.0]
    state.covariance[0, 0] = [[1.0, 3.0], [3.0, 1.0]]
    state.offsets[0, 0] = [[2.0, 1.0], [0.0, 0.0]]
    state.project(1e-3)
    np.testing.assert_allclose(state.mass.sum(axis=1), 1.0)
    assert np.linalg.eigvalsh(state.covariance[0, 0]).min() >= 1e-3 - 1e-9
    np.testing.assert_allclose(state.offsets[0, 0].sum(axis=0), 0.0)


def test_delayed_increment_replays_semantic_slice():
    state = WorldState.initialize(2, 1, 2, 2, 1e-3)
    dynamics = StructuredDynamics.identity(state, 0.9, 1e-3)
    smoother = FixedLagInformationSmoother(state, dynamics, lag=3, floor=1e-3, out_of_window_decay=0.5)
    smoother.advance_to(2)
    prior = smoother.slices[1].prior.block_vector(0, 0)
    precision = np.eye(len(prior)) * 2.0
    target = prior.copy()
    target[0] = 1.0
    emission = state.emission_covariance[0, 0].copy()
    increment = InformationIncrement(
        0, 0, 1, 2, precision, precision @ (target - prior), prior.copy(), emission, 2.0,
    )
    smoother.assimilate([increment])
    assert smoother.slices[1].posterior.mean[0, 0, 0] > 0.5
    assert smoother.current.mean[0, 0, 0] > 0.5
