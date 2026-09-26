from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import StructuredDynamics
from .state import WorldState
from .statistics import InformationIncrement


@dataclass
class Slice:
    time: int
    prior: WorldState
    posterior: WorldState
    increments: list[InformationIncrement] = field(default_factory=list)


class FixedLagInformationSmoother:
    """Block-Laplace realization of L1 with bounded delayed replay."""

    def __init__(
        self, initial: WorldState, dynamics: StructuredDynamics, lag: int,
        floor: float, out_of_window_decay: float,
    ):
        self.dynamics = dynamics
        self.lag = lag
        self.floor = floor
        self.out_of_window_decay = out_of_window_decay
        self.slices: dict[int, Slice] = {0: Slice(0, initial.copy(), initial.copy())}
        self.current_time = 0
        self.folded_updates = 0

    @property
    def current(self) -> WorldState:
        return self.slices[self.current_time].posterior

    def advance_to(self, target_time: int) -> None:
        while self.current_time < target_time:
            previous = self.slices[self.current_time].posterior
            self.current_time += 1
            prior = self.dynamics.step(previous)
            self.slices[self.current_time] = Slice(self.current_time, prior, prior.copy())
            self._schur_marginalize_oldest()

    def assimilate(self, increments: list[InformationIncrement]) -> None:
        if not increments:
            return
        earliest = self.current_time
        oldest = max(0, self.current_time - self.lag)
        for increment in increments:
            target = increment.semantic_time
            attenuation = 1.0
            if target < oldest:
                attenuation = self.out_of_window_decay ** (oldest - target)
                target = oldest
                self.folded_updates += 1
            if target > self.current_time:
                self.advance_to(target)
            adjusted = InformationIncrement(
                increment.cls, increment.primitive, target, increment.arrival_time,
                increment.precision * attenuation, increment.score * attenuation,
                increment.recorded_prior,
                increment.emission_covariance,
                increment.evidence_count * attenuation,
            )
            self.slices[target].increments.append(adjusted)
            earliest = min(earliest, target)
        self._replay_from(earliest)

    def _solve_slice(self, slice_: Slice) -> WorldState:
        posterior = slice_.prior.copy()
        grouped: dict[tuple[int, int], list[InformationIncrement]] = {}
        for increment in slice_.increments:
            grouped.setdefault((increment.cls, increment.primitive), []).append(increment)
        for (cls, primitive), values in grouped.items():
            prior = slice_.prior.block_vector(cls, primitive)
            size = len(prior)
            prior_precision = np.eye(size) / max(slice_.prior.uncertainty[cls, primitive], self.floor)
            precision = prior_precision.copy()
            information = prior_precision @ prior
            for increment in values:
                precision += increment.precision
                information += increment.precision @ increment.recorded_prior + increment.score
            solved = np.linalg.solve(precision + self.floor * np.eye(size), information)
            posterior.set_block_vector(cls, primitive, solved, self.floor)
            posterior.uncertainty[cls, primitive] = float(np.trace(np.linalg.pinv(precision)) / size)
            emission_values = [value for value in values if value.emission_covariance is not None]
            if emission_values:
                evidence = sum(value.evidence_count for value in emission_values)
                posterior.emission_covariance[cls, primitive] = (
                    slice_.prior.emission_covariance[cls, primitive]
                    + sum(value.evidence_count * value.emission_covariance for value in emission_values)
                ) / (1.0 + evidence)
        return posterior.project(self.floor)

    def _replay_from(self, start: int) -> None:
        for time in range(start, self.current_time + 1):
            slice_ = self.slices[time]
            if time > start:
                slice_.prior = self.dynamics.step(self.slices[time - 1].posterior)
            slice_.posterior = self._solve_slice(slice_)

    def _schur_marginalize_oldest(self) -> None:
        """Eliminate expired slices after their posterior is propagated.

        Under the class/primitive block Gaussian approximation, propagating the
        oldest posterior into the retained boundary prior and then dropping its
        factor is the recursive Schur complement of that block.
        """
        oldest = self.current_time - self.lag
        for time in list(self.slices):
            if time < oldest:
                del self.slices[time]
