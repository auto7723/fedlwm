# FedLWM

Anonymous reference implementation of **Federated Latent World Modeling**.
This repository contains the method-level client/server protocol and accepts
only externally prepared, anonymized feature bundles. Raw media, identity
maps, checkpoints, experiment outputs, and author metadata are not included.

## Method implementation

The source follows the three components in the paper.

1. **Hyperspherical semantic reference frame (A).**
   `fedlwm/geometry.py` constructs fixed semantic fiducials and deterministic
   tangent bases. They are non-trainable buffers and are never aggregated as
   client coordinates. `fedlwm/model.py` implements normalized modality
   observations, the frame objective, and class-chart logarithmic maps.
2. **Structured latent world state (B).**
   `fedlwm/state.py` stores primitive mass, means, covariance, centered
   modality offsets, and joint emission covariance. Clients form compact
   sufficient statistics in `fedlwm/statistics.py`; the server reconstructs
   score and positive-semidefinite information increments at the recorded
   prior. Client primitive locations are never averaged. Missing-modality
   inference uses the observed marginal of the joint emission model.
3. **Temporal assimilation and imagination (C).**
   `fedlwm/fixed_lag.py` inserts delayed evidence at its semantic-time slice,
   folds out-of-window information into the lag boundary, and solves the
   structured information system. `fedlwm/dynamics.py` propagates primitive
   means, covariance, mixture mass, and centered modality offsets with
   structure-preserving maps. `fedlwm/server.py` broadcasts a use-time
   semantic ephemeris, which `fedlwm/client.py` consumes through the
   uncertainty-modulated semantic regularizer.

The neural parameter channel is independent of the world channel. The server
supports no parameter exchange, synchronous FedAvg, event-wise FedAsync, and
buffered FedAdam. Personal models remain local; only the configured shared
scope is eligible for parameter exchange. The world-only path therefore
supports heterogeneous private observer shapes without requiring
parameter-space compatibility across clients.

## Numerical scope

The fixed-lag Laplace approximation is represented by class/primitive
information blocks rather than a dense trajectory matrix. Transition,
emission, and frame-calibration parameters are re-estimated on configured
anchor rounds, while latent-state assimilation and prediction run throughout
the service trace. State projection enforces covariance floors, feasible
mixture mass, bounded chart coordinates, and centered modality offsets after
prediction.

The implementation is a transparent numerical realization of the algorithms
in the paper. It keeps the parameter and world clocks separate, records the
prior used to construct each upload, rejects mismatched priors, and never uses
query labels to fit world state or select a readout.

## Data boundary

This release operates on externally prepared anonymized feature bundles,
separating dataset-specific raw-media preprocessing from the method-level
federated protocol. The loader accepts numerical feature files plus anonymized
JSON metadata containing opaque sample/client identifiers, split, label,
modality mask, semantic time, arrival time, and use time. It rejects names,
email addresses, absolute paths, path traversal, and raw-media path fields.

Development and test partitions are explicit. Configuration selection must be
completed on development splits before test evaluation. E-G2 uses a fresh
receiver and a disjoint support/query split; query labels are used only for
metrics.

## Installation and verification

Python 3.8 or newer is required.

```bash
python -m pip install -e .
python -m pytest
fedlwm audit-release --root .
```

The tests use synthetic numerical fixtures only. They verify fixed-frame
geometry, state constraints, timestamped delayed assimilation, world-only and
parameter-channel execution, the four-view evaluation path, privacy-oriented
input validation, and release anonymity.

## External execution contract

Training requires two artifacts supplied separately by the data controller:

- an experiment JSON compatible with the dataclasses in `fedlwm/config.py`;
- an anonymized feature bundle accepted by `fedlwm/data.py`.

The command-line entry point is:

```bash
fedlwm validate-data --config CONFIG.json --data-root FEATURE_BUNDLE
fedlwm train --config CONFIG.json --data-root FEATURE_BUNDLE \
  --output OUTPUT_DIR --evaluation-suffix dev --device auto
fedlwm summarize --root OUTPUT_DIR
```

Formal evaluation must replace `dev` with `test` only after the configuration
is frozen. The generated result records configuration and data-contract hashes,
online and drained metrics separately, and per-view E-P, E-G1, E-G1-prime, and
E-G2 measurements.
