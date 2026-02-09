"""Tests for target generation and sampling.

Uses demo.helpers.create_mechanism_from_dict so extract_dimensions() is unpacked
correctly as (DimensionBoundsSpec, set); a tuple would cause 'tuple' has no attribute 'names'.
"""
from __future__ import annotations

import numpy as np
import pytest

from demo.helpers import create_mechanism_from_dict
from demo.helpers import load_fourbar_data
from target_gen.achievable_target import create_achievable_target
from target_gen.sampling import generate_good_samples
from target_gen.sampling import generate_samples
from target_gen.sampling import generate_valid_samples


@pytest.fixture
def fourbar_data():
    return load_fourbar_data()


@pytest.fixture
def mechanism(fourbar_data):
    return create_mechanism_from_dict(fourbar_data)


@pytest.fixture
def dim_spec(mechanism):
    return mechanism.get_dimension_bounds_spec()


def test_create_achievable_target(mechanism):
    result = create_achievable_target(mechanism, 'coupler_rocker_joint', dim_spec=None, config=None)
    assert result is not None and result.target is not None
    assert result.target.joint_name == 'coupler_rocker_joint'
    assert len(result.target.positions) > 0
    assert result.target_dimensions is not None and result.target_mechanism is not None
    assert result.attempts_needed > 0
    for pos in result.target.positions:
        assert len(pos) == 2 and all(isinstance(c, (int, float)) for c in pos)
    traj = result.target_mechanism.simulate()
    assert traj is not None and traj.shape[0] == mechanism.n_steps


def test_generate_samples(mechanism, dim_spec):
    result = generate_samples(mechanism, dimension_bounds_spec=dim_spec, n_requested=16, sampling_mode='sobol', seed=42)
    assert len(result.samples) == 16 and result.n_generated == 16
    assert result.n_valid + result.n_invalid == 16
    assert result.scores is None
    assert result.samples.shape[1] == len(dim_spec.names) and len(result.is_valid) == 16


def test_generate_samples_with_target(mechanism, dim_spec):
    target_res = create_achievable_target(mechanism, 'coupler_rocker_joint', dim_spec=dim_spec, config=None)
    result = generate_samples(
        mechanism, dimension_bounds_spec=dim_spec, n_requested=16, sampling_mode='sobol',
        target_trajectory=target_res.target, target_joint='coupler_rocker_joint', seed=42,
    )
    assert len(result.samples) == 16 and result.scores is not None and len(result.scores) == 16
    # Scores may include inf/nan for invalid or poor fits; structure is what we assert


def test_generate_valid_samples(mechanism, dim_spec):
    result = generate_valid_samples(
        mechanism, dimension_bounds_spec=dim_spec, n_valid_requested=8, max_attempts=1000,
        sampling_mode='sobol', seed=42,
    )
    assert len(result.samples) == 8 and result.n_valid == 8 and result.n_invalid == 0
    assert all(result.is_valid) and result.samples.shape[1] == len(dim_spec.names)


def test_generate_good_samples(mechanism, dim_spec):
    target_res = create_achievable_target(mechanism, 'coupler_rocker_joint', dim_spec=dim_spec, config=None)
    result = generate_good_samples(
        mechanism, dimension_bounds_spec=dim_spec, target_trajectory=target_res.target,
        n_good_requested=8, epsilon=1000.0, max_attempts=2000, sampling_mode='sobol',
        target_joint='coupler_rocker_joint', seed=42,
    )
    if len(result.samples) > 0:
        assert result.n_valid == len(result.samples) and result.scores is not None
        assert all(result.scores <= 1000.0) and all(result.is_valid)


def test_generate_samples_modes(mechanism, dim_spec):
    for mode in ['sobol', 'behnken']:
        result = generate_samples(
            mechanism, dimension_bounds_spec=dim_spec, n_requested=8, sampling_mode=mode, seed=42,
        )
        assert len(result.samples) == 8 and result.samples.shape[1] == len(dim_spec.names)
