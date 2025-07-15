"""
test_rcd_model.py

Basic unit tests for RCDModel integrity and metric validity.
"""

import numpy as np
from rcd_model import RCDModel

def test_initialization():
    model = RCDModel()
    model.set_parameters(0.6, 0.2, 0.1, 3, 0.05)
    model.initialize_manifolds(seed=123)
    assert model.H.shape == (3, 3)
    assert model.M.shape == (3, 3)
    assert 0 <= model.R <= 1


def test_metrics_valid():
    model = RCDModel()
    model.initialize_manifolds(seed=42)
    gamma = model.compute_phase_synchronization(model.H, model.M)
    rho = model.compute_semantic_correlation(model.H, model.M)
    d = model.compute_procrustes_distance(model.H, model.M)
    assert 0 <= gamma <= 1
    assert -1 <= rho <= 1
    assert d >= 0


def test_simulation_runs():
    model = RCDModel()
    model.set_parameters(0.5, 0.3, 0.2, 3, 0.1)
    results = model.simulate(n_timesteps=50, seed=1)
    assert len(results['H_states']) == 50
    assert len(results['phase_sync']) == 50
    assert not np.isnan(results['reflection'][-1])


def test_coherence_detection():
    model = RCDModel()
    results = model.simulate(n_timesteps=100, seed=5)
    mask, indices = model.detect_coherence(results, threshold=0.9)
    assert isinstance(indices, np.ndarray)
    assert mask.shape == (100,)


if __name__ == "__main__":
    test_initialization()
    test_metrics_valid()
    test_simulation_runs()
    test_coherence_detection()
    print("All RCDModel tests passed.")
