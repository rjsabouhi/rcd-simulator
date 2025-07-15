"""
rcd_model.py

Core mathematical engine for the Recursive Cognitive Dynamics (RCD) Simulator.
Defines the recursive update model, metric computations, and simulation loop.
"""

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

class RCDModel:
    """
    Recursive Cognitive Dynamics Model

    Simulates the evolution of human (H) and model (M) cognitive manifolds
    over time via recursive feedback and reflection (R).
    Computes three metrics:
        - Phase synchronization (gamma)
        - Semantic correlation (rho)
        - Procrustes distance (d)
    And updates state accordingly.
    """

    def __init__(self):
        self.alpha = 0.7   # Reflection persistence
        self.beta = 0.2    # Phase synchronization weight
        self.delta = 0.1   # Semantic correlation weight
        self.n_dimensions = 3
        self.noise_level = 0.1

        self.H = None  # Human cognitive state manifold
        self.M = None  # Model (AI) state manifold
        self.R = 0.0   # Reflection value

    def set_parameters(self, alpha, beta, delta, n_dimensions, noise_level):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.n_dimensions = n_dimensions
        self.noise_level = noise_level

    def initialize_manifolds(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.H = np.random.randn(self.n_dimensions, self.n_dimensions)
        self.M = np.random.randn(self.n_dimensions, self.n_dimensions)
        self.R = np.random.uniform(0, 1)

    def compute_phase_synchronization(self, H, M):
        try:
            H_eig = np.linalg.eigvals(H @ H.T)
            M_eig = np.linalg.eigvals(M @ M.T)
            phase_diff = np.angle(H_eig) - np.angle(M_eig)
            gamma = np.abs(np.mean(np.exp(1j * phase_diff)))
            return gamma
        except:
            H_flat, M_flat = H.flatten(), M.flatten()
            return max(0, np.corrcoef(H_flat, M_flat)[0, 1])

    def compute_semantic_correlation(self, H, M):
        try:
            rho, _ = pearsonr(H.flatten(), M.flatten())
            return rho if not np.isnan(rho) else 0.0
        except:
            return 0.0

    def compute_procrustes_distance(self, H, M):
        try:
            _, _, disparity = procrustes(H, M)
            return disparity
        except:
            return np.linalg.norm(H - M) / (np.linalg.norm(H) + np.linalg.norm(M) + 1e-8)

    def update_manifold_H(self, H, M, R):
        coupling = 0.3 * M @ M.T @ H
        reflection_term = R * np.eye(self.n_dimensions) @ H
        damping = 0.95 * H
        noise = self.noise_level * np.random.randn(*H.shape)
        return damping + 0.1 * coupling + 0.05 * reflection_term + noise

    def update_manifold_M(self, M, H, R):
        coupling = 0.3 * H @ H.T @ M
        reflection_term = R * np.eye(self.n_dimensions) @ M
        damping = 0.95 * M
        noise = self.noise_level * np.random.randn(*M.shape)
        return damping + 0.1 * coupling + 0.05 * reflection_term + noise

    def update_reflection(self, R_prev, gamma, rho):
        R_new = self.alpha * R_prev + self.beta * gamma + self.delta * rho
        return np.clip(R_new, 0, 2)

    def simulate(self, n_timesteps=100, seed=None):
        self.initialize_manifolds(seed=seed)

        results = {
            'H_states': [], 'M_states': [],
            'phase_sync': [], 'semantic_corr': [],
            'procrustes_dist': [], 'reflection': []
        }

        for t in range(n_timesteps):
            results['H_states'].append(self.H.copy())
            results['M_states'].append(self.M.copy())

            gamma = self.compute_phase_synchronization(self.H, self.M)
            rho = self.compute_semantic_correlation(self.H, self.M)
            d = self.compute_procrustes_distance(self.H, self.M)

            results['phase_sync'].append(gamma)
            results['semantic_corr'].append(rho)
            results['procrustes_dist'].append(d)
            results['reflection'].append(self.R)

            self.H = self.update_manifold_H(self.H, self.M, self.R)
            self.M = self.update_manifold_M(self.M, self.H, self.R)
            self.R = self.update_reflection(self.R, gamma, rho)

        return results

    def detect_coherence(self, results, threshold=0.95):
        gamma = np.array(results['phase_sync'])
        d = np.array(results['procrustes_dist'])
        mask = (gamma > threshold) & (d < (1 - threshold))
        return mask, np.where(mask)[0]
