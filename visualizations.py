"""
visualizations.py

Visualization utilities for RCD simulator: plots cognitive manifolds,
metric evolution, coherence detection, and state heatmaps.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_manifold_plot(data):
    H = data['H_states']
    M = data['M_states']
    T = min(20, len(H))
    indices = np.linspace(0, len(H)-1, T, dtype=int)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("H(t) - Human", "M(t) - Model"), specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]])

    for i, t in enumerate(indices):
        for manifold, col, color in [(H, 1, 'Blues'), (M, 2, 'Reds')]:
            mat = manifold[t]
            if mat.shape[0] >= 3:
                x, y, z = mat[0, :3], mat[1, :3], mat[2, :3]
            else:
                x = mat[0, :min(3, mat.shape[1])]
                y = mat[1, :min(3, mat.shape[1])] if mat.shape[0] > 1 else np.zeros_like(x)
                z = mat[2, :min(3, mat.shape[1])] if mat.shape[0] > 2 else np.zeros_like(x)

            opacity = 0.3 + 0.7 * (i / T)

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers+lines',
                marker=dict(size=5, color=t, colorscale=color, opacity=opacity),
                line=dict(width=3), showlegend=False
            ), row=1, col=col)

    fig.update_layout(title="Cognitive Manifold Evolution", height=600)
    return fig

def create_metrics_plot(data):
    t = list(range(len(data['phase_sync'])))
    fig = make_subplots(rows=2, cols=2, subplot_titles=("γ(t) Phase Sync", "ρ(t) Semantic Corr", "d(t) Procrustes", "Combined Metrics"))

    fig.add_trace(go.Scatter(x=t, y=data['phase_sync'], mode='lines+markers', name='γ(t)', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=data['semantic_corr'], mode='lines+markers', name='ρ(t)', line=dict(color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=data['procrustes_dist'], mode='lines+markers', name='d(t)', line=dict(color='red')), row=2, col=1)

    norm_d = 1 - np.array(data['procrustes_dist'])
    fig.add_trace(go.Scatter(x=t, y=data['phase_sync'], mode='lines', name='γ(t)', line=dict(color='blue')), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=data['semantic_corr'], mode='lines', name='ρ(t)', line=dict(color='green')), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=norm_d, mode='lines', name='1 - d(t)', line=dict(color='red')), row=2, col=2)

    fig.update_layout(title="Metric Evolution", height=700)
    return fig

def create_coherence_plot(data, threshold=0.95):
    t = np.arange(len(data['phase_sync']))
    gamma = np.array(data['phase_sync'])
    d = np.array(data['procrustes_dist'])
    mask = (gamma > threshold) & (d < (1 - threshold))

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Coherence over Time", "γ(t) vs 1 - d(t)"))

    fig.add_trace(go.Scatter(x=t, y=gamma, name='γ(t)', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=1 - d, name='1 - d(t)', line=dict(color='red')), row=1, col=1)

    fig.add_hline(y=threshold, line_dash='dash', line_color='gray', row=1, col=1)

    if np.any(mask):
        fig.add_trace(go.Scatter(
            x=t[mask], y=[1.05]*np.sum(mask),
            mode='markers', name='Coherent States',
            marker=dict(symbol='star', size=10, color='gold', line=dict(width=1, color='orange'))
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=gamma, y=1 - d,
        mode='markers', name='Trajectory',
        marker=dict(size=8, color=['gold' if m else 'red' for m in mask], opacity=0.7)
    ), row=2, col=1)

    fig.update_layout(title="Coherence Detection", height=700)
    return fig

def create_manifold_heatmap(data, t):
    H, M = data['H_states'][t], data['M_states'][t]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"H(t={t})", f"M(t={t})"), specs=[[{"type": "heatmap"}, {"type": "heatmap"}]])

    fig.add_trace(go.Heatmap(z=H, colorscale='Blues', name='H'), row=1, col=1)
    fig.add_trace(go.Heatmap(z=M, colorscale='Reds', name='M'), row=1, col=2)

    fig.update_layout(title=f"State Snapshot at t = {t}", height=400)
    return fig