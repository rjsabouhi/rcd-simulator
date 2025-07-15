"""
streamlit_app.py

Streamlit interface for the Recursive Cognitive Dynamics (RCD) Simulator.
Provides interactive controls, visualization panels, and simulation export.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rcd_model import RCDModel
from visualizations import create_manifold_plot, create_metrics_plot, create_coherence_plot, create_manifold_heatmap
import json

# Streamlit config
st.set_page_config(
    page_title="Recursive Cognitive Dynamics Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Recursive Cognitive Dynamics (RCD) Simulator")
st.markdown("""
This app simulates recursive cognitive dynamics between a human (H) and model (M)
cognitive manifold under a reflection feedback loop R(t).
""")

# Session state init
if 'rcd_model' not in st.session_state:
    st.session_state.rcd_model = RCDModel()
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'selected_time' not in st.session_state:
    st.session_state.selected_time = 0

# Sidebar controls
st.sidebar.header("Model Parameters")

alpha = st.sidebar.slider("α (Reflection Persistence)", 0.0, 1.0, 0.7, 0.01)
beta = st.sidebar.slider("β (Phase Sync Weight)", 0.0, 1.0, 0.2, 0.01)
delta = st.sidebar.slider("δ (Semantic Corr Weight)", 0.0, 1.0, 0.1, 0.01)
n_dim = st.sidebar.slider("Manifold Dimensionality", 2, 10, 3, 1)
noise = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1, 0.01)
n_steps = st.sidebar.slider("Time Steps", 50, 500, 200, 10)
coherence_thresh = st.sidebar.slider("Coherence Threshold", 0.8, 0.99, 0.95, 0.01)
seed = st.sidebar.number_input("Random Seed (optional)", value=42, step=1)

# Run simulation button
if st.sidebar.button("Run Simulation"):
    model = RCDModel()
    model.set_parameters(alpha, beta, delta, n_dim, noise)
    data = model.simulate(n_timesteps=n_steps, seed=int(seed))
    st.session_state.rcd_model = model
    st.session_state.simulation_data = data
    st.session_state.selected_time = n_steps - 1

# Main display
if st.session_state.simulation_data:
    data = st.session_state.simulation_data

    tab1, tab2, tab3, tab4 = st.tabs(["Manifold", "Metrics", "Coherence", "Explore"])

    with tab1:
        st.subheader("Cognitive Manifold Evolution")
        fig = create_manifold_plot(data)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Metric Evolution Over Time")
        fig = create_metrics_plot(data)
        st.plotly_chart(fig, use_container_width=True)

        # Reflection plot
        r_fig = go.Figure()
        r_fig.add_trace(go.Scatter(
            x=list(range(len(data['reflection']))),
            y=data['reflection'],
            mode='lines+markers',
            name='R(t)',
            line=dict(color='purple')
        ))
        r_fig.update_layout(title="Recursive Reflection R(t)", xaxis_title="Time", yaxis_title="R")
        st.plotly_chart(r_fig, use_container_width=True)

    with tab3:
        st.subheader("Coherence Analysis")
        fig = create_coherence_plot(data, threshold=coherence_thresh)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("State Explorer")
        t = st.slider("Select Time Step", 0, len(data['H_states'])-1, st.session_state.selected_time)
        st.session_state.selected_time = t

        fig = create_manifold_heatmap(data, t)
        st.plotly_chart(fig, use_container_width=True)

        st.metric("γ(t)", f"{data['phase_sync'][t]:.3f}")
        st.metric("ρ(t)", f"{data['semantic_corr'][t]:.3f}")
        st.metric("d(t)", f"{data['procrustes_dist'][t]:.3f}")
        st.metric("R(t)", f"{data['reflection'][t]:.3f}")

        # Export CSV
        df = pd.DataFrame({
            't': list(range(len(data['phase_sync']))),
            'gamma': data['phase_sync'],
            'rho': data['semantic_corr'],
            'd': data['procrustes_dist'],
            'R': data['reflection']
        })
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="rcd_metrics.csv", mime="text/csv")

        # Export parameters
        params = {
            'alpha': alpha, 'beta': beta, 'delta': delta,
            'n_dimensions': n_dim, 'noise': noise,
            'timesteps': n_steps, 'seed': int(seed)
        }
        json_str = json.dumps(params, indent=2)
        st.download_button("Download Params", json_str, file_name="rcd_parameters.json", mime="application/json")

else:
    st.info("Use the sidebar to configure parameters and run a simulation.")
