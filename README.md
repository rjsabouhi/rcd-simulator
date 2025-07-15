# Recursive Cognitive Dynamics (RCD) Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rcd-simulator.streamlit.app/)

Live demo: https://rcd-simulator.streamlit.app/


The RCD Simulator models the recursive interaction between a human cognitive state (`H(t)`) and an AI generative model state (`M(t)`) over time, under the influence of a reflection feedback signal `R(t)`.

## Overview
- `H(t)`: Human cognitive manifold
- `M(t)`: Model (AI) cognitive manifold
- `R(t)`: Recursive reflection signal
- Metrics:
  - `γ(t)` Phase Synchronization
  - `ρ(t)` Semantic Correlation
  - `d(t)` Procrustes Distance

## Update Equations
- `H_{t+1} = f(H_t, M_t, R_t)`
- `M_{t+1} = g(M_t, H_t, R_t)`
- `R_{t+1} = α·R_t + β·γ(t) + δ·ρ(t)`

## Project Structure
```
rcd-simulator/
├── rcd_model.py            # Core simulation logic
├── visualizations.py       # Plotly visualization utilities
├── streamlit_app.py        # Streamlit frontend interface
├── requirements.txt        # Required packages
└── README.md               # You're here
```

## How to Run
### 1. Clone repo
```bash
git clone https://github.com/yourname/rcd-simulator.git
cd rcd-simulator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit app
```bash
streamlit run streamlit_app.py
```

## Features
- Interactive control over model parameters (α, β, δ, noise, dimensionality)
- Dynamic plots for:
  - 3D manifold evolution
  - Metric time series
  - Coherence detection and phase-space analysis
  - Heatmaps of state matrices
- Data + parameter export to CSV and JSON

## Research Applications
- Models feedback loops in human-AI interaction
- Useful for cognitive modeling, alignment studies, recursive systems
- Extensible to multi-agent recursive cognition

## 📘 Citation / Credits
Developed by [Ryan] for experimental modeling of Recursive Cognitive Dynamics (RCD).

MIT License.
