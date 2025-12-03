# Profit Leakage Detection in Service Operations

**Objective:** Detect abnormal margin behavior using **Bayesian anomaly scoring** and **Statistical Process Control (SPC)** on a synthetic dataset (~12k jobs, 2024).

## Statistical Methods & Domain Significance
- **Bayesian Hierarchical Model**: partial pooling stabilizes segment estimates; compute P(mu<0) to rank risky regions/techs.
- **Shewhart ±3σ & CUSUM**: monitor weekly margin% to catch sudden and persistent drifts (e.g., Q3 parts inflation).
- **Bootstrap CIs**: robust uncertainty on loss rates without normality assumption.

## Structure
- `data/service_orders_synthetic.csv`
- `notebooks/01_data_generation.ipynb`
- `notebooks/02_bayesian_model.ipynb`
- `notebooks/03_control_chart_analysis.ipynb`
- `notebooks/04_summary_dashboard.ipynb`
- `requirements.txt`
