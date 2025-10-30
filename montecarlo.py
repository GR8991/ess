import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from docx import Document
from docx.shared import Inches

# Define function to calculate key outputs based on sampled inputs
def economic_model(batt_cost_initial, batt_cost_decline, discount_rate, revenue_energy, revenue_capacity, degradation_rate):
    # Simplified model for demonstration purposes
    project_lifetime = 20
    initial_cost = batt_cost_initial * 100  # for 100 kWh base capacity
    # Assume cost declines linearly over lifetime
    cost_over_time = initial_cost * np.exp(-batt_cost_decline * np.arange(project_lifetime))
    # Calculate NPV of costs
    npv_cost = np.sum(cost_over_time / ((1 + discount_rate) ** np.arange(1, project_lifetime + 1)))
    # Revenue modeled as stable stream declining by degradation
    annual_revenue = (revenue_energy + revenue_capacity) * 100 * (1 - degradation_rate) ** np.arange(project_lifetime)
    npv_revenue = np.sum(annual_revenue / ((1 + discount_rate) ** np.arange(1, project_lifetime + 1)))
    # Calculate NPV of net cash flow
    npv = npv_revenue - npv_cost
    # Estimate payback year
    cumulative_cashflow = np.cumsum(annual_revenue - cost_over_time)
    payback = np.argmax(cumulative_cashflow > 0) + 1 if np.any(cumulative_cashflow > 0) else project_lifetime
    # LCOS = Total discounted costs / Total delivered energy (simplified)
    total_energy = 100 * project_lifetime * (1 - degradation_rate / 2)  # approx average degrade
    lcos = npv_cost / total_energy
    return npv, payback, lcos

# Run Monte Carlo simulation
def run_simulation(n_sims, params):
    results = []
    for _ in range(n_sims):
        sample = {key: np.random.uniform(low, high) for key, (low, high) in params.items()}
        npv, payback, lcos = economic_model(
            sample['batt_cost_initial'],
            sample['batt_cost_decline'],
            sample['discount_rate'],
            sample['revenue_energy'],
            sample['revenue_capacity'],
            sample['degradation_rate']
        )
        results.append({'NPV': npv, 'Payback': payback, 'LCOS': lcos, **sample})
    return pd.DataFrame(results)

def create_word_report(df, inputs):
    doc = Document()
    doc.add_heading('Monte Carlo Simulation Report - BESS Economic Model', level=1)
    doc.add_heading('Input Parameters', level=2)
    for key, val in inputs.items():
        doc.add_paragraph(f"{key}: {val}")
    doc.add_heading('Simulation Results Summary', level=2)
    desc = df.describe().transpose()
    doc.add_paragraph(desc.to_string())
    # Add plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    df['NPV'].hist(ax=axs[0], bins=30, color='skyblue')
    axs[0].set_title('NPV Distribution')
    df['Payback'].hist(ax=axs[1], bins=30, color='salmon')
    axs[1].set_title('Payback Distribution')
    df['LCOS'].hist(ax=axs[2], bins=30, color='lightgreen')
    axs[2].set_title('LCOS Distribution')
    plt.tight_layout()
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    plt.close(fig)
    img_stream.seek(0)
    doc.add_picture(img_stream, width=Inches(6))
    return doc

# Streamlit UI
st.title('BESS Economic Model Monte Carlo Simulation')

st.sidebar.header('Simulation Parameters')
batt_cost_initial = st.sidebar.slider('Initial Battery Cost ($/kWh)', 300, 500, (350, 450), step=10)
batt_cost_decline = st.sidebar.slider('Battery Cost Decline Rate (%/year)', 0.0, 0.06, (0.01, 0.04), step=0.005)
discount_rate = st.sidebar.slider('Discount Rate', 0.03, 0.09, (0.05, 0.08), step=0.005)
revenue_energy = st.sidebar.slider('Energy Revenue ($/kWh-year)', 30, 70, (40, 60), step=1)
revenue_capacity = st.sidebar.slider('Capacity Revenue ($/kWh-year)', 10, 50, (20, 40), step=1)
degradation_rate = st.sidebar.slider('System Degradation Rate (%/year)', 0.01, 0.04, (0.02, 0.03), step=0.001)
n_sims = st.sidebar.number_input('Number of Simulations', 1000, 10000, 5000, step=500)

params = {
    'batt_cost_initial': batt_cost_initial,
    'batt_cost_decline': batt_cost_decline,
    'discount_rate': discount_rate,
    'revenue_energy': revenue_energy,
    'revenue_capacity': revenue_capacity,
    'degradation_rate': degradation_rate
}

if st.button('Run Simulation'):
    with st.spinner('Running simulations...'):
        df_results = run_simulation(n_sims, params)

    st.success('Simulation complete!')
    st.subheader('Simulation Results')
    st.write(df_results.describe())

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    df_results['NPV'].hist(ax=axs[0], bins=30, color='skyblue')
    axs[0].set_title('NPV Distribution')
    df_results['Payback'].hist(ax=axs[1], bins=30, color='salmon')
    axs[1].set_title('Payback Distribution')
    df_results['LCOS'].hist(ax=axs[2], bins=30, color='lightgreen')
    axs[2].set_title('LCOS Distribution')
    st.pyplot(fig)

    # Download Word report
    doc = create_word_report(df_results, {k: f"{v[0]:.3f} to {v[1]:.3f}" for k, v in params.items()})
    buf = BytesIO()
    doc.save(buf)
    st.download_button(
        label="Download Word Report",
        data=buf.getvalue(),
        file_name="BESS_MonteCarlo_Report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
