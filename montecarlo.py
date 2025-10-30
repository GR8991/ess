import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

def linear_degradation(degradation_rate, years):
    return (1 - degradation_rate) ** years

def nonlinear_degradation(degradation_rate, years):
    return np.exp(-degradation_rate * years**0.5)

def economic_model_with_augmentation(batt_cost_initial, batt_cost_decline, discount_rate, revenue_energy, revenue_capacity, degradation_rate, degradation_model, project_lifetime, augmentations):
    years = np.arange(project_lifetime)
    capacity = np.zeros(project_lifetime)
    # Base capacity 100 kWh from year 0
    capacity[0:] = 100
    
    capex_cashflow = np.zeros(project_lifetime)
    
    # Initial system CAPEX
    capex_cashflow[0] += batt_cost_initial * 100
    
    # Add augmentations
    for (year, add_kwh) in augmentations:
        if 0 <= year < project_lifetime:
            capacity[year:] += add_kwh
            # Apply augmentation cost with 20% premium and cost decline applied to year of investment
            year_cost = batt_cost_initial * np.exp(-batt_cost_decline * year) * 1.20
            capex_cashflow[year] += year_cost * add_kwh
    
    # Calculate SOH each year depending on degradation model and capacity segments
    soh = np.zeros(project_lifetime)
    
    for yr in years:
        total_soh = 0.0
        # Base system degradation from year 0
        base_age = yr
        base_soh = linear_degradation(degradation_rate, base_age) if degradation_model == 'Linear' else nonlinear_degradation(degradation_rate, base_age)
        total_soh += 100 * base_soh
        
        # For later augmentations
        for (aug_year, add_kwh) in augmentations:
            if aug_year <= yr:
                aug_age = yr - aug_year
                aug_soh = linear_degradation(degradation_rate, aug_age) if degradation_model == 'Linear' else nonlinear_degradation(degradation_rate, aug_age)
                total_soh += add_kwh * aug_soh
        
        soh[yr] = total_soh
    
    # Annual revenue depends on available SOH capacity multiplied by unit revenues
    annual_revenue = (revenue_energy + revenue_capacity) * soh
    
    # Cashflow for year - revenue minus capex in that year
    cashflow = annual_revenue - capex_cashflow
    
    # Discount cashflows to NPV
    discount_factors = (1 + discount_rate) ** years
    npv = np.sum(cashflow / discount_factors)
    
    # Calculate cumulative cashflow for payback
    cumulative_cashflow = np.cumsum(cashflow)
    payback = np.argmax(cumulative_cashflow > 0) + 1 if np.any(cumulative_cashflow > 0) else project_lifetime
    
    # LCOS = total discounted cost / total energy delivered
    discounted_cost = np.sum(capex_cashflow / discount_factors)
    discounted_energy = np.sum(soh / discount_factors)
    lcos = discounted_cost / discounted_energy if discounted_energy > 0 else np.nan
    
    return npv, payback, lcos

def run_simulation(n_sims, params, augmentations):
    results = []
    for _ in range(n_sims):
        sample = {}
        for key, val in params.items():
            if isinstance(val, tuple) and len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
                sample[key] = np.random.uniform(val[0], val[1])
            else:
                sample[key] = val
        
        npv, payback, lcos = economic_model_with_augmentation(
            sample['batt_cost_initial'],
            sample['batt_cost_decline'],
            sample['discount_rate'],
            sample['revenue_energy'],
            sample['revenue_capacity'],
            sample['degradation_rate'],
            sample['degradation_model'],
            sample['project_lifetime'],
            augmentations
        )
        results.append({'NPV': npv, 'Payback': payback, 'LCOS': lcos, **sample})
    return pd.DataFrame(results)

def create_word_report(df, inputs, opinion, augmentations):
    doc = Document()
    doc.add_heading('Monte Carlo Simulation Report - BESS Economic Model', level=1)
    doc.add_heading('Input Parameters', level=2)
    for key, val in inputs.items():
        doc.add_paragraph(f"{key}: {val}")
    if augmentations:
        doc.add_heading('Augmentation Events', level=2)
        table = doc.add_table(rows=1, cols=2)
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Year'
        hdr_cells[1].text = 'Added Capacity (kWh)'
        for (year, add_kwh) in augmentations:
            row_cells = table.add_row().cells
            row_cells[0].text = str(year)
            row_cells[1].text = str(add_kwh)
    doc.add_heading('Simulation Results Summary', level=2)
    desc = df.describe().transpose()
    table = doc.add_table(rows=1, cols=len(desc.columns)+1)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Metric'
    for i, colname in enumerate(desc.columns):
        hdr_cells[i+1].text = colname.capitalize()
    for metric, row in desc.iterrows():
        row_cells = table.add_row().cells
        row_cells[0].text = str(metric)
        for i, val in enumerate(row):
            row_cells[i+1].text = f"{val:.2f}"
    doc.add_heading('Interpretation for Non-Economists', level=2)
    p = doc.add_paragraph(opinion)
    p.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
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

def generate_opinion(df):
    mean_npv = df['NPV'].mean()
    mean_payback = df['Payback'].mean()
    mean_lcos = df['LCOS'].mean()
    opinion = (
        f"The average Net Present Value (NPV) of the project is ${mean_npv:,.2f}, indicating general profitability. "
        f"The average payback period is {mean_payback:.1f} years, representing the typical time to recover the investment. "
        f"The Levelized Cost of Storage (LCOS) averages ${mean_lcos:,.2f} per kWh - a key cost-efficiency metric. "
        "These results incorporate uncertainties in costs, revenues, degradation rates, project lifetime, and augmentation strategy, "
        "providing a comprehensive probabilistic economic analysis."
    )
    return opinion

# Streamlit UI
st.title('BESS Economic Model Monte Carlo Simulation with Staged Augmentation')

st.sidebar.header('Simulation Parameters')
batt_cost_initial = st.sidebar.slider('Initial Battery Cost ($/kWh)', 300, 500, (350, 450), step=10)
batt_cost_decline = st.sidebar.slider('Battery Cost Decline Rate (%/year)', 0.0, 0.06, (0.01, 0.04), step=0.005)
discount_rate = st.sidebar.slider('Discount Rate', 0.03, 0.09, (0.05, 0.08), step=0.005)
revenue_energy = st.sidebar.slider('Energy Revenue ($/kWh-year)', 30, 70, (40, 60), step=1)
revenue_capacity = st.sidebar.slider('Capacity Revenue ($/kWh-year)', 10, 50, (20, 40), step=1)
degradation_rate = st.sidebar.slider('Degradation Rate (%/year)', 0.01, 0.04, (0.02, 0.03), step=0.001)
degradation_model = st.sidebar.radio('Degradation Model', ['Linear', 'Nonlinear'])
project_lifetime = st.sidebar.slider('Project Lifetime (years)', 5, 30, 20, step=1)

enable_augmentation = st.sidebar.checkbox('Enable Staged Augmentation')

augmentations = []
if enable_augmentation:
    n_augment = st.sidebar.number_input('Number of Augmentations', 1, 5, 1)
    for i in range(n_augment):
        year = st.sidebar.number_input(f'Augmentation {i+1} Year', 1, project_lifetime-1, step=1, key=f'aug_year_{i}')
        capacity = st.sidebar.number_input(f'Augmentation {i+1} Capacity (kWh)', 1, 500, step=1, key=f'aug_cap_{i}')
        augmentations.append((year, capacity))

n_sims = st.sidebar.number_input('Number of Simulations', 1000, 10000, 5000, step=500)

params = {
    'batt_cost_initial': batt_cost_initial,
    'batt_cost_decline': batt_cost_decline,
    'discount_rate': discount_rate,
    'revenue_energy': revenue_energy,
    'revenue_capacity': revenue_capacity,
    'degradation_rate': degradation_rate,
    'degradation_model': degradation_model,
    'project_lifetime': project_lifetime
}

if st.button('Run Simulation'):
    with st.spinner('Running simulations...'):
        df_results = run_simulation(n_sims, params, augmentations)

    st.success('Simulation complete!')

    st.subheader('Simulation Results Summary Table')
    st.dataframe(df_results.describe().transpose().style.format("{:.2f}"))

    opinion_text = generate_opinion(df_results)
    st.subheader('Interpretation for Non-Economists')
    st.write(opinion_text)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    df_results['NPV'].hist(ax=axs[0], bins=30, color='skyblue')
    axs[0].set_title('NPV Distribution')
    df_results['Payback'].hist(ax=axs[1], bins=30, color='salmon')
    axs[1].set_title('Payback Distribution')
    df_results['LCOS'].hist(ax=axs[2], bins=30, color='lightgreen')
    axs[2].set_title('LCOS Distribution')
    st.pyplot(fig)

    doc = create_word_report(df_results, {k: f"{v[0]:.3f} to {v[1]:.3f}" if isinstance(v, tuple) else str(v) for k, v in params.items()}, opinion_text, augmentations)
    buf = BytesIO()
    doc.save(buf)
    st.download_button(
        label="Download Word Report",
        data=buf.getvalue(),
        file_name="BESS_MonteCarlo_Report.docx",

        mime=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document\"
    )
