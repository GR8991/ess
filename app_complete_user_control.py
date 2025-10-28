# BESS LCOS Analysis - Complete User-Configurable Version
# Version 3.0 - Full User Control: SOH, Augmentation, Approach Selection

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import io
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

st.set_page_config(page_title="BESS LCOS Analysis", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.main { background-color: #f5f5f5; }
.stMetric { background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
h1, h2, h3 { color: #003366; }
.section-header { background: linear-gradient(90deg, #003366, #0055aa); color: white; padding: 15px; border-radius: 5px; margin: 20px 0 10px 0; }
.info-box { background-color: #e3f2fd; padding: 12px; border-left: 4px solid #1976d2; margin: 10px 0; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# Classes
class DegradationProfile:
    def __init__(self, soh_values_dict):
        self.soh_dict = soh_values_dict
        self.floor_soh = min(soh_values_dict.values()) / 100
    def calculate_soh(self, year):
        if year in self.soh_dict:
            return self.soh_dict[year] / 100
        years_list = sorted(self.soh_dict.keys())
        if year < min(years_list):
            return self.soh_dict[min(years_list)] / 100
        if year > max(years_list):
            return self.soh_dict[max(years_list)] / 100
        for i in range(len(years_list)-1):
            if years_list[i] <= year <= years_list[i+1]:
                y1, y2 = years_list[i], years_list[i+1]
                soh1, soh2 = self.soh_dict[y1], self.soh_dict[y2]
                soh = soh1 + (soh2 - soh1) * (year - y1) / (y2 - y1)
                return max(soh / 100, self.floor_soh)
        return self.floor_soh

class CostComponents:
    def __init__(self, battery_cost_kwh):
        self.battery_cost_kwh = battery_cost_kwh
        self.system_cost_per_mwh = battery_cost_kwh * 1000 * 1.05 * 1.08
    def fixed_opex_base(self):
        return 850000
    def insurance(self, initial_capex):
        return initial_capex * 1000000 * 0.0075
    def property_tax(self, initial_capex):
        return initial_capex * 1000000 * 0.005
    def variable_opex(self, year):
        if year <= 5: return 100000
        elif year <= 10: return 240000
        elif year <= 15: return 400000
        else: return 550000

class RevenueModel:
    def __init__(self, power_mw, energy_margin):
        self.power_mw = power_mw
        self.energy_margin = energy_margin
    def capacity_payment_year(self, year):
        if year <= 5: return 50000
        elif year <= 10: return 50000
        elif year <= 15: return 45000
        else: return 40000
    def calculate_capacity_revenue(self, year):
        return (self.power_mw * self.capacity_payment_year(year)) / 1000000
    def calculate_energy_revenue(self, available_mwh, cycles_per_year, margin):
        return (available_mwh * cycles_per_year * margin) / 1000000

# Calculation functions
def calculate_overbuild_cashflow(power_mw, energy_mwh, project_life, cycles, battery_cost, decline_rate, 
                                 inflation, discount_rate, energy_margin, degradation):
    revenue_model = RevenueModel(power_mw, energy_margin)
    overbuilt_capacity = energy_mwh / degradation.floor_soh
    cost_component = CostComponents(battery_cost)
    initial_capex = (overbuilt_capacity * cost_component.system_cost_per_mwh) / 1000000
    years = np.arange(0, project_life + 1)
    cash_flows = []
    
    for year in years:
        soh = degradation.calculate_soh(year)
        available_capacity = overbuilt_capacity * soh
        
        if year == 0:
            cf = {"year": year, "soh": soh * 100, "capacity": available_capacity, "capacity_revenue": 0,
                  "energy_revenue": 0, "total_revenue": 0, "fixed_opex": 0, "variable_opex": 0,
                  "total_opex": 0, "capex": -initial_capex, "net_cf": -initial_capex}
        else:
            capacity_rev = revenue_model.calculate_capacity_revenue(year)
            energy_rev = revenue_model.calculate_energy_revenue(available_capacity, cycles, energy_margin)
            total_rev = capacity_rev + energy_rev
            cost_comp = CostComponents(battery_cost * ((1 + decline_rate) ** (year - 1)))
            fixed_opex = (cost_comp.fixed_opex_base() + cost_comp.insurance(initial_capex) + 
                         cost_comp.property_tax(initial_capex)) * (1 + inflation) ** (year - 1) / 1000000
            variable_opex = cost_comp.variable_opex(year) * (1 + inflation) ** (year - 1) / 1000000
            total_opex = fixed_opex + variable_opex
            cf = {"year": year, "soh": soh * 100, "capacity": available_capacity, "capacity_revenue": capacity_rev,
                  "energy_revenue": energy_rev, "total_revenue": total_rev, "fixed_opex": fixed_opex,
                  "variable_opex": variable_opex, "total_opex": total_opex, "capex": 0, "net_cf": total_rev - total_opex}
        cash_flows.append(cf)
    
    df = pd.DataFrame(cash_flows)
    df["cumulative_cf"] = df["net_cf"].cumsum()
    df["discount_factor"] = 1 / (1 + discount_rate) ** df["year"]
    df["pv_cf"] = df["net_cf"] * df["discount_factor"]
    df["cumulative_pv"] = df["pv_cf"].cumsum()
    
    total_capex = abs(df[df["year"] == 0]["net_cf"].values[0])
    npv = df["pv_cf"].sum()
    total_energy = df[df["year"] > 0]["capacity"].sum() * cycles
    lcos = ((total_capex + df[df["year"] > 0]["total_opex"].sum()) * 1000000) / (total_energy * 1000) if total_energy > 0 else 0
    
    summary = {"initial_capex": total_capex, "total_revenue_nominal": df[df["year"] > 0]["total_revenue"].sum(),
               "total_opex_nominal": df[df["year"] > 0]["total_opex"].sum(), "npv": npv, "total_energy": total_energy,
               "lcos": lcos, "avg_capacity": df[df["year"] > 0]["capacity"].mean(),
               "payback_nominal": next((y for y in df["year"] if df[df["year"] == y]["cumulative_cf"].values[0] > 0), None)}
    
    return {"cashflow": df, "summary": summary, "overbuilt_capacity": overbuilt_capacity}

def calculate_staged_cashflow(power_mw, energy_mwh, project_life, cycles, battery_cost, decline_rate, 
                             inflation, discount_rate, energy_margin, degradation, augmentations):
    revenue_model = RevenueModel(power_mw, energy_margin)
    augmentations = [a for a in augmentations if a is not None and len(a) == 2]
    base_capacity = energy_mwh * 1.075
    cost_component = CostComponents(battery_cost)
    initial_capex = (base_capacity * cost_component.system_cost_per_mwh) / 1000000
    
    aug_costs = {}
    for aug_year, aug_mwh in augmentations:
        aug_cost_kwh = battery_cost * ((1 + decline_rate) ** (aug_year - 1))
        aug_cost_comp = CostComponents(aug_cost_kwh)
        integration_premium = 1.12 if aug_year < 10 else 1.15
        aug_costs[aug_year] = (aug_mwh * aug_cost_comp.system_cost_per_mwh * integration_premium) / 1000000
    
    years = np.arange(0, project_life + 1)
    cash_flows = []
    
    for year in years:
        soh = degradation.calculate_soh(year)
        base_avail = base_capacity * soh
        aug_cap = 0
        for aug_year, aug_mwh in augmentations:
            if year >= aug_year:
                aug_soh = degradation.calculate_soh(year - aug_year)
                aug_cap += aug_mwh * aug_soh
        total_capacity = base_avail + aug_cap
        
        if year == 0:
            cf = {"year": year, "soh": soh * 100, "capacity": base_capacity, "total_capacity": base_capacity,
                  "capacity_revenue": 0, "energy_revenue": 0, "total_revenue": 0, "fixed_opex": 0,
                  "variable_opex": 0, "total_opex": 0, "capex": -initial_capex, "net_cf": -initial_capex}
        else:
            capacity_rev = revenue_model.calculate_capacity_revenue(year)
            energy_rev = revenue_model.calculate_energy_revenue(total_capacity, cycles, energy_margin)
            total_rev = capacity_rev + energy_rev
            cost_comp = CostComponents(battery_cost * ((1 + decline_rate) ** (year - 1)))
            fixed_opex = (cost_comp.fixed_opex_base() + cost_comp.insurance(initial_capex) +
                         cost_comp.property_tax(initial_capex)) * (1 + inflation) ** (year - 1) / 1000000
            variable_opex = cost_comp.variable_opex(year) * (1 + inflation) ** (year - 1) / 1000000
            total_opex = fixed_opex + variable_opex
            aug_capex = sum([-aug_costs.get(aug_year, 0) for aug_year, _ in augmentations if aug_year == year])
            cf = {"year": year, "soh": soh * 100, "capacity": total_capacity, "total_capacity": total_capacity,
                  "capacity_revenue": capacity_rev, "energy_revenue": energy_rev, "total_revenue": total_rev,
                  "fixed_opex": fixed_opex, "variable_opex": variable_opex, "total_opex": total_opex,
                  "capex": aug_capex, "net_cf": total_rev - total_opex + aug_capex}
        cash_flows.append(cf)
    
    df = pd.DataFrame(cash_flows)
    df["cumulative_cf"] = df["net_cf"].cumsum()
    df["discount_factor"] = 1 / (1 + discount_rate) ** df["year"]
    df["pv_cf"] = df["net_cf"] * df["discount_factor"]
    df["cumulative_pv"] = df["pv_cf"].cumsum()
    
    total_capex = abs(df[df["year"] == 0]["net_cf"].values[0]) + sum(aug_costs.values())
    npv = df["pv_cf"].sum()
    total_energy = df[df["year"] > 0]["capacity"].sum() * cycles
    lcos = ((initial_capex + sum([c / (1 + discount_rate) ** y for y, c in aug_costs.items()]) + 
             df[df["year"] > 0]["total_opex"].sum()) * 1000000) / (total_energy * 1000) if total_energy > 0 else 0
    
    summary = {"initial_capex": initial_capex, "total_revenue_nominal": df[df["year"] > 0]["total_revenue"].sum(),
               "total_opex_nominal": df[df["year"] > 0]["total_opex"].sum(), "npv": npv, "total_energy": total_energy,
               "lcos": lcos, "avg_capacity": df[df["year"] > 0]["capacity"].mean(),
               "payback_nominal": next((y for y in df["year"] if df[df["year"] == y]["cumulative_cf"].values[0] > 0), None)}
    
    return {"cashflow": df, "summary": summary, "base_capacity": base_capacity, "aug_costs": aug_costs}

# Main app
def main():
    st.title("⚡ BESS LCOS Analysis Dashboard")
    st.markdown("**Professional Battery Energy Storage System Financial Modeling**")
    
    with st.sidebar:
        st.header("🔧 Configuration")
        page = st.radio("Select:", ["Input & Analysis", "Compare Scenarios", "Generate Reports"])
    
    if page == "Input & Analysis":
        show_input_analysis()
    elif page == "Compare Scenarios":
        show_scenario_comparison()
    else:
        show_reports()

def show_input_analysis():
    st.markdown('<div class="section-header">📊 Project Configuration & Financial Analysis</div>', unsafe_allow_html=True)
    
    # Tabs for configuration
    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Project Setup", "2️⃣ SOH Curve", "3️⃣ Augmentation", "4️⃣ Run Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Project Parameters")
            power_mw = st.number_input("Power Capacity (MW)", value=100, min_value=10, max_value=500, key="power")
            energy_mwh = st.number_input("Energy Capacity (MWh)", value=400, min_value=50, max_value=2000, key="energy")
            project_life = st.number_input("Project Life (years)", value=20, min_value=10, max_value=40, key="life")
            cycles_per_year = st.number_input("Cycles per Year", value=365, min_value=100, max_value=1000, key="cycles")
        
        with col2:
            st.subheader("Financial Assumptions")
            battery_cost = st.number_input("Battery Cost ($/kWh)", value=241, min_value=100, max_value=500, key="batt_cost")
            cost_decline = st.slider("Cost Decline Rate (%/year)", -8, 2, -4, key="decline") / 100
            inflation_rate = st.slider("Inflation Rate (%/year)", 0, 5, 2, key="inflation") / 100
            discount_rate = st.slider("Discount Rate (%)", 3, 15, 7, key="discount") / 100
            energy_margin = st.number_input("Energy Margin ($/MWh)", value=50, min_value=10, max_value=200, key="margin")
    
    with tab2:
        st.subheader("📉 Configure SOH (State of Health) Curve")
        st.markdown('<div class="info-box"><b>Instructions:</b> Enter SOH percentage for each year of project life. SOH represents remaining battery capacity (100% = new, 60% = degraded)</div>', unsafe_allow_html=True)
        
        soh_method = st.radio("SOH Input Method:", ["Quick (5-year intervals)", "Detailed (Every year)"], key="soh_method")
        
        soh_dict = {}
        if soh_method == "Quick (5-year intervals)":
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                soh_dict[0] = st.number_input("Year 0 (%)", 100, 100, 100, key="soh0")
            with col2:
                soh_dict[5] = st.number_input("Year 5 (%)", 60, 100, 88, key="soh5")
            with col3:
                soh_dict[10] = st.number_input("Year 10 (%)", 60, 100, 75, key="soh10")
            with col4:
                soh_dict[15] = st.number_input("Year 15 (%)", 60, 100, 63, key="soh15")
            with col5:
                soh_dict[20] = st.number_input("Year 20 (%)", 60, 100, 60, key="soh20")
        else:
            cols = st.columns(5)
            for i in range(int(project_life) + 1):
                col_idx = i % 5
                with cols[col_idx]:
                    default_soh = max(100 - 5 - (i-1)*2.5, 60) if i > 0 else 100
                    soh_dict[i] = st.number_input(f"Y{i}", 50, 100, int(default_soh), key=f"soh_y{i}")
        
        # Show SOH curve preview
        years_preview = sorted(soh_dict.keys())
        soh_preview = [soh_dict[y] for y in years_preview]
        fig_soh = go.Figure()
        fig_soh.add_trace(go.Scatter(x=years_preview, y=soh_preview, mode='lines+markers', 
                                     line=dict(color='#1976d2', width=3), marker=dict(size=8)))
        fig_soh.update_layout(title="SOH Curve Preview", xaxis_title="Year", yaxis_title="SOH (%)", height=350)
        st.plotly_chart(fig_soh, use_container_width=True)
    
    with tab3:
        st.subheader("🔧 Configure Augmentation Strategy")
        st.markdown('<div class="info-box"><b>What is Augmentation?</b> Adding battery capacity at specific years instead of building everything upfront (Overbuild)</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            num_augmentations = st.number_input("Number of Augmentations", 0, 5, 2, key="num_aug")
        with col2:
            st.info(f"You will add capacity {num_augmentations} times during project life")
        
        augmentations = []
        if num_augmentations > 0:
            st.markdown("**Enter each augmentation event:**")
            cols = st.columns(num_augmentations)
            for i in range(num_augmentations):
                with cols[i]:
                    st.markdown(f"**Augmentation {i+1}**")
                    aug_year = st.number_input(f"Year", 1, int(project_life)-1, min(7+i*7, int(project_life)-1), key=f"aug_year_{i}")
                    aug_mwh = st.number_input(f"Capacity (MWh)", 10, 500, 60-i*10, key=f"aug_mwh_{i}")
                    augmentations.append((aug_year, aug_mwh))
            
            # Show timeline
            timeline_text = " → ".join([f"Year {y}: +{m} MWh" for y, m in augmentations])
            st.success(f"📅 Timeline: {timeline_text}")
    
    with tab4:
        st.subheader("🎯 Select Analysis Approach")
        approach = st.selectbox("Choose approach to analyze:", 
                               ["Initial Build Only (Overbuild)", "Augmentation Only (Staged)", "Both (Comparison)"],
                               key="approach")
        
        if st.button("🚀 RUN ANALYSIS", key="run_btn"):
            with st.spinner("Calculating..."):
                degradation = DegradationProfile(soh_dict)
                
                if approach in ["Initial Build Only (Overbuild)", "Both (Comparison)"]:
                    overbuild_data = calculate_overbuild_cashflow(power_mw, energy_mwh, project_life, cycles_per_year,
                                                                  battery_cost, cost_decline, inflation_rate, discount_rate,
                                                                  energy_margin, degradation)
                    show_analysis_results("Initial Build Approach (Overbuild)", overbuild_data, "overbuild")
                
                if approach in ["Augmentation Only (Staged)", "Both (Comparison)"]:
                    staged_data = calculate_staged_cashflow(power_mw, energy_mwh, project_life, cycles_per_year,
                                                           battery_cost, cost_decline, inflation_rate, discount_rate,
                                                           energy_margin, degradation, augmentations)
                    show_analysis_results("Augmentation Approach (Staged)", staged_data, "staged")
                
                if approach == "Both (Comparison)":
                    show_comparison_analysis(overbuild_data, staged_data)

def show_analysis_results(title, data, approach_type):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial CAPEX ($M)", f"${data['summary']['initial_capex']:.2f}")
    with col2:
        st.metric("Project NPV ($M)", f"${data['summary']['npv']:.2f}")
    with col3:
        st.metric("LCOS ($/MWh)", f"${data['summary']['lcos']:.2f}")
    with col4:
        st.metric("Avg Capacity (MWh)", f"{data['summary']['avg_capacity']:.1f}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Cash Flow", "Charts", "Capacity", "Detailed Table"])
    
    with tab1:
        col1, col2 = st.columns(2)
        df = data["cashflow"]
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["year"], y=df["cumulative_cf"], mode="lines+markers", 
                                    line=dict(color="blue", width=3)))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(title="Cumulative Cash Flow", xaxis_title="Year", yaxis_title="$ Millions", height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["year"], y=df["cumulative_pv"], mode="lines+markers", 
                                    line=dict(color="darkblue", width=3)))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(title="Cumulative NPV", xaxis_title="Year", yaxis_title="$ Millions", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        df_ops = df[df["year"] > 0]
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_ops["year"], y=df_ops["total_revenue"], name="Revenue", marker_color="green"))
            fig.add_trace(go.Bar(x=df_ops["year"], y=df_ops["total_opex"], name="OPEX", marker_color="red"))
            fig.update_layout(title="Revenue vs OPEX", xaxis_title="Year", yaxis_title="$ Millions", barmode="group", height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["year"], y=df["soh"], mode="lines+markers", line=dict(color="purple", width=3)))
            fig.update_layout(title="Battery Degradation", xaxis_title="Year", yaxis_title="SOH (%)", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["year"], y=df["capacity"], mode="lines+markers", 
                                line=dict(color="orange", width=3), fill="tozeroy"))
        fig.update_layout(title="Available Capacity Over Time", xaxis_title="Year", yaxis_title="MWh", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.dataframe(df.round(2), use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, f"cashflow_{approach_type}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

def show_comparison_analysis(overbuild_data, staged_data):
    st.markdown('<div class="section-header">⚖️ Comparative Analysis</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    npv_adv = staged_data["summary"]["npv"] - overbuild_data["summary"]["npv"]
    capex_adv = overbuild_data["summary"]["initial_capex"] - staged_data["summary"]["initial_capex"]
    lcos_adv = overbuild_data["summary"]["lcos"] - staged_data["summary"]["lcos"]
    
    with col1:
        st.metric("NPV Advantage ($M)", f"${npv_adv:.2f}", f"{(npv_adv / abs(overbuild_data['summary']['npv']) * 100):.1f}% better")
    with col2:
        st.metric("CAPEX Savings ($M)", f"${capex_adv:.2f}", f"{(capex_adv / overbuild_data['summary']['initial_capex'] * 100):.1f}% reduction")
    with col3:
        st.metric("LCOS Advantage ($/MWh)", f"${lcos_adv:.2f}", f"{(lcos_adv / overbuild_data['summary']['lcos'] * 100):.1f}% lower")
    with col4:
        util_ob = (overbuild_data["summary"]["avg_capacity"] / 640) * 100
        util_st = (staged_data["summary"]["avg_capacity"] / staged_data["base_capacity"]) * 100
        st.metric("Utilization", f"{util_st:.1f}%", f"{(util_st - util_ob):.1f}% vs Overbuild")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Overbuild", "Staged"], y=[overbuild_data["summary"]["npv"], staged_data["summary"]["npv"]],
                            marker_color=["lightcoral", "lightgreen"]))
        fig.update_layout(title="NPV Comparison", yaxis_title="$ Millions", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Overbuild", "Staged"], y=[overbuild_data["summary"]["lcos"], staged_data["summary"]["lcos"]],
                            marker_color=["lightcoral", "lightgreen"]))
        fig.update_layout(title="LCOS Comparison", yaxis_title="$/MWh", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    fig = go.Figure()
    ob_cf = overbuild_data["cashflow"]
    st_cf = staged_data["cashflow"]
    fig.add_trace(go.Scatter(x=ob_cf["year"], y=ob_cf["capacity"], name="Overbuild", mode="lines+markers", 
                            line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=st_cf["year"], y=st_cf["capacity"], name="Staged", mode="lines+markers", 
                            line=dict(color="green", width=2)))
    fig.update_layout(title="Capacity Comparison", xaxis_title="Year", yaxis_title="MWh", height=450)
    st.plotly_chart(fig, use_container_width=True)

def show_scenario_comparison():
    st.markdown('<div class="section-header">📈 Scenario Analysis</div>', unsafe_allow_html=True)
    st.info("Use default settings from Input & Analysis tab")

def show_reports():
    st.markdown('<div class="section-header">📋 Generate Reports</div>', unsafe_allow_html=True)
    st.info("Configure and run analysis first, then download from the Detailed Table tabs")

if __name__ == "__main__":
    main()
