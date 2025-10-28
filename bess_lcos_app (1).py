# BESS LCOS Analysis - Streamlit Application
# Energy Storage Project Economic Comparison Tool
# Overbuild vs. Staged Augmentation Approaches

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Page Configuration
st.set_page_config(
    page_title="BESS LCOS Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .winner-badge {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .loser-badge {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA MODELS & CALCULATIONS
# ============================================================================

class BESSProject:
    """Base class for BESS project calculations"""
    
    def __init__(self, params):
        self.params = params
        self.project_life = params['project_life']
        self.discount_rate = params['discount_rate']
        self.inflation_rate = params['inflation_rate']
        self.power_mw = params['power_mw']
        
    def calculate_cash_flows(self, capacity_schedule):
        """Calculate 20-year cash flows"""
        cf_data = []
        
        for year in range(self.project_life + 1):
            year_data = {
                'year': year,
                'calendar_year': 2025 + year,
                'soh': capacity_schedule[year]['soh'],
                'available_capacity': capacity_schedule[year]['capacity'],
            }
            
            # Calculate revenues and costs
            if year == 0:
                # Construction year - no operations
                year_data['capacity_revenue'] = 0
                year_data['energy_revenue'] = 0
                year_data['total_revenue'] = 0
                year_data['fixed_opex'] = 0
                year_data['variable_opex'] = 0
                year_data['total_opex'] = 0
                year_data['net_cash_flow'] = 0
            else:
                # Operating years
                year_data['capacity_revenue'] = self._calculate_capacity_revenue(year)
                year_data['energy_revenue'] = self._calculate_energy_revenue(year, year_data['available_capacity'])
                year_data['total_revenue'] = year_data['capacity_revenue'] + year_data['energy_revenue']
                
                year_data['fixed_opex'] = self._calculate_fixed_opex(year)
                year_data['variable_opex'] = self._calculate_variable_opex(year)
                year_data['total_opex'] = year_data['fixed_opex'] + year_data['variable_opex']
                
                year_data['net_cash_flow'] = year_data['total_revenue'] - year_data['total_opex']
            
            # Discount factors
            year_data['discount_factor'] = 1 / ((1 + self.discount_rate) ** year)
            year_data['pv_cash_flow'] = year_data['net_cash_flow'] * year_data['discount_factor']
            
            cf_data.append(year_data)
        
        return pd.DataFrame(cf_data)
    
    def _calculate_capacity_revenue(self, year):
        """Calculate annual capacity revenue"""
        if year <= 5:
            capacity_payment = 50000
        elif year <= 10:
            capacity_payment = 50000
        elif year <= 15:
            capacity_payment = 45000
        else:
            capacity_payment = 40000
        
        # Escalate at 1%/year
        escalation = (1.01 ** (year - 1))
        return (self.power_mw * capacity_payment * escalation) / 1_000_000
    
    def _calculate_energy_revenue(self, year, capacity_mwh):
        """Calculate annual energy revenue"""
        energy_margin = 50  # $/MWh
        annual_cycles = 365
        
        return (capacity_mwh * annual_cycles * energy_margin) / 1_000_000
    
    def _calculate_fixed_opex(self, year):
        """Calculate annual fixed OPEX"""
        base_fixed = 2.157175  # $M
        escalation = (1 + self.inflation_rate) ** (year - 1)
        return base_fixed * escalation
    
    def _calculate_variable_opex(self, year):
        """Calculate annual variable OPEX based on battery age"""
        if year <= 5:
            base_var = 0.1
        elif year <= 10:
            base_var = 0.24
        elif year <= 15:
            base_var = 0.40
        else:
            base_var = 0.55
        
        escalation = (1 + self.inflation_rate) ** (year - 1)
        return base_var * escalation


class OverbuildApproach(BESSProject):
    """APPROACH 1: Initial Overbuild - Build full capacity upfront"""
    
    def __init__(self, params):
        super().__init__(params)
        self.initial_capacity_mwh = 640  # Overbuild capacity
        self.initial_capex = 174.29  # $M
        self.approach_name = "Overbuild"
    
    def generate_capacity_schedule(self):
        """Generate SOH and capacity degradation schedule"""
        schedule = {}
        
        # Year 0
        schedule[0] = {'soh': 100.0, 'capacity': self.initial_capacity_mwh}
        
        # Years 1-20
        soh_values = [95.0, 92.5, 90.0, 87.5, 85.0, 82.5, 80.0, 77.5, 75.0, 72.5,
                     70.0, 67.5, 65.0, 62.5, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0]
        
        for year in range(1, 21):
            soh = soh_values[year - 1]
            capacity = self.initial_capacity_mwh * (soh / 100.0)
            schedule[year] = {'soh': soh, 'capacity': capacity}
        
        return schedule
    
    def get_capex_schedule(self):
        """Return CAPEX schedule (Year 0 only)"""
        return {0: -self.initial_capex}
    
    def calculate_metrics(self, cash_flow_df, capex_schedule):
        """Calculate key project metrics"""
        # Add CAPEX to cash flows
        cf_with_capex = cash_flow_df.copy()
        cf_with_capex['capex'] = cf_with_capex['year'].map(capex_schedule).fillna(0)
        cf_with_capex['net_cash_flow_with_capex'] = cf_with_capex['net_cash_flow'] + cf_with_capex['capex']
        cf_with_capex['cumulative_cf'] = cf_with_capex['net_cash_flow_with_capex'].cumsum()
        cf_with_capex['pv_cf_with_capex'] = cf_with_capex['net_cash_flow_with_capex'] * cf_with_capex['discount_factor']
        cf_with_capex['cumulative_pv'] = cf_with_capex['pv_cf_with_capex'].cumsum()
        
        metrics = {
            'approach': self.approach_name,
            'initial_capex': self.initial_capex,
            'total_capex_nominal': self.initial_capex,
            'total_capex_npv': self.initial_capex,
            'total_opex_nominal': cf_with_capex['total_opex'].sum(),
            'total_opex_npv': (cf_with_capex['total_opex'] * cf_with_capex['discount_factor']).sum(),
            'total_revenue_nominal': cf_with_capex['total_revenue'].sum(),
            'total_revenue_npv': (cf_with_capex['total_revenue'] * cf_with_capex['discount_factor']).sum(),
            'cumulative_cf_nominal': cf_with_capex['cumulative_cf'].iloc[-1],
            'cumulative_pv': cf_with_capex['cumulative_pv'].iloc[-1],
            'payback_period': self._calculate_payback(cf_with_capex),
            'total_energy_mwh': (cf_with_capex['available_capacity'].sum() * 365) / 1_000_000,
            'average_capacity': cf_with_capex['available_capacity'].mean(),
            'utilization_rate': (cf_with_capex['available_capacity'].mean() / self.initial_capacity_mwh) * 100,
        }
        
        # LCOS Calculation
        levelized_cost = ((metrics['total_capex_npv'] + metrics['total_opex_npv']) * 1_000_000) / \
                        (metrics['total_energy_mwh'] * 1_000_000)
        metrics['lcos'] = levelized_cost
        metrics['lcos_kwh'] = levelized_cost / 1000
        
        return metrics, cf_with_capex
    
    def _calculate_payback(self, cf_df):
        """Calculate payback period in years"""
        positive_mask = cf_df['cumulative_cf'] >= 0
        if positive_mask.any():
            return cf_df[positive_mask]['year'].min()
        return None


class StagedAugmentationApproach(BESSProject):
    """APPROACH 2: Staged Augmentation - Smaller initial build + augmentations at Year 7 & 14"""
    
    def __init__(self, params):
        super().__init__(params)
        self.initial_capacity_mwh = 430  # Staged initial capacity
        self.initial_capex = 117.10  # $M
        self.aug1_capacity = 60  # MWh at Year 7
        self.aug1_cost_nominal = 12.39  # $M
        self.aug1_cost_pv = 7.72  # $M PV
        self.aug2_capacity = 50  # MWh at Year 14
        self.aug2_cost_nominal = 8.07  # $M
        self.aug2_cost_pv = 3.13  # $M PV
        self.approach_name = "Staged Augmentation"
    
    def generate_capacity_schedule(self):
        """Generate SOH and capacity degradation with augmentations"""
        schedule = {}
        
        # Year 0
        schedule[0] = {'soh': 100.0, 'capacity': self.initial_capacity_mwh, 'augmentation': 0}
        
        soh_values = [95.0, 92.5, 90.0, 87.5, 85.0, 82.5, 80.0, 77.5, 75.0, 72.5,
                     70.0, 67.5, 65.0, 62.5, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0]
        
        augmentation_capacity = 0
        
        for year in range(1, 21):
            soh = soh_values[year - 1]
            
            # Augmentation events
            if year == 7:
                augmentation_capacity += self.aug1_capacity
            elif year == 14:
                augmentation_capacity += self.aug2_capacity
            
            base_capacity = self.initial_capacity_mwh * (soh / 100.0)
            total_capacity = base_capacity + augmentation_capacity
            
            schedule[year] = {
                'soh': soh,
                'capacity': total_capacity,
                'augmentation': augmentation_capacity
            }
        
        return schedule
    
    def get_capex_schedule(self):
        """Return CAPEX schedule (Year 0, 7, 14)"""
        return {0: -self.initial_capex, 7: -self.aug1_cost_nominal, 14: -self.aug2_cost_nominal}
    
    def get_capex_pv_schedule(self):
        """Return CAPEX schedule in PV terms"""
        return {0: -self.initial_capex, 7: -self.aug1_cost_pv, 14: -self.aug2_cost_pv}
    
    def calculate_metrics(self, cash_flow_df, capex_schedule):
        """Calculate key project metrics"""
        cf_with_capex = cash_flow_df.copy()
        cf_with_capex['capex'] = cf_with_capex['year'].map(capex_schedule).fillna(0)
        cf_with_capex['net_cash_flow_with_capex'] = cf_with_capex['net_cash_flow'] + cf_with_capex['capex']
        cf_with_capex['cumulative_cf'] = cf_with_capex['net_cash_flow_with_capex'].cumsum()
        cf_with_capex['pv_cf_with_capex'] = cf_with_capex['net_cash_flow_with_capex'] * cf_with_capex['discount_factor']
        cf_with_capex['cumulative_pv'] = cf_with_capex['pv_cf_with_capex'].cumsum()
        
        total_capex_nominal = self.initial_capex + self.aug1_cost_nominal + self.aug2_cost_nominal
        total_capex_npv = self.initial_capex + self.aug1_cost_pv + self.aug2_cost_pv
        
        metrics = {
            'approach': self.approach_name,
            'initial_capex': self.initial_capex,
            'total_capex_nominal': total_capex_nominal,
            'total_capex_npv': total_capex_npv,
            'total_opex_nominal': cf_with_capex['total_opex'].sum(),
            'total_opex_npv': (cf_with_capex['total_opex'] * cf_with_capex['discount_factor']).sum(),
            'total_revenue_nominal': cf_with_capex['total_revenue'].sum(),
            'total_revenue_npv': (cf_with_capex['total_revenue'] * cf_with_capex['discount_factor']).sum(),
            'cumulative_cf_nominal': cf_with_capex['cumulative_cf'].iloc[-1],
            'cumulative_pv': cf_with_capex['cumulative_pv'].iloc[-1],
            'payback_period': self._calculate_payback(cf_with_capex),
            'total_energy_mwh': (cf_with_capex['available_capacity'].sum() * 365) / 1_000_000,
            'average_capacity': cf_with_capex['available_capacity'].mean(),
            'utilization_rate': (cf_with_capex['available_capacity'].mean() / self.initial_capacity_mwh) * 100,
        }
        
        # LCOS Calculation
        levelized_cost = ((metrics['total_capex_npv'] + metrics['total_opex_npv']) * 1_000_000) / \
                        (metrics['total_energy_mwh'] * 1_000_000)
        metrics['lcos'] = levelized_cost
        metrics['lcos_kwh'] = levelized_cost / 1000
        
        return metrics, cf_with_capex
    
    def _calculate_payback(self, cf_df):
        """Calculate payback period in years"""
        positive_mask = cf_df['cumulative_cf'] >= 0
        if positive_mask.any():
            return cf_df[positive_mask]['year'].min()
        return None


# ============================================================================
# STREAMLIT INTERFACE
# ============================================================================

def main():
    st.title("⚡ BESS LCOS Analysis Tool")
    st.subheader("Energy Storage Project Economic Comparison")
    st.markdown("Overbuild vs. Staged Augmentation Approaches")
    
    # Sidebar - Input Parameters
    with st.sidebar:
        st.header("📊 Project Parameters")
        
        with st.expander("Basic Settings", expanded=True):
            power_mw = st.number_input("Power Capacity (MW)", value=100, min_value=1, max_value=1000)
            energy_mwh = st.number_input("Energy Capacity (MWh)", value=400, min_value=1, max_value=5000)
            project_life = st.number_input("Project Life (years)", value=20, min_value=5, max_value=30)
        
        with st.expander("Financial Parameters"):
            discount_rate = st.slider("Discount Rate (%)", min_value=1, max_value=15, value=7, step=1) / 100
            inflation_rate = st.slider("Inflation Rate (%)", min_value=0, max_value=5, value=2, step=1) / 100
        
        st.markdown("---")
        
        # Run Analysis Button
        run_analysis = st.button("🚀 Run Analysis", use_container_width=True)
    
    # Main Content
    if run_analysis:
        # Initialize parameters
        params = {
            'power_mw': power_mw,
            'energy_mwh': energy_mwh,
            'project_life': project_life,
            'discount_rate': discount_rate,
            'inflation_rate': inflation_rate,
        }
        
        # Create project instances
        overbuild = OverbuildApproach(params)
        staged = StagedAugmentationApproach(params)
        
        # Generate capacity schedules
        overbuild_schedule = overbuild.generate_capacity_schedule()
        staged_schedule = staged.generate_capacity_schedule()
        
        # Calculate cash flows
        overbuild_cf = overbuild.calculate_cash_flows(overbuild_schedule)
        staged_cf = staged.calculate_cash_flows(staged_schedule)
        
        # Calculate metrics
        overbuild_metrics, overbuild_cf_full = overbuild.calculate_metrics(
            overbuild_cf, overbuild.get_capex_schedule()
        )
        staged_metrics, staged_cf_full = staged.calculate_metrics(
            staged_cf, staged.get_capex_pv_schedule()
        )
        
        # Display Executive Summary
        st.header("📈 Executive Summary Comparison")
        
        # Key Metrics Comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Project NPV @ 7%",
                f"${overbuild_metrics['cumulative_pv']:.2f}M",
                delta=f"{staged_metrics['cumulative_pv'] - overbuild_metrics['cumulative_pv']:.2f}M",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "LCOS ($/MWh)",
                f"${overbuild_metrics['lcos']:.2f}",
                delta=f"{overbuild_metrics['lcos'] - staged_metrics['lcos']:.2f}",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Total CAPEX (NPV)",
                f"${overbuild_metrics['total_capex_npv']:.2f}M",
                delta=f"{staged_metrics['total_capex_npv'] - overbuild_metrics['total_capex_npv']:.2f}M",
                delta_color="inverse"
            )
        
        # Detailed Comparison Table
        st.subheader("💰 Financial Metrics Comparison")
        
        comparison_data = {
            'Metric': [
                'Initial CAPEX ($M)',
                'Total CAPEX NPV ($M)',
                'Total OPEX Nominal ($M)',
                'Total OPEX NPV ($M)',
                'Total Revenue NPV ($M)',
                'Project NPV @ 7% ($M)',
                'Cumulative Cash Flow ($M)',
                'Payback Period (Years)',
            ],
            'Overbuild': [
                f"${overbuild_metrics['initial_capex']:.2f}",
                f"${overbuild_metrics['total_capex_npv']:.2f}",
                f"${overbuild_metrics['total_opex_nominal']:.2f}",
                f"${overbuild_metrics['total_opex_npv']:.2f}",
                f"${overbuild_metrics['total_revenue_npv']:.2f}",
                f"${overbuild_metrics['cumulative_pv']:.2f}",
                f"${overbuild_metrics['cumulative_cf_nominal']:.2f}",
                str(overbuild_metrics['payback_period']) if overbuild_metrics['payback_period'] else "Never",
            ],
            'Staged Augmentation': [
                f"${staged_metrics['initial_capex']:.2f}",
                f"${staged_metrics['total_capex_npv']:.2f}",
                f"${staged_metrics['total_opex_nominal']:.2f}",
                f"${staged_metrics['total_opex_npv']:.2f}",
                f"${staged_metrics['total_revenue_npv']:.2f}",
                f"${staged_metrics['cumulative_pv']:.2f}",
                f"${staged_metrics['cumulative_cf_nominal']:.2f}",
                str(staged_metrics['payback_period']) if staged_metrics['payback_period'] else "Never",
            ],
            'Difference': [
                f"${staged_metrics['initial_capex'] - overbuild_metrics['initial_capex']:.2f}",
                f"${staged_metrics['total_capex_npv'] - overbuild_metrics['total_capex_npv']:.2f}",
                f"${staged_metrics['total_opex_nominal'] - overbuild_metrics['total_opex_nominal']:.2f}",
                f"${staged_metrics['total_opex_npv'] - overbuild_metrics['total_opex_npv']:.2f}",
                f"${staged_metrics['total_revenue_npv'] - overbuild_metrics['total_revenue_npv']:.2f}",
                f"${staged_metrics['cumulative_pv'] - overbuild_metrics['cumulative_pv']:.2f}",
                f"${staged_metrics['cumulative_cf_nominal'] - overbuild_metrics['cumulative_cf_nominal']:.2f}",
                "-",
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Capacity & Utilization Metrics
        st.subheader("⚙️ Capacity & Utilization Metrics")
        
        capacity_data = {
            'Metric': [
                'Initial Installed Capacity (MWh)',
                'Year 1 Available Capacity (MWh)',
                'Year 10 Available Capacity (MWh)',
                'Year 20 Available Capacity (MWh)',
                'Average Capacity (20 years)',
                'Capacity Utilization Rate (%)',
                'Total Energy Discharged (GWh)',
            ],
            'Overbuild': [
                f"{overbuild_schedule[0]['capacity']:.1f}",
                f"{overbuild_schedule[1]['capacity']:.1f}",
                f"{overbuild_schedule[10]['capacity']:.1f}",
                f"{overbuild_schedule[20]['capacity']:.1f}",
                f"{overbuild_metrics['average_capacity']:.1f}",
                f"{overbuild_metrics['utilization_rate']:.1f}",
                f"{overbuild_metrics['total_energy_mwh'] / 1000:.2f}",
            ],
            'Staged Augmentation': [
                f"{staged_schedule[0]['capacity']:.1f}",
                f"{staged_schedule[1]['capacity']:.1f}",
                f"{staged_schedule[10]['capacity']:.1f}",
                f"{staged_schedule[20]['capacity']:.1f}",
                f"{staged_metrics['average_capacity']:.1f}",
                f"{staged_metrics['utilization_rate']:.1f}",
                f"{staged_metrics['total_energy_mwh'] / 1000:.2f}",
            ]
        }
        
        capacity_df = pd.DataFrame(capacity_data)
        st.dataframe(capacity_df, use_container_width=True)
        
        # Visualizations
        st.subheader("📊 Charts & Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(
            ["NPV Comparison", "Capacity Over Time", "Cumulative Cash Flow", "LCOS Breakdown"]
        )
        
        with tab1:
            # NPV Comparison Chart
            npv_data = {
                'Approach': ['Overbuild', 'Staged Augmentation'],
                'NPV @ 7%': [overbuild_metrics['cumulative_pv'], staged_metrics['cumulative_pv']],
                'Colors': ['#e74c3c', '#2ecc71']
            }
            
            fig_npv = go.Figure(data=[
                go.Bar(
                    x=npv_data['Approach'],
                    y=npv_data['NPV @ 7%'],
                    marker_color=npv_data['Colors'],
                    text=[f"${v:.2f}M" for v in npv_data['NPV @ 7%']],
                    textposition='auto',
                )
            ])
            
            fig_npv.update_layout(
                title="Project NPV Comparison @ 7% Discount Rate",
                yaxis_title="NPV ($M)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_npv, use_container_width=True)
        
        with tab2:
            # Capacity Over Time
            capacity_years = list(range(0, 21))
            overbuild_cap = [overbuild_schedule[y]['capacity'] for y in capacity_years]
            staged_cap = [staged_schedule[y]['capacity'] for y in capacity_years]
            
            fig_cap = go.Figure()
            fig_cap.add_trace(go.Scatter(
                x=capacity_years, y=overbuild_cap,
                mode='lines+markers', name='Overbuild',
                line=dict(color='#e74c3c', width=2)
            ))
            fig_cap.add_trace(go.Scatter(
                x=capacity_years, y=staged_cap,
                mode='lines+markers', name='Staged Augmentation',
                line=dict(color='#2ecc71', width=2)
            ))
            
            fig_cap.update_layout(
                title="Available Capacity Over 20 Years",
                xaxis_title="Year",
                yaxis_title="Available Capacity (MWh)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_cap, use_container_width=True)
        
        with tab3:
            # Cumulative Cash Flow
            fig_cf = go.Figure()
            fig_cf.add_trace(go.Scatter(
                x=overbuild_cf_full['year'], y=overbuild_cf_full['cumulative_cf'],
                mode='lines+markers', name='Overbuild (Nominal)',
                line=dict(color='#e74c3c', width=2, dash='solid')
            ))
            fig_cf.add_trace(go.Scatter(
                x=overbuild_cf_full['year'], y=overbuild_cf_full['cumulative_pv'],
                mode='lines+markers', name='Overbuild (NPV @ 7%)',
                line=dict(color='#c0392b', width=2, dash='dash')
            ))
            fig_cf.add_trace(go.Scatter(
                x=staged_cf_full['year'], y=staged_cf_full['cumulative_cf'],
                mode='lines+markers', name='Staged (Nominal)',
                line=dict(color='#2ecc71', width=2, dash='solid')
            ))
            fig_cf.add_trace(go.Scatter(
                x=staged_cf_full['year'], y=staged_cf_full['cumulative_pv'],
                mode='lines+markers', name='Staged (NPV @ 7%)',
                line=dict(color='#27ae60', width=2, dash='dash')
            ))
            
            fig_cf.update_layout(
                title="Cumulative Cash Flow Comparison",
                xaxis_title="Year",
                yaxis_title="Cumulative Cash Flow ($M)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_cf, use_container_width=True)
        
        with tab4:
            # LCOS Breakdown
            lcos_data = {
                'Component': ['CAPEX (NPV)', 'OPEX (NPV)'],
                'Overbuild': [
                    overbuild_metrics['total_capex_npv'],
                    overbuild_metrics['total_opex_npv']
                ],
                'Staged': [
                    staged_metrics['total_capex_npv'],
                    staged_metrics['total_opex_npv']
                ]
            }
            
            fig_lcos = go.Figure()
            fig_lcos.add_trace(go.Bar(name='Overbuild', x=lcos_data['Component'], y=lcos_data['Overbuild']))
            fig_lcos.add_trace(go.Bar(name='Staged', x=lcos_data['Component'], y=lcos_data['Staged']))
            
            fig_lcos.update_layout(
                title="NPV Cost Components",
                yaxis_title="Cost ($M)",
                barmode='group',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_lcos, use_container_width=True)
        
        # Year-by-Year Details
        st.subheader("📋 Year-by-Year Cash Flow Analysis")
        
        detail_tab1, detail_tab2 = st.tabs(["Overbuild", "Staged Augmentation"])
        
        with detail_tab1:
            st.write("**Overbuild Approach - 20 Year Cash Flow**")
            display_df = overbuild_cf_full[[
                'year', 'calendar_year', 'soh', 'available_capacity',
                'capacity_revenue', 'energy_revenue', 'total_revenue',
                'total_opex', 'net_cash_flow', 'cumulative_cf'
            ]].copy()
            
            display_df.columns = [
                'Year', 'Calendar', 'SOH %', 'Capacity (MWh)',
                'Cap Rev ($M)', 'Energy Rev ($M)', 'Total Rev ($M)',
                'OPEX ($M)', 'Net CF ($M)', 'Cum CF ($M)'
            ]
            
            st.dataframe(display_df, use_container_width=True, height=400)
        
        with detail_tab2:
            st.write("**Staged Augmentation Approach - 20 Year Cash Flow**")
            display_df = staged_cf_full[[
                'year', 'calendar_year', 'soh', 'available_capacity',
                'capacity_revenue', 'energy_revenue', 'total_revenue',
                'total_opex', 'net_cash_flow', 'cumulative_cf'
            ]].copy()
            
            display_df.columns = [
                'Year', 'Calendar', 'SOH %', 'Capacity (MWh)',
                'Cap Rev ($M)', 'Energy Rev ($M)', 'Total Rev ($M)',
                'OPEX ($M)', 'Net CF ($M)', 'Cum CF ($M)'
            ]
            
            st.dataframe(display_df, use_container_width=True, height=400)
        
        # Conclusion
        st.markdown("---")
        st.subheader("🎯 Analysis Conclusion")
        
        npv_diff = staged_metrics['cumulative_pv'] - overbuild_metrics['cumulative_pv']
        capex_diff = overbuild_metrics['total_capex_npv'] - staged_metrics['total_capex_npv']
        lcos_diff = overbuild_metrics['lcos'] - staged_metrics['lcos']
        utilization_diff = staged_metrics['utilization_rate'] - overbuild_metrics['utilization_rate']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            ### Staged Augmentation Advantages:
            
            ✅ **NPV Advantage:** ${npv_diff:.2f}M ({(npv_diff / abs(overbuild_metrics['cumulative_pv']) * 100):.1f}% better)
            
            ✅ **CAPEX Savings:** ${capex_diff:.2f}M ({(capex_diff / overbuild_metrics['total_capex_npv'] * 100):.1f}% reduction)
            
            ✅ **LCOS Reduction:** ${lcos_diff:.2f}/MWh ({(lcos_diff / overbuild_metrics['lcos'] * 100):.1f}% lower)
            
            ✅ **Higher Utilization:** {utilization_diff:.1f}% more efficient
            """)
        
        with col2:
            st.markdown(f"""
            ### Key Insights:
            
            • Staged approach requires ${staged_metrics['initial_capex']:.2f}M upfront (vs ${overbuild_metrics['initial_capex']:.2f}M)
            
            • Augmentations occur at Year 7 and 14 based on capacity needs
            
            • Technology risk reduced through staged approach
            
            • Better aligns capacity with actual degradation
            
            • Lower initial capital investment improves ROI
            """)


if __name__ == "__main__":
    main()
