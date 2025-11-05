import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="3D BESS Visualization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔌 Dual 55MW BESS - 3-Breaker Optimization System")
st.markdown("### Interactive 3D Visualization of Optimized Circuit Breaker Configuration")

st.sidebar.header("System Control Panel")

def create_3d_system_layout():
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[10],
        mode='markers+text',
        marker=dict(size=15, color='green', symbol='square'),
        text=['CB-A'],
        textposition='top center',
        name='CB-A'
    ))
    
    bus_a_x = np.linspace(0, 10, 50)
    bus_a_y = np.zeros(50)
    bus_a_z = 10 * np.ones(50)
    fig.add_trace(go.Scatter3d(
        x=bus_a_x, y=bus_a_y, z=bus_a_z,
        mode='lines',
        line=dict(color='red', width=5),
        name='Bus A (34.5kV)'
    ))
    
    tf_a_positions = np.linspace(0, 10, 11)
    fig.add_trace(go.Scatter3d(
        x=tf_a_positions, y=np.ones(11)*0.5, z=10 * np.ones(11),
        mode='markers+text',
        marker=dict(size=8, color='yellow', symbol='diamond'),
        text=[f'TF-A{i+1}' for i in range(11)],
        textposition='top center',
        name='Unit A Transformers'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0], y=[5], z=[10],
        mode='markers+text',
        marker=dict(size=15, color='green', symbol='square'),
        text=['CB-B'],
        textposition='top center',
        name='CB-B'
    ))
    
    bus_b_x = np.linspace(0, 10, 50)
    bus_b_y = 5 * np.ones(50)
    bus_b_z = 10 * np.ones(50)
    fig.add_trace(go.Scatter3d(
        x=bus_b_x, y=bus_b_y, z=bus_b_z,
        mode='lines',
        line=dict(color='red', width=5),
        name='Bus B (34.5kV)'
    ))
    
    tf_b_positions = np.linspace(0, 10, 11)
    fig.add_trace(go.Scatter3d(
        x=tf_b_positions, y=5 * np.ones(11), z=10 * np.ones(11),
        mode='markers+text',
        marker=dict(size=8, color='yellow', symbol='diamond'),
        text=[f'TF-B{i+1}' for i in range(11)],
        textposition='top center',
        name='Unit B Transformers'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[5], y=[2.5], z=[10],
        mode='markers+text',
        marker=dict(size=12, color='purple', symbol='square'),
        text=['Coupler'],
        textposition='top center',
        name='Bus Coupler'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[5, 5], y=[0, 5], z=[10, 10],
        mode='lines',
        line=dict(color='purple', width=2, dash='dash'),
        name='Coupler Connection'
    ))
    
    fig.update_layout(
        title="3D BESS System Architecture",
        scene=dict(
            xaxis=dict(title='Distance (m)'),
            yaxis=dict(title='Unit Separation (m)'),
            zaxis=dict(title='Elevation (m)')
        ),
        hovermode='closest',
        width=1000,
        height=700
    )
    
    return fig

def create_3d_breaker_model(breaker_name):
    fig = go.Figure()
    
    if "Open" in breaker_name or "Bus Coupler" in breaker_name:
        status_color = 'red'
        status_text = 'OPEN'
    else:
        status_color = 'green'
        status_text = 'CLOSED'
    
    fig.add_trace(go.Scatter3d(
        x=[0.75], y=[0.5], z=[1.25],
        mode='markers+text',
        marker=dict(size=20, color=status_color, symbol='diamond'),
        text=[status_text],
        name='Status'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0.3], y=[0.3], z=[0.5],
        mode='markers+text',
        marker=dict(size=15, color='blue', symbol='square'),
        text=['Relay'],
        name='Protection Relay'
    ))
    
    fig.update_layout(
        title=f"Circuit Breaker: {breaker_name}",
        width=800,
        height=600
    )
    
    return fig

def create_power_flow_diagram(scenario):
    fig = go.Figure()
    
    if scenario == "Normal Operation":
        fig.add_trace(go.Scatter(
            x=[0, 1, 2, 3],
            y=[2, 2, 2, 2],
            mode='lines+markers+text',
            line=dict(color='orange', width=4),
            marker=dict(size=10),
            text=['GRID', 'POI-A', 'CB-A', '55 MW'],
            textposition='top center',
            name='Unit A'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1, 2, 3],
            y=[0, 0, 0, 0],
            mode='lines+markers+text',
            line=dict(color='orange', width=4),
            marker=dict(size=10),
            text=['GRID', 'POI-B', 'CB-B', '55 MW'],
            textposition='top center',
            name='Unit B'
        ))
        
        fig.add_annotation(
            x=3.5, y=1,
            text='<b>TOTAL: 110 MW</b>',
            showarrow=False,
            font=dict(size=16, color='green')
        )
        
        total_mw = 110
        unit_a = 55
        unit_b = 55
    
    elif scenario == "Unit A Offline":
        fig.add_trace(go.Scatter(
            x=[0, 1, 2, 3],
            y=[0, 0, 0, 0],
            mode='lines+markers+text',
            line=dict(color='orange', width=4),
            marker=dict(size=10),
            text=['GRID', 'POI-B', 'CB-B', '55 MW'],
            textposition='top center',
            name='Unit B'
        ))
        
        fig.add_annotation(
            x=3.5, y=1,
            text='<b>TOTAL: 55 MW</b><br>(50% capacity)',
            showarrow=False,
            font=dict(size=14, color='orange')
        )
        
        total_mw = 55
        unit_a = 0
        unit_b = 55
    
    else:
        fig.add_annotation(
            x=1.5, y=1,
            text='<b>Emergency Mode</b><br>Unit B supplies both<br>TOTAL: 55 MW',
            showarrow=False,
            font=dict(size=14, color='green')
        )
        
        total_mw = 55
        unit_a = 27.5
        unit_b = 27.5
    
    fig.update_layout(
        title=f"Power Flow - {scenario}",
        xaxis=dict(title='System Path', showgrid=False),
        yaxis=dict(title='Unit Distribution', showgrid=False),
        hovermode='closest',
        height=400,
        width=1000,
        showlegend=True
    )
    
    return fig, total_mw, unit_a, unit_b

def create_protection_diagram(fault):
    if fault == "Normal Operation":
        response = "All systems normal - All relays armed"
        impact = "0%"
        status = "100% OPERATIONAL"
        status_color = "green"
    elif fault == "Single Transformer Fault":
        response = "TF-A5 internal fuse blown in 100ms"
        impact = "9%"
        status = "91% operational"
        status_color = "orange"
    elif fault == "Unit A Complete Failure":
        response = "CB-A protection relay trips in 5 cycles"
        impact = "50%"
        status = "50% (Unit B only)"
        status_color = "red"
    else:
        response = "Both units trip - Grid protection activates"
        impact = "100%"
        status = "System isolated"
        status_color = "red"
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Protection Response'],
        y=[100],
        marker_color=status_color,
        text=[response],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Protection Coordination - {fault}",
        height=400,
        width=800,
        showlegend=False
    )
    
    return fig, impact, status

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏗️ System Layout",
    "⚡ Circuit Breakers",
    "🔄 Power Flow",
    "🛡️ Protection",
    "📊 Specifications"
])

with tab1:
    st.header("3D System Architecture")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = create_3d_system_layout()
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Legend")
        st.markdown("""
        **Red:** 34.5 kV HV
        
        **Green:** Breakers
        
        **Yellow:** Transformers
        
        **Purple:** Coupler
        """)
        
        st.subheader("Components")
        st.metric("Total Breakers", "3")
        st.metric("Total Transformers", "22")
        st.metric("Total Power", "110 MW")

with tab2:
    st.header("Circuit Breaker Specifications")
    
    breaker_type = st.selectbox(
        "Select Circuit Breaker:",
        ["CB-A (Unit A Main)", "CB-B (Unit B Main)", "Bus Coupler CB"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = create_3d_breaker_model(breaker_type)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if breaker_type == "CB-A (Unit A Main)":
            st.subheader("CB-A Specifications")
            specs = {
                "Current": "1200 A",
                "Breaking": "16 kA",
                "Voltage": "34.5 kV",
                "Type": "Vacuum",
                "Status": "CLOSED",
                "Function": "Unit A",
                "Cost": "$40,000"
            }
        elif breaker_type == "CB-B (Unit B Main)":
            st.subheader("CB-B Specifications")
            specs = {
                "Current": "1200 A",
                "Breaking": "16 kA",
                "Voltage": "34.5 kV",
                "Type": "Vacuum",
                "Status": "CLOSED",
                "Function": "Unit B",
                "Cost": "$40,000"
            }
        else:
            st.subheader("Bus Coupler CB")
            specs = {
                "Current": "1200 A",
                "Breaking": "16 kA",
                "Voltage": "34.5 kV",
                "Type": "Vacuum",
                "Status": "OPEN",
                "Function": "Emergency Tie",
                "Cost": "$45,000"
            }
        
        for key, value in specs.items():
            st.metric(key, value)
        
        st.subheader("Protection Relays")
        st.checkbox("50/51 Overcurrent", value=True, disabled=True)
        st.checkbox("27 Voltage Loss", value=True, disabled=True)
        st.checkbox("81 Frequency", value=True, disabled=True)
        if breaker_type == "Bus Coupler CB":
            st.checkbox("Sync-Check", value=True, disabled=True)

with tab3:
    st.header("Power Flow Visualization")
    
    scenario = st.radio(
        "Select Operating Scenario:",
        ["Normal Operation", "Unit A Offline", "Emergency Mode"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, total_mw, unit_a, unit_b = create_power_flow_diagram(scenario)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Power Distribution")
        
        if scenario == "Normal Operation":
            st.success("✓ Both units operating")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Unit A", f"{unit_a} MW")
            with col_b:
                st.metric("Unit B", f"{unit_b} MW")
            st.metric("Total to Grid", f"{total_mw} MW")
            st.info("Bus Coupler: OPEN")
        
        elif scenario == "Unit A Offline":
            st.warning("⚠ Unit A offline")
            st.metric("Unit A", f"{unit_a} MW")
            st.metric("Unit B", f"{unit_b} MW")
            st.metric("Total to Grid", f"{total_mw} MW")
            st.info("Bus Coupler: AVAILABLE")
        
        else:
            st.info("🔄 Emergency Mode Active")
            st.metric("Unit A", f"{unit_a} MW (Backup)")
            st.metric("Unit B", f"{unit_b} MW (Primary)")
            st.metric("Total to Grid", f"{total_mw} MW")
            st.success("Bus Coupler: CLOSED")

with tab4:
    st.header("Protection Coordination")
    
    fault = st.selectbox(
        "Select Fault Scenario:",
        [
            "Normal Operation",
            "Single Transformer Fault",
            "Unit A Complete Failure",
            "Grid Voltage Loss"
        ]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, impact, status = create_protection_diagram(fault)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fault Response")
        
        if fault == "Normal Operation":
            st.success("✓ All systems normal")
        elif fault == "Single Transformer Fault":
            st.warning("⚠ Single TF fault")
            st.metric("Capacity Loss", impact)
            st.metric("System Status", status)
        elif fault == "Unit A Complete Failure":
            st.error("Unit A offline")
            st.metric("Capacity Loss", impact)
            st.metric("System Status", status)
        else:
            st.error("Grid disconnected")
            st.metric("Capacity Loss", impact)
            st.metric("System Status", status)

with tab5:
    st.header("Detailed Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Specifications")
        system_specs = pd.DataFrame({
            "Parameter": [
                "Total Capacity",
                "Total Energy",
                "Number of Units",
                "Transformer per Unit",
                "Circuit Breakers",
                "Operating Voltage",
                "Secondary Voltage",
                "Frequency"
            ],
            "Value": [
                "110 MW",
                "55 MWh",
                "2 (55 MW each)",
                "11 (5.14 MVA)",
                "3 + Emergency tie",
                "34.5 kV",
                "0.69 kV",
                "60 Hz"
            ]
        })
        st.dataframe(system_specs, use_container_width=True)
    
    with col2:
        st.subheader("Cost Breakdown")
        cost_data = pd.DataFrame({
            "Component": [
                "3 Circuit Breakers",
                "22 Transformers",
                "Bus Bars & Cables",
                "Grounding",
                "Installation",
                "TOTAL"
            ],
            "Cost": [
                "$125,000",
                "$1,980,000",
                "$200,000",
                "$40,000",
                "$430,000",
                "$2,775,000"
            ]
        })
        st.dataframe(cost_data, use_container_width=True)
    
    st.divider()
    
    st.subheader("Circuit Breaker Details")
    cb_specs = pd.DataFrame({
        "Breaker": ["CB-A", "CB-B", "Bus Coupler"],
        "Current": ["1200A", "1200A", "1200A"],
        "Breaking": ["16 kA", "16 kA", "16 kA"],
        "Voltage": ["34.5 kV", "34.5 kV", "34.5 kV"],
        "Status": ["CLOSED", "CLOSED", "OPEN"],
        "Cost": ["$40K", "$40K", "$45K"]
    })
    st.dataframe(cb_specs, use_container_width=True)
    
    st.subheader("Transformer Details")
    tf_specs = pd.DataFrame({
        "Item": [
            "Per Unit MVA",
            "Primary Voltage",
            "Secondary Voltage",
            "Impedance",
            "Internal Fuse",
            "Bushing Type",
            "Connection",
            "Total Units"
        ],
        "Value": [
            "5.14 MVA",
            "34.5 kV",
            "0.69 kV",
            "9%",
            "15 A",
            "Loop Feed / Radial",
            "Daisy-chain Series",
            "22"
        ]
    })
    st.dataframe(tf_specs, use_container_width=True)