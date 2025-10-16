# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go

# App title
st.title("⚡ Real-Time Circuit Breaker Trip Simulator")

st.markdown("""
Watch fault current evolve in real-time and see which protection zone trips the breaker.
""")

# Sidebar: Breaker Settings
st.sidebar.header("⚙️ Breaker Settings")
inst_pickup = st.sidebar.number_input("Instantaneous Pickup (kA)", value=30.0, step=1.0)
st_pickup = st.sidebar.number_input("Short-Time Pickup (kA)", value=10.0, step=1.0)
st_rating = st.sidebar.number_input("Short-Time Rating (kA)", value=25.0, step=1.0)
st_delay = st.sidebar.number_input("Short-Time Delay (s)", value=3.0, step=0.5)
lt_pickup = st.sidebar.number_input("Long-Time Pickup (A)", value=500.0, step=50.0)

# Sidebar: Fault Scenario
st.sidebar.header("🔥 Fault Scenario")
fault_start = st.sidebar.number_input("Fault Start Time (s)", value=1.0, step=0.1, min_value=0.1)
fault_level = st.sidebar.number_input("Fault Current (kA)", value=20.0, step=1.0)
duration = st.sidebar.number_input("Simulation Duration (s)", value=6.0, step=1.0)

# Start simulation button
if st.sidebar.button("▶️ Start Simulation"):
    # Initialize
    dt = 0.05  # time step (50 ms updates)
    times = np.arange(0, duration, dt)
    current_values = []
    
    # Placeholders
    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    trip_placeholder = st.empty()
    
    tripped = False
    trip_time = None
    trip_zone = None
    
    # Determine trip conditions
    if fault_level >= inst_pickup:
        trip_zone = "⚡ Zone 1: Instantaneous"
        trip_time = fault_start + 0.05  # trips almost instantly
    elif st_pickup <= fault_level < inst_pickup:
        trip_zone = "⏱️ Zone 2: Short-Time Delayed"
        trip_time = fault_start + st_delay
    elif (fault_level * 1000) > lt_pickup:
        trip_zone = "🐌 Zone 3: Long-Time Inverse"
        k = 0.14  # standard inverse constant
        Ipu = (fault_level * 1000) / lt_pickup
        trip_time = fault_start + k / (Ipu - 1) if Ipu > 1 else np.inf
    else:
        trip_zone = "✅ No Trip"
        trip_time = np.inf
    
    # Real-time animation loop
    for t in times:
        # Update current
        if t >= fault_start:
            I = fault_level
        else:
            I = 0.5  # normal load current
        
        current_values.append(I)
        
        # Check if tripped
        if t >= trip_time and not tripped:
            tripped = True
            status_placeholder.error(f"🔴 BREAKER TRIPPED at {t:.2f}s | {trip_zone}")
        
        # Update plot
        fig = go.Figure()
        
        # Current trace
        fig.add_trace(go.Scatter(
            x=times[:len(current_values)],
            y=current_values,
            mode='lines',
            name='Fault Current',
            line=dict(color='blue', width=3)
        ))
        
        # Protection zones
        fig.add_hline(y=inst_pickup, line_dash="dash", line_color="red", 
                      annotation_text="Instantaneous Pickup", annotation_position="right")
        fig.add_hline(y=st_pickup, line_dash="dash", line_color="orange",
                      annotation_text="Short-Time Pickup", annotation_position="right")
        fig.add_hline(y=lt_pickup/1000, line_dash="dash", line_color="green",
                      annotation_text="Long-Time Pickup", annotation_position="right")
        
        # Trip marker
        if tripped:
            fig.add_vline(x=trip_time, line_dash="dot", line_color="red", line_width=4,
                          annotation_text="TRIP", annotation_position="top")
        
        fig.update_layout(
            title="Real-Time Fault Current",
            xaxis_title="Time (s)",
            yaxis_title="Current (kA)",
            height=450,
            showlegend=True
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Status update
        if not tripped:
            status_placeholder.info(f"⏳ Time: {t:.2f}s | Current: {I:.1f} kA | Status: Monitoring...")
        
        time.sleep(dt)  # real-time delay
    
    # Final summary
    trip_placeholder.success("✅ Simulation Complete")

else:
    st.info("👈 Configure settings in the sidebar and click **Start Simulation**")
