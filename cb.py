# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import time

# App title
st.title("Circuit Breaker Trip Zone Visualization")

st.markdown("""
Visualize the three protection zones of a circuit breaker:
1. Instantaneous trip (Zone 1)
2. Short-time delayed trip (Zone 2)
3. Long-time inverse-time trip (Zone 3)
""")

# User inputs
freq = st.sidebar.number_input("System Frequency (Hz)", value=60, step=1)
inst_pickup = st.sidebar.number_input("Instantaneous Pickup (kA)", value=30.0, step=1.0)
st_pickup = st.sidebar.number_input("Short-Time Pickup (kA)", value=10.0, step=1.0)
st_rating = st.sidebar.number_input("Short-Time Rating (kA)", value=25.0, step=1.0)
st_delay = st.sidebar.number_input("Short-Time Delay (s)", value=3.0, step=0.5)
lt_pickup = st.sidebar.number_input("Long-Time Pickup (A)", value=500.0, step=50.0)
lt_curve = st.sidebar.selectbox("Inverse-Time Curve", ["Standard Inverse", "Very Inverse", "Extremely Inverse"])

# Generate time series for current waveform
duration = st.sidebar.number_input("Simulation Duration (s)", value=5.0, step=1.0)
fs = 200  # samples per second for display
times = np.linspace(0, duration, int(fs * duration))

# Simulate a fault current event
fault_start = st.sidebar.number_input("Fault Start Time (s)", value=1.0, step=0.5)
fault_level = st.sidebar.number_input("Fault Current (kA)", value=20.0, step=1.0)
current = np.zeros_like(times)
current[times >= fault_start] = fault_level

# Determine trip times
# Zone 1: instantaneous
z1_trip = fault_start if fault_level >= inst_pickup else np.inf

# Zone 2: short-time
z2_trip = (fault_start + st_delay) if st_pickup < fault_level <= st_rating else np.inf

# Zone 3: long-time inverse-time
# Simple inverse-time calculation: t = k * (I / Ipickup - 1)^-1
k_map = {"Standard Inverse": 0.14, "Very Inverse": 13.5, "Extremely Inverse": 80.0}
k = k_map[lt_curve]
Ipu = lt_pickup / 1000.0  # convert A to kA base
lt_trip = fault_start + k / (fault_level / Ipu - 1) if fault_level/ Ipu > 1 else np.inf

# Build DataFrame
df = pd.DataFrame({
    "Time (s)": times,
    "Current (kA)": current
})

# Plot waveform
base = alt.Chart(df).mark_line().encode(
    x="Time (s)",
    y=alt.Y("Current (kA)", scale=alt.Scale(domain=[0, max(current.max(), st_rating*1.1)]))
)

# Mark trip zones
zones = []
if z1_trip < np.inf:
    zones.append(alt.Chart(pd.DataFrame({"t": [z1_trip]}))
                 .mark_rule(color="red", strokeDash=[4,4])
                 .encode(x="t:Q"))
if z2_trip < np.inf:
    zones.append(alt.Chart(pd.DataFrame({"t": [z2_trip]}))
                 .mark_rule(color="orange", strokeDash=[4,4])
                 .encode(x="t:Q"))
if lt_trip < np.inf:
    zones.append(alt.Chart(pd.DataFrame({"t": [lt_trip]}))
                 .mark_rule(color="blue", strokeDash=[4,4])
                 .encode(x="t:Q"))

chart = base
for z in zones:
    chart += z

chart = chart.properties(
    width=700, height=400, title="Fault Current and Breaker Trip Points"
)

st.altair_chart(chart, use_container_width=True)

# Display trip times
st.markdown("**Trip Times:**")
st.write(f"Instantaneous Trip (Zone 1): {z1_trip if z1_trip<np.inf else 'Not triggered'} s")
st.write(f"Short-Time Trip (Zone 2): {z2_trip if z2_trip<np.inf else 'Not triggered'} s")
st.write(f"Long-Time Trip (Zone 3): {lt_trip if lt_trip<np.inf else 'Not triggered'} s")
