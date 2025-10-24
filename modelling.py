import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# App title
st.title("Battery Cell Modeling Visualization")

# Sidebar battery parameters
st.sidebar.header("Input Battery Parameters")
max_voltage = st.sidebar.number_input('Max Cell Voltage (V)', min_value=2.0, max_value=5.0, value=4.2)
min_voltage = st.sidebar.number_input('Min Cell Voltage (V)', min_value=2.0, max_value=5.0, value=3.0)
nom_voltage = st.sidebar.number_input('Nominal Cell Voltage (V)', min_value=2.0, max_value=5.0, value=3.7)

Rser = st.sidebar.number_input('Series Resistance Rser (Ohms)', min_value=0.001, max_value=1.0, value=0.05)
Rsd = st.sidebar.number_input('Self-Discharge Resistance Rsd (kOhms)', min_value=0.1, max_value=1000.0, value=100.0)

chemistry = st.sidebar.selectbox(
    "Battery Chemistry",
    options=["Lithium-ion", "Lead-acid", "Flow Battery", "Other"]
)

# Calculations
st.header("Battery Cell Overview")
st.write(f"**Chemistry:** {chemistry}")
st.write(f"**Voltage Range:** {min_voltage} V - {max_voltage} V (Nominal: {nom_voltage} V)")
st.write(f"**Rser:** {Rser} Ω | **Rsd:** {Rsd} kΩ")

# Efficiency calculation (simple estimation)
round_trip_eff = (1 - (Rser / (Rser + 0.01))) * 100  # simplistic model
st.subheader("Estimated Round-Trip Efficiency")
st.write(f"{round_trip_eff:.2f}%")

# Self-discharge rate (simplified calculation)
self_discharge_per_day = 100 / Rsd  # percent per day (dummy example)
st.subheader("Estimated Self-Discharge Rate")
st.write(f"{self_discharge_per_day:.2f}% per day")

# Voltage range plot
st.subheader("Voltage Range Visualization")
voltage_points = np.array([min_voltage, nom_voltage, max_voltage])
labels = ["Min", "Nom", "Max"]

fig, ax = plt.subplots()
ax.plot(labels, voltage_points, marker='o')
ax.set_ylabel('Voltage (V)')
ax.set_title('Cell Voltage Range')
st.pyplot(fig)

# Show resistance impact
st.subheader("Effect of Series Resistance on Efficiency")
rser_values = np.linspace(0.001, 0.5, 100)
efficiencies = (1 - (rser_values/(rser_values + 0.01))) * 100
fig2, ax2 = plt.subplots()
ax2.plot(rser_values, efficiencies)
ax2.set_xlabel("Rser (Ω)")
ax2.set_ylabel("Efficiency (%)")
ax2.set_title("Series Resistance vs. Efficiency")
st.pyplot(fig2)

st.info("For more complex chemistry models, equations must be adapted per chemistry (see your documentation).")
