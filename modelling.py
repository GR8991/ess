import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# App title
st.title("Battery Cell Modeling with SoC-dependent Series Resistance")

# --- SIDEBAR INPUTS ---

st.sidebar.header("Input Battery Parameters")
max_voltage = st.sidebar.number_input('Max Cell Voltage (V)', min_value=2.0, max_value=5.0, value=4.2)
min_voltage = st.sidebar.number_input('Min Cell Voltage (V)', min_value=2.0, max_value=5.0, value=3.0)
nom_voltage = st.sidebar.number_input('Nominal Cell Voltage (V)', min_value=2.0, max_value=5.0, value=3.7)

chemistry = st.sidebar.selectbox(
    "Battery Chemistry",
    options=["Lithium-ion", "Lead-acid", "Flow Battery", "Other"]
)

Rsd = st.sidebar.number_input('Self-Discharge Resistance Rsd (kOhms)', min_value=0.1, max_value=1000.0, value=100.0)
capacity = st.sidebar.number_input('Cell Capacity (Ah)', min_value=0.1, max_value=1000.0, value=50.0)

charge_current = st.sidebar.number_input('Charge Current (A)', min_value=0.1, max_value=100.0, value=10.0)
discharge_current = st.sidebar.number_input('Discharge Current (A)', min_value=0.1, max_value=100.0, value=10.0)

st.sidebar.header("Series Resistance Modeling")
Rcell = st.sidebar.number_input('Base Cell Resistance Rcell (mΩ)', min_value=0.1, max_value=100.0, value=7.2)
c = st.sidebar.number_input('SoC-Dependent Parameter c', min_value=0.1, max_value=100.0, value=31.42)

# --- DERIVED PARAMETERS ---

Rcell_ohm = Rcell / 1000  # convert mΩ to Ω

# Time arrays for Charge/Discharge
time_to_charge = capacity / charge_current
time_to_discharge = capacity / discharge_current

time_charge = np.linspace(0, time_to_charge, 100)
soc_charge = (time_charge / time_to_charge) * 100  # percent
soc_array_charge = soc_charge / 100            # normalized 0-1

time_discharge = np.linspace(0, time_to_discharge, 100)
soc_discharge = 100 - (time_discharge / time_to_discharge) * 100
soc_array_discharge = soc_discharge / 100      # normalized 0-1

# Cell voltage profiles (simplified linear)
voltage_charge = min_voltage + (soc_charge / 100) * (max_voltage - min_voltage)
voltage_discharge = max_voltage - ((100 - soc_discharge) / 100) * (max_voltage - min_voltage)

# --- SOC-DEPENDENT SERIES RESISTANCE ---
# Rser = Rcell * 1/(1 + c*SoC)
Rser_dynamic_charge = Rcell_ohm * (1 / (1 + c * soc_array_charge))
Rser_dynamic_discharge = Rcell_ohm * (1 / (1 + c * soc_array_discharge))

# --- VOLTAGE DROP WITH DYNAMIC Rser ---
voltage_drop_charge = charge_current * Rser_dynamic_charge
terminal_voltage_charge = voltage_charge - voltage_drop_charge

voltage_drop_discharge = discharge_current * Rser_dynamic_discharge
terminal_voltage_discharge = voltage_discharge - voltage_drop_discharge

# --- LAYOUT/OUTPUT ---
st.header("Battery Cell Overview")
st.write(f"**Chemistry:** {chemistry}")
st.write(f"**Voltage Range:** {min_voltage} V - {max_voltage} V (Nominal: {nom_voltage} V)")
st.write(f"**Capacity:** {capacity} Ah")
st.write(f"**Self-discharge resistance (Rsd):** {Rsd} kΩ")

# Plot: Series Resistance vs State of Charge
st.subheader("Series Resistance vs State of Charge")
fig_soc, ax_soc = plt.subplots()
ax_soc.plot(soc_charge, Rser_dynamic_charge * 1000, label='Rser (Charge)', color='orange') # convert Ω to mΩ
ax_soc.plot(soc_discharge, Rser_dynamic_discharge * 1000, label='Rser (Discharge)', color='green')
ax_soc.set_xlabel('State of Charge (%)')
ax_soc.set_ylabel('Series Resistance (mΩ)')
ax_soc.set_title('SoC-dependent Series Resistance')
ax_soc.legend()
ax_soc.grid(True, alpha=0.3)
st.pyplot(fig_soc)

# --- CHARGE CYCLE ---
st.subheader("Charge Cycle Modeling")
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(time_charge, soc_charge, 'b-', linewidth=2, label='SOC')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('State of Charge (%)')
ax1.set_title('Charge - State of Charge')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax2.plot(time_charge, voltage_charge, 'g-', linewidth=2, label='Ideal Voltage')
ax2.plot(time_charge, terminal_voltage_charge, 'r--', linewidth=2, label='Terminal Voltage (w/ Rser)')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Voltage (V)')
ax2.set_title('Charge - Cell Voltage')
ax2.legend()
ax2.axhline(y=max_voltage, color='k', linestyle=':', alpha=0.5)
ax2.grid(True, alpha=0.3)
st.pyplot(fig1)

# --- DISCHARGE CYCLE ---
st.subheader("Discharge Cycle Modeling")
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
ax3.plot(time_discharge, soc_discharge, 'b-', linewidth=2, label='SOC')
ax3.set_xlabel('Time (hours)')
ax3.set_ylabel('State of Charge (%)')
ax3.set_title('Discharge - State of Charge')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax4.plot(time_discharge, voltage_discharge, 'g-', linewidth=2, label='Ideal Voltage')
ax4.plot(time_discharge, terminal_voltage_discharge, 'r--', linewidth=2, label='Terminal Voltage (w/ Rser)')
ax4.set_xlabel('Time (hours)')
ax4.set_ylabel('Voltage (V)')
ax4.set_title('Discharge - Cell Voltage')
ax4.legend()
ax4.axhline(y=min_voltage, color='k', linestyle=':', alpha=0.5)
ax4.grid(True, alpha=0.3)
st.pyplot(fig2)

# --- EFFICIENCY / ENERGY METRICS (simple) ---
st.subheader("Performance Metrics")
energy_in_charge = charge_current * time_to_charge * np.mean(voltage_charge)  # Wh
energy_out_discharge = discharge_current * time_to_discharge * np.mean(voltage_discharge)  # Wh

loss_charge = np.sum((charge_current ** 2) * Rser_dynamic_charge * (time_to_charge/100))  # Wh (approx)
loss_discharge = np.sum((discharge_current ** 2) * Rser_dynamic_discharge * (time_to_discharge/100))  # Wh (approx)
total_loss = loss_charge + loss_discharge

round_trip_efficiency = (energy_out_discharge / energy_in_charge) * 100 if energy_in_charge > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Energy In (Charge)", f"{energy_in_charge:.2f} Wh")
with col2:
    st.metric("Energy Out (Discharge)", f"{energy_out_discharge:.2f} Wh")
with col3:
    st.metric("Total Loss", f"{total_loss:.2f} Wh")
with col4:
    st.metric("Round-Trip Efficiency", f"{round_trip_efficiency:.2f}%")

# --- SELF-DISCHARGE (optional, unchanged) ---
storage_days = st.slider('Storage Duration (days)', 1, 365, 30)
storage_time_hours = np.linspace(0, storage_days * 24, 100)
discharge_rate_constant = Rsd * 1000 * capacity
voltage_storage = max_voltage * np.exp(-storage_time_hours / discharge_rate_constant)
soc_storage = ((voltage_storage - min_voltage) / (max_voltage - min_voltage)) * 100
fig4, (ax7, ax8) = plt.subplots(1, 2, figsize=(12, 4))
ax7.plot(storage_time_hours / 24, soc_storage, 'b-', linewidth=2)
ax7.set_xlabel('Storage Time (days)')
ax7.set_ylabel('State of Charge (%)')
ax7.set_title(f'Self-Discharge Over {storage_days} Days')
ax7.grid(True, alpha=0.3)
ax8.plot(storage_time_hours / 24, voltage_storage, 'g-', linewidth=2)
ax8.axhline(y=min_voltage, color='r', linestyle='--', alpha=0.5)
ax8.set_xlabel('Storage Time (days)')
ax8.set_ylabel('Voltage (V)')
ax8.set_title(f'Voltage Loss Over {storage_days} Days')
ax8.grid(True, alpha=0.3)
st.pyplot(fig4)
soc_loss = 100 - soc_storage[-1]
st.write(f"**SOC Loss after {storage_days} days:** {soc_loss:.2f}%")
st.write(f"**Remaining Voltage:** {voltage_storage[-1]:.2f} V")

# --- END ---

st.info("This app includes SoC-dependent series resistance (Rser), improving realism for charge/discharge modelling. Adjust parameters in the sidebar to see dynamic results.")
