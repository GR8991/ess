import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# App title
st.title("Battery Cell Modeling with Charge/Discharge Cycles")

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

# Charge/Discharge parameters
st.sidebar.header("Charge/Discharge Parameters")
charge_current = st.sidebar.number_input('Charge Current (A)', min_value=0.1, max_value=100.0, value=10.0)
discharge_current = st.sidebar.number_input('Discharge Current (A)', min_value=0.1, max_value=100.0, value=10.0)
capacity = st.sidebar.number_input('Cell Capacity (Ah)', min_value=0.1, max_value=1000.0, value=50.0)

# Battery cell overview
st.header("Battery Cell Overview")
st.write(f"**Chemistry:** {chemistry}")
st.write(f"**Voltage Range:** {min_voltage} V - {max_voltage} V (Nominal: {nom_voltage} V)")
st.write(f"**Rser:** {Rser} Ω | **Rsd:** {Rsd} kΩ")
st.write(f"**Capacity:** {capacity} Ah")

# ============= CHARGE CYCLE MODELING =============
st.subheader("Charge Cycle Modeling")

# Time for full charge (hours)
time_to_charge = capacity / charge_current
time_charge = np.linspace(0, time_to_charge, 100)
soc_charge = (time_charge / time_to_charge) * 100  # State of Charge (%)

# Voltage during charge (simplified linear model)
voltage_charge = min_voltage + (soc_charge / 100) * (max_voltage - min_voltage)

# Apply voltage drop due to series resistance during charge
voltage_drop_charge = charge_current * Rser
terminal_voltage_charge = voltage_charge - voltage_drop_charge

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: State of Charge vs Time
ax1.plot(time_charge, soc_charge, 'b-', linewidth=2, label='SOC')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('State of Charge (%)')
ax1.set_title('Charge Cycle - State of Charge')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Cell Voltage vs Time
ax2.plot(time_charge, voltage_charge, 'g-', linewidth=2, label='Ideal Voltage')
ax2.plot(time_charge, terminal_voltage_charge, 'r--', linewidth=2, label='Terminal Voltage (with Rser drop)')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Voltage (V)')
ax2.set_title('Charge Cycle - Cell Voltage')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axhline(y=max_voltage, color='k', linestyle=':', alpha=0.5, label='Max Voltage')

st.pyplot(fig1)

# ============= DISCHARGE CYCLE MODELING =============
st.subheader("Discharge Cycle Modeling")

# Time for full discharge (hours)
time_to_discharge = capacity / discharge_current
time_discharge = np.linspace(0, time_to_discharge, 100)
soc_discharge = 100 - (time_discharge / time_to_discharge) * 100  # State of Charge (%)

# Voltage during discharge (simplified linear model)
voltage_discharge = max_voltage - ((100 - soc_discharge) / 100) * (max_voltage - min_voltage)

# Apply voltage drop due to series resistance during discharge
voltage_drop_discharge = discharge_current * Rser
terminal_voltage_discharge = voltage_discharge - voltage_drop_discharge

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 3: State of Charge vs Time
ax3.plot(time_discharge, soc_discharge, 'b-', linewidth=2, label='SOC')
ax3.set_xlabel('Time (hours)')
ax3.set_ylabel('State of Charge (%)')
ax3.set_title('Discharge Cycle - State of Charge')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Cell Voltage vs Time
ax4.plot(time_discharge, voltage_discharge, 'g-', linewidth=2, label='Ideal Voltage')
ax4.plot(time_discharge, terminal_voltage_discharge, 'r--', linewidth=2, label='Terminal Voltage (with Rser drop)')
ax4.set_xlabel('Time (hours)')
ax4.set_ylabel('Voltage (V)')
ax4.set_title('Discharge Cycle - Cell Voltage')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.axhline(y=min_voltage, color='k', linestyle=':', alpha=0.5, label='Min Voltage')

st.pyplot(fig2)

# ============= CHARGE/DISCHARGE CYCLE TOGETHER =============
st.subheader("Complete Charge-Discharge Cycle")

# Combine charge and discharge
total_time_charge_discharge = time_to_charge + time_to_discharge
time_combined = np.concatenate([time_charge, time_charge[-1] + time_discharge])
soc_combined = np.concatenate([soc_charge, soc_discharge])
voltage_combined = np.concatenate([voltage_charge, voltage_discharge])
terminal_voltage_combined = np.concatenate([terminal_voltage_charge, terminal_voltage_discharge])

fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 5: Complete SOC cycle
ax5.plot(time_combined, soc_combined, 'b-', linewidth=2)
ax5.fill_between(time_combined, soc_combined, alpha=0.3)
ax5.set_xlabel('Time (hours)')
ax5.set_ylabel('State of Charge (%)')
ax5.set_title('Complete Cycle - State of Charge')
ax5.grid(True, alpha=0.3)
ax5.axvline(x=time_to_charge, color='r', linestyle='--', alpha=0.5, label='Charge ends / Discharge starts')
ax5.legend()

# Plot 6: Complete voltage cycle
ax6.plot(time_combined, voltage_combined, 'g-', linewidth=2, label='Ideal Voltage')
ax6.plot(time_combined, terminal_voltage_combined, 'r--', linewidth=2, label='Terminal Voltage (with Rser drop)')
ax6.axhline(y=max_voltage, color='k', linestyle=':', alpha=0.5, label='Max/Min Voltage')
ax6.axhline(y=min_voltage, color='k', linestyle=':', alpha=0.5)
ax6.set_xlabel('Time (hours)')
ax6.set_ylabel('Voltage (V)')
ax6.set_title('Complete Cycle - Cell Voltage')
ax6.grid(True, alpha=0.3)
ax6.axvline(x=time_to_charge, color='r', linestyle='--', alpha=0.5)
ax6.legend()

st.pyplot(fig3)

# ============= EFFICIENCY CALCULATIONS =============
st.subheader("Efficiency Analysis")

# Energy calculations
energy_in_charge = charge_current * time_to_charge * np.mean(voltage_charge)  # Wh
energy_out_discharge = discharge_current * time_to_discharge * np.mean(voltage_discharge)  # Wh

# Losses due to series resistance
loss_charge = (charge_current ** 2) * Rser * time_to_charge  # Wh
loss_discharge = (discharge_current ** 2) * Rser * time_to_discharge  # Wh
total_loss = loss_charge + loss_discharge

# Round-trip efficiency
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

# ============= SELF-DISCHARGE MODELING =============
st.subheader("Self-Discharge Over Time (Storage)")

storage_days = st.slider('Storage Duration (days)', 1, 365, 30)
storage_time_hours = np.linspace(0, storage_days * 24, 100)

# Self-discharge rate (simplified exponential decay)
# V_discharge = V_initial * exp(-t / (Rsd * C))
discharge_rate_constant = Rsd * 1000 * capacity  # time constant in hours
voltage_storage = max_voltage * np.exp(-storage_time_hours / discharge_rate_constant)
soc_storage = ((voltage_storage - min_voltage) / (max_voltage - min_voltage)) * 100

fig4, (ax7, ax8) = plt.subplots(1, 2, figsize=(12, 4))

ax7.plot(storage_time_hours / 24, soc_storage, 'b-', linewidth=2)
ax7.set_xlabel('Storage Time (days)')
ax7.set_ylabel('State of Charge (%)')
ax7.set_title(f'Self-Discharge Over {storage_days} Days')
ax7.grid(True, alpha=0.3)

ax8.plot(storage_time_hours / 24, voltage_storage, 'g-', linewidth=2)
ax8.axhline(y=min_voltage, color='r', linestyle='--', alpha=0.5, label='Min Voltage (Dead)')
ax8.set_xlabel('Storage Time (days)')
ax8.set_ylabel('Voltage (V)')
ax8.set_title(f'Voltage Loss Over {storage_days} Days')
ax8.grid(True, alpha=0.3)
ax8.legend()

st.pyplot(fig4)

# Self-discharge loss
soc_loss = 100 - soc_storage[-1]
st.write(f"**SOC Loss after {storage_days} days:** {soc_loss:.2f}%")
st.write(f"**Remaining Voltage:** {voltage_storage[-1]:.2f} V")

# Info box
st.info("This app models charge, discharge, and self-discharge cycles. Customize parameters and see how Rser, Rsd, and current affect battery performance.")
