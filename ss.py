import streamlit as st
import numpy as np

st.title("PGR-Ud & Uq Calculator for PCS (690V Base)")

st.sidebar.header("Input Power Requirements")
P_pcs = st.sidebar.number_input("Active Power P_pcs (MW)", value=50.0, format="%.3f")
Q_pcs = st.sidebar.number_input("Reactive Power Q_pcs (MVAR)", value=20.0, format="%.3f")
Voltage = st.sidebar.number_input("Base voltage (V)", value=20.0, format="%.3f")

st.header("Calculation Steps")

# Step 1: Apparent Power and Power Angle
S_pcs = np.sqrt(P_pcs**2 + Q_pcs**2)
sin_phi = Q_pcs / S_pcs if S_pcs != 0 else 0
phi_rad = np.arcsin(abs(sin_phi))
if sin_phi < 0:
    phi_rad = -phi_rad
phi_deg = phi_rad * 180 / np.pi

st.subheader("1. Apparent Power & Power Angle")
st.write(f"S_pcs = √({P_pcs:.3f}² + {Q_pcs:.3f}²) = {S_pcs:.3f} MVA")
st.write(f"sin(phi) = {sin_phi:.6f}")
st.write(f"phi angle = {phi_deg:.2f}°")

# Step 2: Determine Operating Mode
if abs(Q_pcs) > 20:
    U_terminal_pu = 0.94
    k_ratio = 0.22
    mode = "High Reactive Power Mode"
else:
    U_terminal_pu = 0.885
    k_ratio = 0.45
    mode = "Low Reactive Power Mode"

st.subheader("2. Operating Mode")
st.write(f"Mode: {mode}")
st.write(f"U_terminal = {U_terminal_pu:.3f} pu, k_ratio = {k_ratio:.2f}")

# Step 3: Voltage Angle
delta_rad = phi_rad * k_ratio
delta_deg = delta_rad * 180 / np.pi

st.subheader("3. Voltage Angle")
st.write(f"delta = phi × k_ratio = {phi_deg:.2f}° × {k_ratio:.2f} = {delta_deg:.2f}°")

# Step 4: DQ Components
Ud_pu = U_terminal_pu * np.cos(delta_rad)
Uq_pu = U_terminal_pu * np.sin(delta_rad)

st.subheader("4. DQ Components (per unit)")
st.write(f"Ud = {Ud_pu:.6f} pu")
st.write(f"Uq = {Uq_pu:.6f} pu")

# Step 5: Convert to Voltage
base_voltage = Voltage
Ud_volts = Ud_pu * base_voltage
Uq_volts = Uq_pu * base_voltage

st.subheader("5. Actual Voltage (V)")
st.write(f"Ud = {Ud_volts:.2f} V")
st.write(f"Uq = {Uq_volts:.2f} V")

# Verification
U_calc_pu = np.sqrt(Ud_pu**2 + Uq_pu**2)
st.subheader("6. Verification")
st.write(f"Terminal voltage calc = √(Ud² + Uq²) = {U_calc_pu:.6f} pu")
st.write(f"Expected terminal voltage = {U_terminal_pu:.3f} pu")
st.write(f"Difference = {abs(U_calc_pu - U_terminal_pu):.6f} pu")
