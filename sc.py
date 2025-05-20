import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Transformer Short Circuit Calculator",
    page_icon="⚡",
    layout="centered"
)

st.title("PGR_Transformer Short Circuit Calculator 🔌")
st.markdown("Calculate fault current from transformer rating and impedance.")

#st.set_page_config(page_title="Transformer Short‑Circuit Calculator", layout="wide")

# ─────────────────── Sidebar Inputs ────────────────────
st.sidebar.header("Input Parameters")
Z_percent = st.sidebar.number_input("Impedance (%)", min_value=0.1, value=6.0, step=0.1)
S_kVA      = st.sidebar.number_input("Power Rating (kVA)", min_value=1.0, value=3600.0, step=1.0)
V_HV       = st.sidebar.number_input("HV Voltage (V)", 100.0, 1_000_000.0, 33_000.0, 100.0)
V_LV       = st.sidebar.number_input("LV Voltage (V)", 100.0, 1_000_000.0,   950.0, 50.0)

auto_zero_factor = 0.85   # for Dyn transformers, per IEC example

fault_side = st.sidebar.radio("Fault Side", ["HV Side", "LV Side"])

# ─────────────────── Base Values ───────────────────────
V_base = V_HV if fault_side == "HV Side" else V_LV   # volts
Z1 = Z_percent / 100
Z2 = Z1
Z0 = auto_zero_factor * Z1

# Base current (A)
I_base = (S_kVA * 1e3) / (math.sqrt(3) * V_base)

# Fault currents (A)
S_va = S_kVA * 1e3
I_3ph = S_va / (math.sqrt(3) * V_base * Z1)
I_pp  = S_va / ((Z1 + Z2) * V_base)
I_pn  = (3 * S_va) / (math.sqrt(3) * V_base * (Z0 + Z1 + Z2))
I_pe  = (3 * S_va) / (math.sqrt(3) * V_base * (Z0 + Z1 + Z2))

# ─────────────────── Results Display ───────────────────
left, right = st.columns(2)
with left:
    st.header(f"Sequence Impedances (pu)")
    st.write(f"**Z₁ (positive):** {Z1:.4f}")
    st.write(f"**Z₂ (negative):** {Z2:.4f}")
    st.write(f"**Z₀ (zero):** {Z0:.4f}")

with right:
    st.header(f"Fault Currents on {fault_side}")
    st.write(f"3‑Φ fault: **{I_3ph/1e3:.2f} kA**")
    st.write(f"L‑L fault: **{I_pp/1e3:.2f} kA**")
    st.write(f"L‑N fault: **{I_pn/1e3:.2f} kA**")
    st.write(f"L‑E fault: **{I_pe/1e3:.2f} kA**")

# ─────────────────── Phasor Diagram ────────────────────
# We visualise sequence voltages for a 1 pu sequence current (angle 0°)
I1 = 1.0
angle_I1 = 0.0
Z1_cplx = Z1                          # purely mag (no angle, assume R negligible)
Z2_cplx = Z2 * np.exp(1j * math.pi)   # 180° shift illustrative
Z0_cplx = Z0 * np.exp(-1j * 2*math.pi/3)  # -120° shift
V1 = I1 * Z1_cplx
V2 = I1 * Z2_cplx
V0 = I1 * Z0_cplx

fig, ax = plt.subplots(figsize=(5,5))
origin = [0], [0]
colors = {"V₁ (positive)":"tab:blue", "V₂ (negative)":"tab:red", "V₀ (zero)":"tab:green"}
for (label, vec), col in zip({"V₁ (positive)":V1, "V₂ (negative)":V2, "V₀ (zero)":V0}.items(), colors.values()):
    ax.quiver(*origin, np.real(vec), np.imag(vec), angles='xy', scale_units='xy', scale=1, color=col, label=label)

ax.set_xlabel('Real')
ax.set_ylabel('Imag')
ax.set_title('Sequence Voltage Phasor Diagram (pu)')
ax.grid(True)
ax.set_aspect('equal')
ax.legend(loc='upper right')

# Fit axes
max_val = max(abs(np.real([V1,V2,V0])).max(), abs(np.imag([V1,V2,V0])).max())
ax.set_xlim(-max_val*1.2, max_val*1.2)
ax.set_ylim(-max_val*1.2, max_val*1.2)

st.pyplot(fig)

st.caption("Phasor diagram assumes unit positive‑sequence current; angles for negative and zero sequences are illustrative.")
