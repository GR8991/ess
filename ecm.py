# ==========================================================
# Battery Equivalent Circuit Model (1-RC Thevenin) Simulator
# Author: ScholarGPT (2025)
# Description: Interactive Streamlit app for simulating
#              battery voltage, SoC, and losses dynamically.
# ==========================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Basic ECM simulator function
# ----------------------------------------------------------
def simulate_ecm(I_profile, dt, params, SoC0=1.0):
    """Simulates a 1-RC Thevenin battery ECM."""

    R0, R1, C1, C_nom = (
        params["R0"],
        params["R1"],
        params["C1"],
        params["C_nom"],
    )
    t = np.arange(0, len(I_profile) * dt, dt)
    SoC = np.zeros_like(t)
    V_rc = np.zeros_like(t)
    V_term = np.zeros_like(t)
    P_loss = np.zeros_like(t)
    SoC[0] = SoC0

    def E_ocv(soc):
        """Simple Open-Circuit Voltage model for LFP."""
        soc = np.clip(soc, 0, 1)
        return 3.0 + 0.6 * soc - 0.1 * np.exp(-10 * (soc - 0.9))

    for k in range(1, len(t)):
        dVrc = (-V_rc[k - 1] / (R1 * C1) + I_profile[k - 1] / C1) * dt
        V_rc[k] = V_rc[k - 1] + dVrc
        SoC[k] = SoC[k - 1] - (I_profile[k - 1] * dt) / (3600 * C_nom)
        V_term[k] = E_ocv(SoC[k]) - I_profile[k] * R0 - V_rc[k]
        P_loss[k] = (I_profile[k] ** 2) * R0 / 1000  # in kW

    return t, SoC, V_term, P_loss


# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
st.set_page_config(page_title="Battery ECM Simulator", layout="centered")

st.title("🔋 Battery Equivalent Circuit Model (ECM) Simulator")
st.write(
    """
This tool simulates a **1-RC Thevenin model** of a battery cell or module.
You can adjust model parameters and current profile, then visualize:
- Voltage response
- State of Charge (SoC)
- Ohmic (I²R) losses
    """
)

# --- Sidebar parameters ---
st.sidebar.header("Simulation Parameters")

# Model parameters
R0 = st.sidebar.slider("Ohmic Resistance R₀ (Ω)", 0.0005, 0.02, 0.005, 0.0005)
R1 = st.sidebar.slider("Polarization Resistance R₁ (Ω)", 0.001, 0.05, 0.01, 0.001)
C1 = st.sidebar.slider("Capacitance C₁ (F)", 10.0, 5000.0, 1000.0, 10.0)
C_nom = st.sidebar.slider("Nominal Capacity (Ah)", 1.0, 300.0, 100.0, 1.0)
SoC0 = st.sidebar.slider("Initial SoC (0–1)", 0.0, 1.0, 1.0, 0.01)
t_end = st.sidebar.slider("Simulation Time (s)", 60, 7200, 1800, 60)
I_const = st.sidebar.slider("Constant Current (A)", -200, 200, 50, 1)
dt = st.sidebar.slider("Time Step (s)", 0.1, 5.0, 1.0, 0.1)

# --- Generate current profile ---
st.subheader("Current Profile")
profile_type = st.radio(
    "Select current input type:",
    ["Constant Current", "Step Change", "Sinusoidal Load"],
    horizontal=True,
)

steps = int(t_end / dt)
t = np.arange(0, steps * dt, dt)

if profile_type == "Constant Current":
    I_profile = np.ones(steps) * I_const
elif profile_type == "Step Change":
    I_profile = np.ones(steps) * I_const
    I_profile[int(steps / 2) :] = -I_const  # change direction halfway
else:  # Sinusoidal load
    I_profile = I_const * np.sin(2 * np.pi * t / (t_end / 2))

st.line_chart(I_profile, height=150)

# --- Run simulation ---
params = {"R0": R0, "R1": R1, "C1": C1, "C_nom": C_nom}
t, SoC, V, P_loss = simulate_ecm(I_profile, dt, params, SoC0)

# --- Plot results ---
st.subheader("Simulation Results")

fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

ax[0].plot(t, V, label="Terminal Voltage (V)")
ax[0].set_ylabel("Voltage (V)")
ax[0].grid(True)

ax[1].plot(t, SoC * 100, label="SoC", color="orange")
ax[1].set_ylabel("SoC (%)")
ax[1].grid(True)

ax[2].plot(t, P_loss, label="I²R Loss", color="red")
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("Loss (kW)")
ax[2].grid(True)

for a in ax:
    a.legend()

st.pyplot(fig)

# --- Efficiency calculation ---
E_in = np.trapz(np.abs(I_profile * V) / 1000, t)  # kWh
E_loss = np.trapz(P_loss / 1000, t)  # kWh
eff = (E_in - E_loss) / E_in * 100 if E_in > 0 else 0

st.metric("Estimated Round-Trip Efficiency", f"{eff:.2f} %")

st.caption(
    "Model: V = Eocv(SOC) − I·R0 − Vrc;   dVrc/dt = −Vrc/(R1·C1) + I/C1;   dSoC/dt = −I/(3600·C_nom)"
)

st.write("---")
st.write("© 2025 ScholarGPT — for educational & research purposes.")
