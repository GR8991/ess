# ==========================================================
# Battery Equivalent Circuit Model Simulator (Advanced)
# Author: ScholarGPT (2025)
# Description: Streamlit app to simulate charge/discharge
# cycles of a Li-ion battery ECM with temp and RC dynamics.
# ==========================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# ECM Simulation Functions
# ----------------------------------------------------------

def ocv_curve(soc):
    """Example Open-Circuit Voltage curve for LFP chemistry."""
    soc = np.clip(soc, 0, 1)
    return 3.0 + 0.6 * soc - 0.1 * np.exp(-10 * (soc - 0.9))


def simulate_ecm(
    I_profile, dt, params, SoC0=1.0, model="1RC", temp_C=25.0
):
    """Simulates 1-RC or 2-RC Thevenin ECM."""
    R0_ref, R1, C1, R2, C2, C_nom, alpha_T = (
        params["R0_ref"],
        params["R1"],
        params["C1"],
        params["R2"],
        params["C2"],
        params["C_nom"],
        params["alpha_T"],
    )

    # Temperature correction for R0
    R0 = R0_ref * (1 + alpha_T * (25 - temp_C))

    t = np.arange(0, len(I_profile) * dt, dt)
    SoC = np.zeros_like(t)
    V_rc1 = np.zeros_like(t)
    V_rc2 = np.zeros_like(t)
    V_term = np.zeros_like(t)
    P_loss = np.zeros_like(t)
    SoC[0] = SoC0

    for k in range(1, len(t)):
        # RC network 1
        dVrc1 = (-V_rc1[k - 1] / (R1 * C1) + I_profile[k - 1] / C1) * dt
        V_rc1[k] = V_rc1[k - 1] + dVrc1

        # RC network 2 (if enabled)
        if model == "2RC":
            dVrc2 = (-V_rc2[k - 1] / (R2 * C2) + I_profile[k - 1] / C2) * dt
            V_rc2[k] = V_rc2[k - 1] + dVrc2

        # SoC update
        SoC[k] = SoC[k - 1] - (I_profile[k - 1] * dt) / (3600 * C_nom)

        # Terminal voltage
        Eocv = ocv_curve(SoC[k])
        V_term[k] = Eocv - I_profile[k] * R0 - V_rc1[k] - V_rc2[k]

        # Power loss (ohmic only)
        P_loss[k] = (I_profile[k] ** 2) * R0 / 1000  # kW

    return t, SoC, V_term, P_loss


# ----------------------------------------------------------
# Streamlit Interface
# ----------------------------------------------------------

st.set_page_config(page_title="Battery ECM Simulator", layout="centered")
st.title("🔋 Advanced Battery Equivalent Circuit Model (ECM) Simulator")

st.markdown(
    """
This simulator models a **Li-ion battery** using a 1-RC or 2-RC Thevenin model.  
You can visualize **voltage, SoC, and losses** under different temperatures and cycles.
    """
)

# Sidebar parameters
st.sidebar.header("Simulation Settings")

model_type = st.sidebar.radio("ECM Type", ["1RC", "2RC"], horizontal=True)
R0_ref = st.sidebar.slider("Ohmic Resistance R₀_ref (Ω)", 0.0005, 0.02, 0.005, 0.0005)
R1 = st.sidebar.slider("Polarization Resistance R₁ (Ω)", 0.001, 0.05, 0.01, 0.001)
C1 = st.sidebar.slider("Capacitance C₁ (F)", 10.0, 5000.0, 1000.0, 10.0)
R2 = st.sidebar.slider("Polarization Resistance R₂ (Ω)", 0.001, 0.05, 0.01, 0.001)
C2 = st.sidebar.slider("Capacitance C₂ (F)", 10.0, 5000.0, 500.0, 10.0)
C_nom = st.sidebar.slider("Nominal Capacity (Ah)", 10.0, 300.0, 100.0, 1.0)
alpha_T = st.sidebar.slider("Temp Coeff α (per °C)", 0.0, 0.01, 0.004, 0.001)
temp_C = st.sidebar.slider("Battery Temperature (°C)", -10.0, 60.0, 25.0, 1.0)
SoC0 = st.sidebar.slider("Initial SoC (0–1)", 0.0, 1.0, 1.0, 0.01)
t_end = st.sidebar.slider("Simulation Time (s)", 60, 7200, 3600, 60)
dt = st.sidebar.slider("Time Step (s)", 0.1, 5.0, 1.0, 0.1)
I_max = st.sidebar.slider("Discharge Current (A)", 10, 200, 100, 5)

# Current profile type
profile = st.radio("Select Profile:", ["Single Cycle", "Multi-Cycle", "Sinusoidal"], horizontal=True)

steps = int(t_end / dt)
t = np.arange(0, steps * dt, dt)

# Generate current profile
if profile == "Single Cycle":
    I_profile = np.ones(steps) * I_max
    I_profile[int(steps / 2) :] = -I_max
elif profile == "Multi-Cycle":
    cycles = int(np.floor(t_end / (t_end / 4)))
    I_profile = I_max * np.sign(np.sin(2 * np.pi * cycles * t / t_end))
else:  # Sinusoidal
    I_profile = I_max * np.sin(2 * np.pi * t / (t_end / 2))

st.line_chart(I_profile, height=150)

# Run simulation
params = {
    "R0_ref": R0_ref,
    "R1": R1,
    "C1": C1,
    "R2": R2,
    "C2": C2,
    "C_nom": C_nom,
    "alpha_T": alpha_T,
}

t, SoC, V, P_loss = simulate_ecm(I_profile, dt, params, SoC0, model_type, temp_C)

# Compute energy and efficiency
E_out = np.trapz(np.clip(I_profile * V, 0, None), t) / 3600  # kWh
E_in = np.trapz(np.clip(-I_profile * V, 0, None), t) / 3600  # kWh
E_loss = np.trapz(P_loss / 1000, t)  # kWh
eff = (E_out / E_in) * 100 if E_in > 0 else 0

# ----------------------------------------------------------
# Results Visualization
# ----------------------------------------------------------

st.subheader("Simulation Results")

fig, ax = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

ax[0].plot(t, V, label="Terminal Voltage (V)")
ax[0].set_ylabel("Voltage (V)")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(t, SoC * 100, label="State of Charge", color="orange")
ax[1].set_ylabel("SoC (%)")
ax[1].grid(True)
ax[1].legend()

ax[2].plot(t, P_loss, label="Ohmic Loss (kW)", color="red")
ax[2].set_ylabel("Loss (kW)")
ax[2].grid(True)
ax[2].legend()

ax[3].plot(t, I_profile, label="Current (A)", color="green")
ax[3].set_xlabel("Time (s)")
ax[3].set_ylabel("Current (A)")
ax[3].grid(True)
ax[3].legend()

st.pyplot(fig)

# ----------------------------------------------------------
# Key Performance Metrics
# ----------------------------------------------------------
st.markdown("### ⚡ Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Energy Out (kWh)", f"{E_out:.3f}")
col2.metric("Energy In (kWh)", f"{E_in:.3f}")
col3.metric("Round Trip Efficiency", f"{eff:.2f} %")

st.caption(
    "Equations: V = Eocv(SOC) − I·R₀ − Σ(Vrc);  dVrc/dt = −Vrc/(R·C) + I/C;  dSoC/dt = −I/(3600·C_nom)"
)

st.write("---")
st.write("© 2025 ScholarGPT — Advanced ECM Educational Simulator")
