# app.py — Streamlit app to compute Ud and Uq on a 690 V base
# using the user's function logic, with robustness and explainers.
# Run:  streamlit run app.py

import math
from typing import Dict, Tuple

import numpy as np
import streamlit as st

st.set_page_config(page_title="Ud/Uq @ 690 V Calculator", page_icon="⚡", layout="centered")

# -------------------------------
# Core calculation function
# -------------------------------

def calculate_Ud_690V_base(
    P_MW: float,
    Q_MVAR: float,
    *,
    high_q_threshold: float = 20.0,
    u_terminal_pu_high_q: float = 0.94,
    k_ratio_high_q: float = 0.22,
    u_terminal_pu_low_q: float = 0.885,
    k_ratio_low_q: float = 0.45,
    base_voltage: float = 690.0,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute Ud and Uq on a 690 V base from active/reactive power (MW / MVAr).

    The logic follows the user's function:
      - Compute apparent power S, derive sin(phi) = Q / S
      - phi = asin(sin_phi) with sign from Q
      - If |Q| > threshold, use (U_terminal_pu=0.94, k=0.22), else (0.885, 0.45)
      - delta = k * phi
      - Ud_pu = U_term * cos(delta), Uq_pu = U_term * sin(delta)
      - Convert to volts on specified base (default 690 V)

    Returns (Ud_volts, Uq_volts, details_dict)
    """
    # Apparent power magnitude
    S = math.hypot(P_MW, Q_MVAR)  # sqrt(P^2 + Q^2)

    if S == 0.0:
        # Degenerate case: no power flow
        sin_phi = 0.0
        phi_rad = 0.0
    else:
        sin_phi = Q_MVAR / S
        # Robust domain handling for asin
        sin_phi = float(np.clip(sin_phi, -1.0, 1.0))
        phi_rad = float(np.arcsin(sin_phi))  # preserves sign of Q

    # Select parameters based on |Q|
    if abs(Q_MVAR) > high_q_threshold:  # High reactive power region
        U_terminal_pu = u_terminal_pu_high_q
        k_ratio = k_ratio_high_q
        region = "High-Q"
    else:  # Low reactive power region
        U_terminal_pu = u_terminal_pu_low_q
        k_ratio = k_ratio_low_q
        region = "Low-Q"

    # Angle used for dq orientation
    delta_rad = phi_rad * k_ratio

    # dq components in per-unit
    Ud_pu = U_terminal_pu * math.cos(delta_rad)
    Uq_pu = U_terminal_pu * math.sin(delta_rad)

    # Convert to volts on selected base
    Ud_volts = Ud_pu * base_voltage
    Uq_volts = Uq_pu * base_voltage

    details = {
        "S_MVA": S,  # in MVA since P, Q are in MW/MVAr
        "sin_phi": sin_phi,
        "phi_deg": math.degrees(phi_rad),
        "region": 1.0 if region == "High-Q" else 0.0,  # for quick inspection
        "region_label": 0.0,  # placeholder to keep dict numeric; label shown separately
        "U_terminal_pu": U_terminal_pu,
        "k_ratio": k_ratio,
        "delta_deg": math.degrees(delta_rad),
        "Ud_pu": Ud_pu,
        "Uq_pu": Uq_pu,
        "base_voltage_V": base_voltage,
    }
    # can't store string in details cleanly alongside floats for some ops; report label separately in UI

    return Ud_volts, Uq_volts, details


# -------------------------------
# UI
# -------------------------------

st.title("⚡ Ud/Uq on 690 V Base")
st.caption("Compute D–Q voltage components from active/reactive power. Matches your source function with a few safety checks.")

with st.sidebar:
    st.header("Inputs")
    P_MW = st.number_input("Active Power P (MW)", value=50.0, step=1.0, format="%0.3f")
    Q_MVAR = st.number_input("Reactive Power Q (MVAr)", value=10.0, step=1.0, format="%0.3f")

    st.divider()
    st.subheader("Logic Parameters")
    high_q_threshold = st.number_input("High-Q threshold |Q| (MVAr)", value=20.0, step=1.0)
    col1, col2 = st.columns(2)
    with col1:
        u_term_high = st.number_input("U_term pu (High-Q)", value=0.94, step=0.001, format="%0.3f")
        k_high = st.number_input("k ratio (High-Q)", value=0.22, step=0.01, format="%0.02f")
    with col2:
        u_term_low = st.number_input("U_term pu (Low-Q)", value=0.885, step=0.001, format="%0.3f")
        k_low = st.number_input("k ratio (Low-Q)", value=0.45, step=0.01, format="%0.02f")

    st.divider()
    base_voltage = st.number_input("Base voltage (V)", value=690.0, step=10.0)

# Compute
Ud_V, Uq_V, info = calculate_Ud_690V_base(
    P_MW,
    Q_MVAR,
    high_q_threshold=high_q_threshold,
    u_terminal_pu_high_q=u_term_high,
    k_ratio_high_q=k_high,
    u_terminal_pu_low_q=u_term_low,
    k_ratio_low_q=k_low,
    base_voltage=base_voltage,
)

# Display results
st.subheader("Results")
colA, colB = st.columns(2)
with colA:
    st.metric("Ud (V)", f"{Ud_V:,.2f}")
with colB:
    st.metric("Uq (V)", f"{Uq_V:,.2f}")

# Details
with st.expander("Show details / intermediate values"):
    st.write(
        {
            "Apparent power S (MVA)": round(info["S_MVA"], 6),
            "sin(phi)": round(info["sin_phi"], 6),
            "phi (deg)": round(info["phi_deg"], 6),
            "Region": "High-Q" if abs(Q_MVAR) > high_q_threshold else "Low-Q",
            "U_terminal (pu)": info["U_terminal_pu"],
            "k ratio": info["k_ratio"],
            "delta (deg)": round(info["delta_deg"], 6),
            "Ud (pu)": round(info["Ud_pu"], 6),
            "Uq (pu)": round(info["Uq_pu"], 6),
            "Base voltage (V)": info["base_voltage_V"],
        }
    )

st.markdown(
    """
**Notes**
- Uses `asin(Q/S)`; input is clipped to [-1, 1] for numerical safety.
- The high/low-Q region picks (U_terminal_pu, k_ratio) per your rule: |Q| > threshold → High-Q.
- If P and Q are both 0, Ud = Uq = 0.
- Change the base voltage to compute components on a different voltage base.
    """
)

# Optional: a tiny phasor-style plot for Ud/Uq (vector length equals U_terminal in pu scaled to volts)
try:
    import matplotlib.pyplot as plt

    st.subheader("Vector view")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.quiver(0, 0, Ud_V, Uq_V, angles='xy', scale_units='xy', scale=1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Ud (V)")
    ax.set_ylabel("Uq (V)")
    # Set symmetric limits based on magnitude
    mag = max(abs(Ud_V), abs(Uq_V))
    lim = max(100.0, math.ceil(mag / 100.0) * 100.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    st.pyplot(fig)
except Exception as e:
    st.info("Matplotlib not available — skip vector plot.")
