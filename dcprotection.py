import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import interp1d

def fault_current_biexp(t, I0, A, alpha, beta):
    return I0 * (A * np.exp(-alpha * t) + (1 - A) * np.exp(-beta * t))

def interpolate_time_current(I_by_Ir, data_points):
    I_vals, t_vals = zip(*data_points)
    log_I = np.log10(I_vals)
    log_t = np.log10(t_vals)
    f = interp1d(log_I, log_t, kind='linear', fill_value='extrapolate')
    interp_t = 10**f(np.log10(I_by_Ir))
    return interp_t

def calculate_protection(v_dc, p_rack, n_racks):
    container_power = p_rack * n_racks
    i_rack = p_rack / v_dc if v_dc > 0 else 0
    i_total = i_rack * n_racks
    fuse = i_total * 1.25
    breaker = (i_total / 0.8) * 1.25 if i_total > 0 else 0
    return container_power, i_total, fuse, breaker

st.set_page_config(page_title="BESS DC Protection Simulator", layout="wide")

st.sidebar.header("BESS Configuration & Fault")
v_dc = st.sidebar.number_input("DC Bus Voltage (V)", value=1331.2, format="%.1f")
p_rack_mw = st.sidebar.number_input("Rack Power (MW)", value=0.839, format="%.3f")
n_racks = st.sidebar.number_input("Racks per Container", value=6, min_value=1, step=1)

I_nom = (p_rack_mw * 1e6) / v_dc if v_dc > 0 else 0
I_total = I_nom * n_racks
st.sidebar.markdown(f"**Nominal Current (Container Level):** {I_total:.0f} A")

def prepare_slider_vals(i_total):
    if i_total <= 0:
        min_v, max_v, default_v = 1.0, 1000.0, 500.0
    else:
        min_v = float(i_total)
        max_v = float(5 * i_total)
        default_v = float(min(max(3 * i_total, min_v), max_v))
        if min_v >= max_v:
            min_v, max_v, default_v = 1.0, 1000.0, 500.0
    return min_v, max_v, default_v

min_I0, max_I0, default_I0 = prepare_slider_vals(I_total)
step_size = max(1.0, (max_I0 - min_I0) / 100.0)

I0 = st.sidebar.slider(
    "Initial Fault Current I0 (A)",
    min_value=min_I0,
    max_value=max_I0,
    value=default_I0,
    step=step_size,
    format="%.1f"
)

A = st.sidebar.slider("Decay weight A (fast decay)", 0.0, 1.0, 0.7, 0.05)
alpha = st.sidebar.slider("Fast Decay rate α (1/s)", 1.0, 10.0, 5.0, 0.1)
beta = st.sidebar.slider("Slow Decay rate β (1/s)", 0.01, 1.0, 0.2, 0.01)

gf_threshold = st.sidebar.slider("Ground Fault Sensitivity (A)", 0.1, 100.0, 10.0, 0.1)
gf_current = st.sidebar.slider("Simulated Ground Fault Current (A)", 0.0, 100.0, 0.0, 0.1)

# Run everything dynamically
container_power, i_total, fuse_rtg, breaker_rtg = calculate_protection(v_dc, p_rack_mw * 1e6, n_racks)

st.subheader("Container Protection Ratings")
st.write(f"Total Container Power: {container_power / 1e6:.3f} MW")
st.write(f"Total Current: {i_total:.0f} A")
st.write(f"Recommended Fuse Rating: {fuse_rtg:.0f} A")
st.write(f"Recommended Circuit Breaker Rating: {breaker_rtg:.0f} A")

t = np.linspace(0, 5, 1000)
fault_current = fault_current_biexp(t, I0, A, alpha, beta)

fuse_data = [(1, 600), (1.5, 180), (2, 60), (3, 20), (5, 5), (10, 0.5)]
breaker_data = [(1, 300), (1.2, 150), (1.5, 60), (2, 30), (3, 10), (5, 3), (10, 0.5)]

rated_current = i_total * 1.25 if i_total > 0 else 1
I_by_Ir = fault_current / rated_current

fuse_trip = interpolate_time_current(I_by_Ir, fuse_data)
breaker_trip = interpolate_time_current(I_by_Ir, breaker_data)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(t, fault_current, 'r-', label="Fault Current")
ax1.set_yscale("log")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Fault Current (A)", color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1.grid(True, which='both', linestyle='--')

ax2 = ax1.twinx()
ax2.plot(t, fuse_trip, 'b-', label="Fuse Trip Time")
ax2.plot(t, breaker_trip, 'g-', label="Breaker Trip Time")
ax2.set_yscale("log")
ax2.set_ylabel("Trip Time (s)", color='b')
ax2.tick_params(axis='y', labelcolor='b')

fig.legend(loc='upper right')
st.pyplot(fig)

st.subheader("Protection Coordination Curve")
I_ratio_range = np.logspace(0, 1, 500)
fuse_trip_curve = interpolate_time_current(I_ratio_range, fuse_data)
breaker_trip_curve = interpolate_time_current(I_ratio_range, breaker_data)

fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(I_ratio_range, fuse_trip_curve, label='Fuse Trip Curve', color='blue')
ax.plot(I_ratio_range, breaker_trip_curve, label='Breaker Trip Curve', color='green')
ax.fill_between(I_ratio_range, breaker_trip_curve, fuse_trip_curve,
                where=(fuse_trip_curve > breaker_trip_curve), color='yellow', alpha=0.3,
                label='Coordination Zone')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Current Multiple of Rated Current (I / I_r)')
ax.set_ylabel('Trip Time (s)')
ax.legend()
ax.grid(True, which='both', linestyle='--')
st.pyplot(fig2)

st.subheader("Ground Fault Detection Simulation")
time_gf = np.linspace(0, 5, 500)
gf_signal = np.piecewise(time_gf, [time_gf < 2, time_gf >= 2], [0, gf_current])

fig3, ax = plt.subplots(figsize=(10, 4))
ax.plot(time_gf, gf_signal, label='Ground Fault Current Signal (A)')
ax.axhline(gf_threshold, color='red', linestyle='--', label='Detection Threshold')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current (A)')
ax.legend()
ax.grid(True)
st.pyplot(fig3)

if gf_current >= gf_threshold:
    st.error(f"Ground fault detected! Fault current {gf_current:.1f} A exceeds threshold {gf_threshold} A.")
else:
    st.success("No ground fault detected.")
