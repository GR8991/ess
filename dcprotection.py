'''import streamlit as st

def calculate_protection(bess_voltage, rack_power, racks_count):
    # Calculate total container power
    container_power = rack_power * racks_count  # in Watts
    
    # Calculate current (I = P / V)
    current_per_rack = rack_power / bess_voltage
    total_current = current_per_rack * racks_count
    
    # Fuse rating with 1.25 margin
    fuse_rating = total_current * 1.25
    
    # Circuit breaker rating: using 80% rule + 1.25 margin
    breaker_rating = (total_current / 0.8) * 1.25
    
    return {
        "Total Container Power (MW)": container_power / 1e6,
        "Total Current (A)": total_current,
        "Fuse Rating (A)": fuse_rating,
        "Circuit Breaker Rating (A)": breaker_rating
    }

st.title("BESS DC Protection Study")

st.markdown("""
Enter your BESS parameters to calculate fuse and circuit breaker ratings for DC protection.
""")

bess_voltage = st.number_input("BESS Voltage (V DC)", value=1331.2, min_value=0.0, format="%.1f")
rack_power_mw = st.number_input("Rack Power (MW)", value=0.839, min_value=0.0, format="%.3f")
racks_count = st.number_input("Number of Racks per Container", value=6, min_value=1, step=1)

rack_power = rack_power_mw * 1e6  # Convert MW to W

if st.button("Calculate Protection Ratings"):
    results = calculate_protection(bess_voltage, rack_power, racks_count)
    st.success("Protection Ratings Calculated:")
    st.write(f"Total Container Power: {results['Total Container Power (MW)']:.3f} MW")
    st.write(f"Total Current: {results['Total Current (A)']:.1f} A")
    st.write(f"Recommended Fuse Rating: {results['Fuse Rating (A)']:.0f} A")
    st.write(f"Recommended Circuit Breaker Rating: {results['Circuit Breaker Rating (A)']:.0f} A")

st.markdown("""
---
**Disclaimer:** This tool provides preliminary protection ratings based on basic electrical calculations.  
Final device selection should be validated against manufacturer datasheets and applicable standards.
""")
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def calculate_protection(bess_voltage, rack_power, racks_count):
    container_power = rack_power * racks_count  # in Watts
    current_per_rack = rack_power / bess_voltage
    total_current = current_per_rack * racks_count
    fuse_rating = total_current * 1.25
    breaker_rating = (total_current / 0.8) * 1.25
    return {
        "Total Container Power (MW)": container_power / 1e6,
        "Total Current (A)": total_current,
        "Fuse Rating (A)": fuse_rating,
        "Circuit Breaker Rating (A)": breaker_rating
    }

def fault_current(time, peak_fault_current, decay_rate):
    return peak_fault_current * np.exp(-decay_rate * time)

def time_current_curve(current, rating, a, b, c):
    I_ratio = current / rating
    return a * np.power(I_ratio, b) + c

def main():
    st.title("BESS DC Protection Study with Fault Simulation")

    bess_voltage = st.number_input("BESS Voltage (V DC)", value=1331.2, min_value=0.0, format="%.1f")
    rack_power_mw = st.number_input("Rack Power (MW)", value=0.839, min_value=0.0, format="%.3f")
    racks_count = st.number_input("Number of Racks per Container", value=6, min_value=1, step=1)

    rack_power = rack_power_mw * 1e6

    results = calculate_protection(bess_voltage, rack_power, racks_count)

    st.subheader("Container Protection Ratings")
    st.write(f"Total Container Power: {results['Total Container Power (MW)']:.3f} MW")
    st.write(f"Total Current: {results['Total Current (A)']:.1f} A")
    st.write(f"Recommended Fuse Rating: {results['Fuse Rating (A)']:.0f} A")
    st.write(f"Recommended Circuit Breaker Rating: {results['Circuit Breaker Rating (A)']:.0f} A")

    st.markdown("---")
    st.subheader("Fault Simulation and Time-Current Curves")

    peak_fault_current = st.slider("Peak Fault Current (A)", min_value=results['Total Current (A)'], max_value=5*results['Total Current (A)'], value=2*results['Total Current (A)'], step=100.0)
    decay_rate = st.slider("Fault Current Decay Rate (1/s)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    time = np.linspace(0, 5, 500)
    fault_curr = fault_current(time, peak_fault_current, decay_rate)

    fuse_a, fuse_b, fuse_c = 0.5, -3.0, 0.05  # example fuse curve parameters
    breaker_a, breaker_b, breaker_c = 1.0, -2.5, 0.1  # example breaker curve parameters

    fuse_time = time_current_curve(fault_curr, results['Fuse Rating (A)'], fuse_a, fuse_b, fuse_c)
    breaker_time = time_current_curve(fault_curr, results['Circuit Breaker Rating (A)'], breaker_a, breaker_b, breaker_c)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, fault_curr, label="Fault Current (A)", color='red')
    ax.plot(time, fuse_time, label="Fuse Trip Time (s)", color='blue')
    ax.plot(time, breaker_time, label="Breaker Trip Time (s)", color='green')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A) / Trip Time (s)")
    ax.set_title("Fault Current and Time-Current Characteristic Curves")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Ground Fault Detection Simulation")
    ground_fault_sensitivity = st.slider("Ground Fault Sensitivity (A)", min_value=0.1, max_value=50.0, value=10.0, step=0.1)
    ground_fault_current = st.slider("Simulated Ground Fault Current (A)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

    if ground_fault_current >= ground_fault_sensitivity:
        st.error(f"Ground fault detected! Fault current: {ground_fault_current} A exceeds sensitivity {ground_fault_sensitivity} A.")
    else:
        st.success("No ground fault detected.")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

#── Calculation Functions ──────────────────────────────────────────────────────

def calculate_protection(v_dc, p_rack, n_racks):
    """Calculate container power, current, fuse & breaker ratings."""
    container_power = p_rack * n_racks
    i_rack = p_rack / v_dc
    i_total = i_rack * n_racks
    fuse = i_total * 1.25
    breaker = (i_total / 0.8) * 1.25
    return container_power, i_total, fuse, breaker

def fault_current(t, i_peak, decay):
    return i_peak * np.exp(-decay * t)

def trip_time(i, rating, a, b, c):
    return a * (i / rating)**b + c

#── Streamlit Layout ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BESS DC Protection Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar: Input Parameters
with st.sidebar:
    st.header("BESS Configuration")
    v_dc = st.number_input("DC Bus Voltage (V)",  value=1331.2, format="%.1f")
    p_rack_mw = st.number_input("Rack Power (MW)",   value=0.839, format="%.3f")
    n_racks = st.number_input("Racks per Container", value=6, min_value=1, step=1)
    st.markdown("---")
    st.header("Fault Simulation")
    i_peak = st.slider("Peak Fault Current (A)",
                       min_value=0.0,
                       max_value=5 * (p_rack_mw*1e6/v_dc)*n_racks,
                       value=2 * (p_rack_mw*1e6/v_dc)*n_racks,
                       step=100.0)
    decay = st.slider("Decay Rate (1/s)", 0.1, 5.0, 1.0, 0.1)
    st.markdown("---")
    st.header("Ground Fault")
    gf_sensitivity = st.slider("Sensitivity (A)", 0.1, 100.0, 10.0, 0.1)
    gf_current = st.slider("Fault Current (A)", 0.0, 100.0, 0.0, 0.1)

# Compute Protection Ratings
p_rack = p_rack_mw * 1e6
container_power, i_total, fuse_rtg, breaker_rtg = calculate_protection(v_dc, p_rack, n_racks)

# Main: Tabs
tabs = st.tabs([
    "Protection Sizing",
    "Fault & T–I Curves",
    "Ground Fault Detection"
])

#── Tab 1: Protection Sizing ───────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Container-Level DC Protection")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Power", f"{container_power/1e6:.3f} MW")
        st.metric("Total Current", f"{i_total:.0f} A")
    with col2:
        st.metric("Fuse Rating", f"{fuse_rtg:.0f} A")
        st.metric("Breaker Rating", f"{breaker_rtg:.0f} A")

    st.markdown("""
    > **Notes:**  
    > - Fuse = 1.25 × I_total  
    > - Breaker = 1.25 × (I_total/0.8)  
    > - Voltage rating assumed ≥1500 V DC  
    """)
    
#── Tab 2: Fault Simulation & Time-Current Curves ─────────────────────────────
with tabs[1]:
    st.subheader("Fault Current Simulation")
    t = np.linspace(0, 5, 500)
    i_fault = fault_current(t, i_peak, decay)

    st.line_chart({
        "Fault Current (A)": i_fault
    }, use_container_width=True)

    st.subheader("Time–Current Characteristic Curves")
    # Example curve parameters (replace with real device data for accuracy)
    fuse_t = trip_time(i_fault, fuse_rtg, a=0.5, b=-3.0, c=0.05)
    breaker_t = trip_time(i_fault, breaker_rtg, a=1.0, b=-2.5, c=0.1)

    fig, ax = plt.subplots()
    ax.plot(t, fuse_t, label="Fuse Trip Time", color="blue")
    ax.plot(t, breaker_t, label="Breaker Trip Time", color="green")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trip Time (s)")
    ax.set_title("Time–Current Curves")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("""
    *Curve model: \(t = a \,(I/I_r)^b + c\)*  
    Replace \(a,b,c\) with manufacturer’s curve data for precise coordination.
    """)

#── Tab 3: Ground Fault Detection ──────────────────────────────────────────────
with tabs[2]:
    st.subheader("Ground Fault Simulation")
    st.write(f"Detection Threshold: **{gf_sensitivity:.1f} A**")
    if gf_current >= gf_sensitivity:
        st.error(f"⚠️ Ground fault detected! Fault current = {gf_current:.1f} A")
    else:
        st.success(f"No ground fault. Simulated = {gf_current:.1f} A")

    st.markdown("""
    - Use residual-current monitors or differential sensors on the DC bus.  
    - Trip the DC breaker or isolate the faulty string upon detection.
    """)

#── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("""
**Disclaimer:** This app provides preliminary calculations and visualizations.  
For final design, validate with detailed fault studies, device datasheets, and applicable standards (IEC/IEEE).
""")'''
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
    i_rack = p_rack / v_dc
    i_total = i_rack * n_racks
    fuse = i_total * 1.25
    breaker = (i_total / 0.8) * 1.25
    return container_power, i_total, fuse, breaker

st.set_page_config(page_title="Enhanced BESS DC Protection", layout="wide")

st.sidebar.header("BESS Configuration & Fault")
v_dc = st.sidebar.number_input("DC Bus Voltage (V)", value=1331.2, format="%.1f")
p_rack_mw = st.sidebar.number_input("Rack Power (MW)", value=0.839, format="%.3f")
n_racks = st.sidebar.number_input("Racks per Container", value=6, min_value=1, step=1)

I_nom = (p_rack_mw * 1e6) / v_dc if v_dc > 0 else 0
I_total = I_nom * n_racks

st.sidebar.markdown(f"**Nominal Current (Container Level):** {I_total:.0f} A")

# Validate slider range parameters
if I_total <= 0:
    min_I0 = 1
    max_I0 = 1000
    default_I0 = 500
else:
    min_I0 = I_total
    max_I0 = 5 * I_total
    default_I0 = min(max(3 * I_total, min_I0), max_I0)

I0 = st.sidebar.slider("Initial Fault Current I0 (A)", min_value=min_I0, max_value=max_I0, value=default_I0, step=1000)
A = st.sidebar.slider("Decay weight A (fast decay)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
alpha = st.sidebar.slider("Fast Decay rate α (1/s)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
beta = st.sidebar.slider("Slow Decay rate β (1/s)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)

t = np.linspace(0, 5, 1000)
fault_current = fault_current_biexp(t, I0, A, alpha, beta)

fuse_data = [(1, 600), (1.5, 180), (2, 60), (3, 20), (5, 5), (10, 0.5)]
breaker_data = [(1, 300), (1.2, 150), (1.5, 60), (2, 30), (3, 10), (5, 3), (10, 0.5)]

rated_current = I_total * 1.25 if I_total > 0 else 1  # Avoid division by zero
I_by_Ir = fault_current / rated_current

fuse_trip = interpolate_time_current(I_by_Ir, fuse_data)
breaker_trip = interpolate_time_current(I_by_Ir, breaker_data)

fig, ax1 = plt.subplots(figsize=(10, 6))

color_current = 'tab:red'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Fault Current (A)', color=color_current)
ax1.plot(t, fault_current, color=color_current, label='Fault Current')
ax1.tick_params(axis='y', labelcolor=color_current)
ax1.set_yscale('log')
ax1.grid(True, which='both', linestyle='--')

ax2 = ax1.twinx()

color_time = 'tab:blue'
ax2.set_ylabel('Trip Time (s)', color=color_time)
ax2.plot(t, fuse_trip, color='blue', label='Fuse Trip Time')
ax2.plot(t, breaker_trip, color='green', label='Breaker Trip Time')
ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor=color_time)

fig.suptitle('Fault Current and Time-Current Characteristic Curves')
ax2.legend(loc='upper right')

st.title("Enhanced BESS DC Protection Simulation")
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
ax.set_title('Fuse & Breaker Coordination Time-Current Curves')
ax.legend()
ax.grid(True, which='both', linestyle='--')

st.pyplot(fig2)

st.subheader("Ground Fault Detection Simulation")
gf_threshold = st.slider("Ground Fault Sensitivity (A)", 0.1, 100.0, 10.0, 0.1)
gf_current = st.slider("Simulated Ground Fault Current (A)", 0.0, 100.0, 0.0, 0.1)

time_gf = np.linspace(0, 5, 500)
gf_signal = np.piecewise(time_gf, [time_gf < 2, time_gf >= 2], [0, gf_current])

fig3, ax = plt.subplots(figsize=(10, 4))
ax.plot(time_gf, gf_signal, label='Ground Fault Current Signal (A)')
ax.axhline(gf_threshold, color='red', linestyle='--', label='Detection Threshold')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current (A)')
ax.set_title('Ground Fault Detection')
ax.legend()
ax.grid(True)

st.pyplot(fig3)

if gf_current >= gf_threshold:
    st.error(f"Ground fault detected! Current {gf_current:.1f} A exceeds threshold {gf_threshold} A.")
else:
    st.success("No ground fault detected.")


