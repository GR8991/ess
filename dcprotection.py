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
""")'''
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

