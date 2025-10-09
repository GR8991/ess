import streamlit as st

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
