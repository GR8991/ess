# transformer_designer.py

import streamlit as st
import numpy as np

st.title("AI-Based Transformer Design Prototype")

# 1. User Inputs
st.header("Input Transformer Specifications")
hv_voltage = st.number_input("Primary (HV) Line-to-Line Voltage (V)", value=33000)
lv_voltage = st.number_input("Secondary (LV) Line-to-Line Voltage (V)", value=690)
mva_total  = st.number_input("Total Transformer Rating (MVA)", value=12.0, format="%.1f")
num_lv     = st.number_input("Number of LV Windings", value=4, min_value=1, step=1)

# On-submit
if st.button("Design Transformer"):
    # 2. Basic Electrical Calculations
    hv_phase_v = hv_voltage / np.sqrt(3)
    lv_phase_v = lv_voltage / np.sqrt(3)
    hv_current = (mva_total * 1e6) / (np.sqrt(3) * hv_voltage)
    mva_per_lv = mva_total / num_lv
    lv_current = (mva_per_lv * 1e6) / (np.sqrt(3) * lv_voltage)

    # 3. Turns Ratio & Turns Calculation
    turns_ratio = hv_phase_v / lv_phase_v
    emf_const   = 4.44 * 50 * 1.7  # 4.44·f·Bmax
    # Assume core area (cm²) from empirical scaling
    core_area   = 400  # placeholder
    turns_per_volt = 1 / (emf_const * core_area / 1e4)
    hv_turns    = int(hv_phase_v * turns_per_volt)
    lv_turns    = int(lv_phase_v * turns_per_volt)

    # 4. Conductor Sizing (basic)
    current_density = 2.5  # A/mm²
    hv_cond_area = hv_current / current_density
    lv_cond_area = lv_current / current_density

    # 5. Display Results
    st.subheader("Electrical & Winding Results")
    st.write(f"• HV Phase Voltage: {hv_phase_v:,.1f} V")
    st.write(f"• LV Phase Voltage: {lv_phase_v:,.1f} V")
    st.write(f"• HV Line Current: {hv_current:,.1f} A")
    st.write(f"• LV Current per Winding: {lv_current:,.1f} A")
    st.write(f"• Turns Ratio (HV:LV): {turns_ratio:.2f}:1")
    st.write(f"• HV Turns/Phase: {hv_turns}")
    st.write(f"• LV Turns/Phase: {lv_turns}")
    st.write(f"• HV Conductor Area: {hv_cond_area:,.1f} mm²")
    st.write(f"• LV Conductor Area (per winding): {lv_cond_area:,.1f} mm²")

    # 6. Optimization Placeholder
    st.subheader("Optimization Recommendations")
    st.info("Optimization module not yet implemented. Future ML-based suggestions will appear here.")

    # 7. Export Report
    report = f"""
## Transformer Design Report

- **HV Voltage**: {hv_voltage} V  
- **LV Voltage**: {lv_voltage} V  
- **Rating**: {mva_total} MVA  
- **LV Windings**: {num_lv}  

**Calculated Parameters**  
- HV Phase Voltage: {hv_phase_v:,.1f} V  
- LV Phase Voltage: {lv_phase_v:,.1f} V  
- HV Current: {hv_current:,.1f} A  
- LV Current per Winding: {lv_current:,.1f} A  
- Turns Ratio: {turns_ratio:.2f}:1  
- HV Turns/Phase: {hv_turns}  
- LV Turns/Phase: {lv_turns}  
- HV Conductor Area: {hv_cond_area:,.1f} mm²  
- LV Conductor Area: {lv_cond_area:,.1f} mm²  
"""
    st.download_button("Download Design Report", report, file_name="transformer_design_report.md")
