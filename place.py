# streamlit_app.py

import streamlit as st
import pandas as pd

st.set_page_config(page_title="BESS Site Sizing & Impedance Calculator", layout="centered")

st.title("ERCOT-Compliant BESS Site Sizing & Impedance Calculator")

st.markdown("""
This tool estimates:
- Site acreage required (MWh/acre)  
- AC block-level and site-level equivalent impedance  
in accordance with ERCOT interconnection and NFPA 855 safety clearances.
""")

# 1. Project & Container Inputs
st.header("1. Project & Container Specifications")
total_mwh = st.number_input("Total BESS Capacity (MWh)", min_value=0.1, step=0.1, value=100.0)
cont_mwh  = st.number_input("Per-Container Energy (MWh)", min_value=0.1, step=0.1, value=5.0)
length_ft = st.number_input("Container Length (ft)", min_value=1.0, step=1.0, value=50.0)
width_ft  = st.number_input("Container Width (ft)",  min_value=1.0, step=1.0, value=10.0)

# 2. Safety Clearances (ERCOT / NFPA 855)
st.header("2. Safety & Spacing Requirements")
clearance_ft = st.number_input("Minimum Fire Separation (ft)", min_value=5.0, step=1.0, value=15.0)
perimeter_ft = st.number_input("Perimeter Setback (ft)",       min_value=5.0, step=1.0, value=33.0)
road_pct     = st.slider("Internal Road & Equipment Allowance (%)", 0, 50, 25)

# 3. Base & Grid Impedance Inputs
st.header("3. Grid & Transformer Impedance")
site_mva   = st.number_input("Site Base MVA", min_value=1.0, step=1.0, value=100.0)
volt_kv    = st.number_input("Collector Voltage (kV)", min_value=1.0, step=1.0, value=34.5)
sc_mva     = st.number_input("ERCOT Short-Circuit MVA at POI", min_value=10.0, step=10.0, value=2000.0)
tx_pct_z   = st.number_input("Step-up Transformer %Z", min_value=1.0, max_value=15.0, step=0.1, value=6.0)

# 4. Calculations
st.header("4. Calculations & Results")

# Site sizing
n_containers = total_mwh / cont_mwh
area_cont     = (length_ft * width_ft) / 43560
eff_len       = length_ft + 2 * clearance_ft
eff_wid       = width_ft  + 2 * clearance_ft
area_eff      = (eff_len * eff_wid) / 43560
gross_area    = n_containers * area_eff
site_area     = gross_area * (1 + road_pct / 100) + (perimeter_ft * 4 * (eff_len + eff_wid) / 43560)
mwh_per_acre  = total_mwh / site_area

# Impedance conversion
z_tx_ohm = (tx_pct_z / 100) * (volt_kv**2 / site_mva)
z_source = (site_mva / sc_mva) * (1)        # as per-unit
z_base   = volt_kv**2 / site_mva
z_tx_pu  = z_tx_ohm / z_base

st.subheader("Site Footprint")
st.write(f"• Number of Containers: **{n_containers:.1f}**")
st.write(f"• Gross Footprint (acres): **{gross_area:.2f}**")
st.write(f"• Total Site Area w/ Allowances & Setbacks (acres): **{site_area:.2f}**")
st.write(f"• Achieved MWh per Acre: **{mwh_per_acre:.1f}**")

st.subheader("Equivalent Impedance")
st.write(f"• Transformer Impedance: {z_tx_pu:.3f} pu @ {volt_kv} kV/{site_mva} MVA")
st.write(f"• ERCOT Source Impedance: {z_source:.4f} pu")

# 5. Impedance Matrices (block & site)
if st.button("Show Impedance Matrices"):
    import numpy as np
    # Simple 2-block example for demonstration
    buses = ["Block1", "Block2", "Collector", "Grid"]
    y = np.zeros((4,4))
    # diagonal sums
    y[0,0] = 1/area_eff   + 1/z_tx_pu
    y[1,1] = y[0,0]
    y[0,1] = y[1,0] = -1/area_eff
    y[2,2] = 2*(1/z_tx_pu) + 1/z_source
    y[3,3] = 1/z_source
    y[0,2] = y[2,0] = -1/z_tx_pu
    y[1,2] = y[2,1] = -1/z_tx_pu
    y[2,3] = y[3,2] = -1/z_source
    z = np.linalg.inv(y)
    df_z = pd.DataFrame(z, index=buses, columns=buses)
    st.write("### Y-Bus Matrix (pu)")
    st.dataframe(pd.DataFrame(y, index=buses, columns=buses))
    st.write("### Z-Bus Matrix (pu)")
    st.dataframe(df_z)

# 6. Download Results
st.header("5. Download Data")
df = pd.DataFrame({
    "Metric": ["Gross Area (ac)", "Site Area (ac)", "MWh/acre", "Z_tx (pu)", "Z_source (pu)"],
    "Value": [gross_area, site_area, mwh_per_acre, z_tx_pu, z_source]
})
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV Report", data=csv, file_name="bess_site_report.csv", mime="text/csv")
