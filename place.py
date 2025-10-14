# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

st.set_page_config(page_title="BESS Site Sizing & Layout", layout="centered")

st.title("ERCOT-Compliant BESS Site Sizing, Impedance & Layout")

# 1. Inputs
st.header("1. Project & Container Specifications")
total_mwh = st.number_input("Total BESS Capacity (MWh)", min_value=0.1, step=0.1, value=100.0)
cont_mwh  = st.number_input("Per-Container Energy (MWh)", min_value=0.1, step=0.1, value=5.0)
length_ft = st.number_input("Container Length (ft)", min_value=1.0, step=1.0, value=50.0)
width_ft  = st.number_input("Container Width (ft)",  min_value=1.0, step=1.0, value=10.0)

st.header("2. Safety & Spacing Requirements")
clearance_ft = st.number_input("Fire Separation (ft)", min_value=1.0, step=1.0, value=15.0)
perimeter_ft = st.number_input("Perimeter Setback (ft)", min_value=1.0, step=1.0, value=33.0)
road_pct     = st.slider("Allowance (%)", 0, 50, 25)

# 3. Calculations
n_containers = int(ceil(total_mwh/cont_mwh))
eff_len = length_ft + 2 * clearance_ft
eff_wid = width_ft  + 2 * clearance_ft

# Site sizing
gross_area = n_containers * eff_len * eff_wid / 43560
site_area  = gross_area * (1 + road_pct/100) + ((perimeter_ft*2 + eff_len*ceil(np.sqrt(n_containers)))*(perimeter_ft*2 + eff_wid*ceil(np.sqrt(n_containers))))/43560
mwh_per_acre = total_mwh / site_area

st.subheader("Site Footprint Results")
st.write(f"- Containers: **{n_containers}**")
st.write(f"- Site Area (ac): **{site_area:.2f}**")
st.write(f"- MWh per Acre: **{mwh_per_acre:.1f}**")

# 4. CAD-Style Layout Drawing
st.header("4. CAD-Style Layout")
cols = int(ceil(np.sqrt(n_containers)))
rows = int(ceil(n_containers/cols))

fig, ax = plt.subplots(figsize=(8, 8))
# total block dimensions
block_width = cols * eff_wid
block_height = rows * eff_len

# draw perimeter
perim_x = -perimeter_ft
perim_y = -perimeter_ft
perim_width = block_width + 2*perimeter_ft
perim_height = block_height + 2*perimeter_ft
ax.add_patch(plt.Rectangle((perim_x, perim_y), perim_width, perim_height,
                           fill=False, edgecolor='red', linewidth=2, label='Perimeter Setback'))

# draw containers with clearances
for i in range(n_containers):
    r = i // cols
    c = i % cols
    x = c * eff_wid
    y = block_height - (r+1) * eff_len
    # container footprint box (with clearance)
    ax.add_patch(plt.Rectangle((x, y), eff_wid, eff_len,
                               fill=True, color='#cccccc', edgecolor='black'))
    # inner container
    ax.add_patch(plt.Rectangle((x+clearance_ft, y+clearance_ft),
                               length_ft, width_ft, fill=True, color='#666666'))

ax.set_aspect('equal')
ax.set_xlim(perim_x-10, perim_x+perim_width+10)
ax.set_ylim(perim_y-10, perim_y+perim_height+10)
ax.set_xlabel("Feet")
ax.set_ylabel("Feet")
ax.set_title("Container Layout with Clearances & Perimeter Setback")
ax.legend(loc='upper right')

st.pyplot(fig)

# 5. Download CSV
st.header("5. Download Report")
df = pd.DataFrame({
    "Metric": ["Containers","Site Area (ac)","MWh/acre"],
    "Value": [n_containers, site_area, mwh_per_acre]
})
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv, file_name="bess_layout_report.csv", mime="text/csv")
