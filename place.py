# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ezdxf
from math import ceil
from io import BytesIO

st.set_page_config(page_title="BESS Site Layout & CAD Export", layout="centered")

st.title("ERCOT-Compliant BESS Site Layout & CAD Export")

st.markdown("""
This tool enforces NFPA 855 minimum clearances (10 ft/3 m between containers and to perimeter)
and generates both an on-screen CAD-style layout and a downloadable DXF file with marked distances.
""")

# 1. Inputs
st.header("1. Project & Container Specifications")
total_mwh   = st.number_input("Total BESS Capacity (MWh)", min_value=0.1, step=0.1, value=100.0)
cont_mwh    = st.number_input("Per-Container Energy (MWh)", min_value=0.1, step=0.1, value=5.0)
length_ft   = st.number_input("Container Length (ft)", min_value=1.0, step=1.0, value=50.0)
width_ft    = st.number_input("Container Width (ft)",  min_value=1.0, step=1.0, value=10.0)

st.header("2. NFPA 855 Safety Clearances")
clearance_ft = st.number_input("Container Separation (ft)", min_value=10.0, step=1.0, value=10.0)
perimeter_ft = st.number_input("Perimeter Setback (ft)",     min_value=10.0, step=1.0, value=10.0)
road_pct     = st.slider("Allowance (%) for Roads & Equipment", 0, 50, 25)

# Validate NFPA 855
if clearance_ft < 10 or perimeter_ft < 10:
    st.error("NFPA 855 requires at least 10 ft (3 m) separation and setback.")
    st.stop()

# 3. Calculations
n_containers = int(ceil(total_mwh / cont_mwh))
eff_len = length_ft + 2 * clearance_ft
eff_wid = width_ft  + 2 * clearance_ft

gross_area = n_containers * eff_len * eff_wid / 43560
site_area  = gross_area * (1 + road_pct/100)
mwh_per_acre = total_mwh / site_area

st.subheader("Site Footprint Results")
st.write(f"- **Containers:** {n_containers}")
st.write(f"- **Site Area (acres):** {site_area:.2f}")
st.write(f"- **MWh per Acre:** {mwh_per_acre:.1f}")

# 4. On-Screen Layout
st.header("4. On-Screen CAD-Style Layout")
cols = int(ceil(np.sqrt(n_containers)))
rows = int(ceil(n_containers / cols))
block_width  = cols * eff_wid
block_height = rows * eff_len

fig, ax = plt.subplots(figsize=(8, 8))
# Perimeter
ax.add_patch(plt.Rectangle(
    (-perimeter_ft, -perimeter_ft),
    block_width + 2*perimeter_ft,
    block_height + 2*perimeter_ft,
    fill=False, edgecolor='red', linewidth=2, label='Perimeter Setback'))

# Containers
for i in range(n_containers):
    r = i // cols
    c = i % cols
    x = c * eff_wid
    y = block_height - (r+1) * eff_len
    # clearance zone
    ax.add_patch(plt.Rectangle((x, y), eff_wid, eff_len,
                               fill=True, color='#cccccc', edgecolor='black'))
    # container
    ax.add_patch(plt.Rectangle((x+clearance_ft, y+clearance_ft),
                               length_ft, width_ft, fill=True, color='#666666'))

# Mark distances
ax.annotate(f"{clearance_ft} ft", xy=(0, -perimeter_ft/2),
            xytext=(block_width/2, -perimeter_ft/2), ha='center')
ax.annotate(f"{perimeter_ft} ft", xy=(-perimeter_ft/2, 0),
            xytext=(-perimeter_ft/2, block_height/2), va='center', rotation=90)

ax.set_aspect('equal')
ax.set_xlim(-perimeter_ft-10, block_width + perimeter_ft+10)
ax.set_ylim(-perimeter_ft-10, block_height + perimeter_ft+10)
ax.set_xlabel("Feet")
ax.set_ylabel("Feet")
ax.set_title("BESS Container Layout with NFPA 855 Clearances")
ax.legend(loc='upper right')
st.pyplot(fig)

# 5. DXF Export
st.header("5. Download DXF CAD File")

def create_dxf():
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    # Perimeter
    per_x = -perimeter_ft
    per_y = -perimeter_ft
    per_w = block_width + 2*perimeter_ft
    per_h = block_height + 2*perimeter_ft
    msp.add_lwpolyline([
        (per_x, per_y), (per_x+per_w, per_y),
        (per_x+per_w, per_y+per_h), (per_x, per_y+per_h),
        (per_x, per_y)
    ], dxfattribs={'color': 1})
    # Containers
    for i in range(n_containers):
        r = i // cols
        c = i % cols
        x = c * eff_wid
        y = block_height - (r+1) * eff_len
        # clearance zone
        msp.add_lwpolyline([
            (x, y), (x+eff_wid, y), (x+eff_wid, y+eff_len),
            (x, y+eff_len), (x, y)
        ], dxfattribs={'color': 8})
        # container
        msp.add_lwpolyline([
            (x+clearance_ft, y+clearance_ft),
            (x+clearance_ft+length_ft, y+clearance_ft),
            (x+clearance_ft+length_ft, y+clearance_ft+width_ft),
            (x+clearance_ft, y+clearance_ft+width_ft),
            (x+clearance_ft, y+clearance_ft)
        ], dxfattribs={'color': 7})
    # Dimension lines
    mid_x = block_width / 2
    msp.add_linear_dim(base=(0, -perimeter_ft), p1=(0, -perimeter_ft),
                       p2=(block_width, -perimeter_ft),
                       override={'dimtxsty': 'OpenSans', 'dimscale': 1})
    msp.add_linear_dim(base=(-perimeter_ft, 0), p1=(-perimeter_ft, 0),
                       p2=(-perimeter_ft, block_height),
                       override={'dimtxsty': 'OpenSans', 'dimscale': 1})
    return doc

if st.button("Generate DXF"):
    dxf_doc = create_dxf()
    stream = BytesIO()
    dxf_doc.write(stream)
    stream.seek(0)
    st.download_button("Download BESS Layout.dxf", data=stream, file_name="bess_layout.dxf", mime="application/dxf")

# 6. CSV Report
st.header("6. Download Summary CSV")
df = pd.DataFrame({
    "Metric": ["Containers", "Site Area (ac)", "MWh per Acre"],
    "Value": [n_containers, round(site_area, 2), round(mwh_per_acre, 1)]
})
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv, file_name="bess_layout_report.csv", mime="text/csv")
