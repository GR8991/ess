import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="3D Grounding Visualizer", layout="wide")
st.title("⚡ 3D Electrode + Equipotential Visualizer")

# -------- Helpers --------
def norm(s: str) -> str:
    """lowercase + remove non-alnum so 'Current(Amps) ' -> 'currentamps'"""
    return re.sub(r'[^a-z0-9]+', '', str(s).strip().lower())

def map_columns(df: pd.DataFrame) -> dict:
    """
    Map many possible header spellings to canonical names.
    Shows a warning if any required field is missing.
    """
    canon = {
        "x1": ["x1", "xstart", "x_1", "x1m"],
        "y1": ["y1", "ystart", "y_1", "y1m"],
        "z1": ["z1", "zstart", "z_1", "z1m"],
        "x2": ["x2", "xend", "x_2", "x2m"],
        "y2": ["y2", "yend", "y_2", "y2m"],
        "z2": ["z2", "zend", "z_2", "z2m"],
        "length": ["length", "len", "l"],
        "radius": ["radius", "rad", "r", "diameter", "dia"],
        "current": ["current", "i", "amps", "currenta", "currentamps", "icur"],
        "electrode": ["electrode", "name", "electrodename", "id", "asy"]
    }
    # Build lookup of normalized -> original
    ncols = {norm(c): c for c in df.columns}
    mapping = {}
    for key, aliases in canon.items():
        for alias in aliases:
            if alias in ncols:
                mapping[key] = ncols[alias]
                break
        if key not in mapping:
            # not strictly required: radius/current/electrode can be missing
            if key in ["x1","y1","z1","x2","y2","z2"]:
                st.warning(f"Missing column for '{key}'. Check your headers.")
    return mapping

def auto_distribute_currents(L: np.ndarray, If: float, mode: str):
    if mode == "Equally":
        return np.full_like(L, If / len(L), dtype=float)
    w = L / (L.sum() if L.sum() > 0 else 1.0)
    return If * w

# -------- Sidebar controls --------
with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])
    rho = st.number_input("Soil resistivity ρ (Ω·m)", min_value=1.0, value=100.0, step=1.0)
    Ifault = st.number_input("Total fault current If (A)", min_value=0.0, value=25000.0, step=100.0)

    st.divider()
    st.subheader("Grid / Isosurface")
    nx = st.slider("X samples", 15, 80, 35)
    ny = st.slider("Y samples", 15, 80, 35)
    nz = st.slider("Z samples (depth)", 8, 60, 25)
    surf_count = st.slider("Isosurface count", 2, 8, 4)
    iso_max_frac = st.slider("Isomax (fraction of Vmax)", 10, 90, 40) / 100.0

    st.divider()
    st.subheader("Currents")
    scale_to_If = st.checkbox("Scale/normalize currents to sum = Ifault", value=True)
    fill_missing = st.selectbox("If current column missing/NaN:", ["Equally", "Proportional to length"])
    radius_mm = st.checkbox("Radius column is in mm (convert to m)", value=True)

# -------- Load data --------
if not uploaded:
    st.info("Upload your table. Expected columns (flexible names ok): "
            "x1,y1,z1,x2,y2,z2,length,radius,current,electrode")
    st.stop()

df = pd.read_excel(uploaded) if uploaded.name.lower().endswith(".xlsx") else pd.read_csv(uploaded)
st.write("### Raw columns:", list(df.columns))

mapping = map_columns(df)
req = ["x1","y1","z1","x2","y2","z2"]
if any(k not in mapping for k in req):
    st.error("Missing required geometry columns. Please fix headers and retry.")
    st.stop()

# Extract columns with fallbacks
X1 = df[mapping["x1"]].to_numpy(dtype=float)
Y1 = df[mapping["y1"]].to_numpy(dtype=float)
Z1 = df[mapping["z1"]].to_numpy(dtype=float)
X2 = df[mapping["x2"]].to_numpy(dtype=float)
Y2 = df[mapping["y2"]].to_numpy(dtype=float)
Z2 = df[mapping["z2"]].to_numpy(dtype=float)

L = None
if "length" in mapping:
    L = np.abs(df[mapping["length"]].to_numpy(dtype=float))
else:
    L = np.sqrt((X2-X1)**2 + (Y2-Y1)**2 + (Z2-Z1)**2)

Rcol = np.full(len(df), 0.05)  # default 50 mm
if "radius" in mapping:
    Rcol = np.abs(df[mapping["radius"]].to_numpy(dtype=float))
    if radius_mm:
        Rcol = Rcol / 1000.0

# Electrode labels
labels = None
if "electrode" in mapping:
    labels = df[mapping["electrode"]].astype(str).tolist()
else:
    labels = [f"SEG{i+1}" for i in range(len(df))]

# Currents
Iseg = np.full(len(df), np.nan)
if "current" in mapping:
    Iseg = pd.to_numeric(df[mapping["current"]], errors="coerce").to_numpy()

# Fill/scale currents
missing = np.isnan(Iseg)
if missing.any():
    Iseg[missing] = auto_distribute_currents(L[missing], Ifault, fill_missing)
if scale_to_If and Iseg.sum() > 0:
    Iseg = Iseg * (Ifault / Iseg.sum())

# Midpoints for fast potential superposition
XM = 0.5*(X1+X2); YM = 0.5*(Y1+Y2); ZM = 0.5*(Z1+Z2)

# -------- Plot bounds --------
pad = 0.15 * max(
    (np.max([X1.max(), X2.max()]) - np.min([X1.min(), X2.min()])),
    (np.max([Y1.max(), Y2.max()]) - np.min([Y1.min(), Y2.min()]))
)
xmin = min(X1.min(), X2.min()) - pad
xmax = max(X1.max(), X2.max()) + pad
ymin = min(Y1.min(), Y2.min()) - pad
ymax = max(Y1.max(), Y2.max()) + pad
zmax = max(Z1.max(), Z2.max(), 0.0)  # depth downwards (positive)
zmin = 0.0  # surface

# 3D grid (surface z=0 to depth zmax)
X, Y, Z = np.mgrid[
    xmin:xmax:complex(nx),
    ymin:ymax:complex(ny),
    zmin:zmax:complex(nz)
]

# -------- Potential field (simple 1/r superposition using midpoints) --------
V = np.zeros_like(X, dtype=float)
for xm, ym, zm, I in zip(XM, YM, ZM, Iseg):
    R = np.sqrt((X - xm)**2 + (Y - ym)**2 + (Z - zm)**2)
    R = np.maximum(R, 0.15)  # avoid singularities ~15 cm
    V += (rho * I) / (4.0 * np.pi * R)

# -------- Figure --------
fig = go.Figure()

# Electrodes (3D lines)
for i in range(len(df)):
    fig.add_trace(go.Scatter3d(
        x=[X1[i], X2[i]], y=[Y1[i], Y2[i]], z=[Z1[i], Z2[i]],
        mode="lines+markers",
        line=dict(width=max(2.0, 1000*Rcol[i])),  # scale radius for visibility
        marker=dict(size=3),
        name=f"{labels[i]} (I={Iseg[i]:.2f} A)"
    ))

# Equipotential isosurfaces
Vmax = float(np.nanmax(V))
fig.add_trace(go.Isosurface(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=V.flatten(),
    isomin=0.0,
    isomax=Vmax * iso_max_frac,
    surface_count=surf_count,
    colorscale="Viridis",
    opacity=0.6,
    caps=dict(x_show=False, y_show=False, z_show=False),
    name="Equipotential"
))

# Ground surface (z=0) as a translucent plane
fig.add_trace(go.Surface(
    x=np.linspace(xmin, xmax, 2),
    y=np.linspace(ymin, ymax, 2),
    z=np.zeros((2, 2)),
    showscale=False,
    opacity=0.2,
    name="Ground"
))

fig.update_layout(
    scene=dict(
        xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m, down)",
        aspectmode="data"
    ),
    legend=dict(itemsizing="constant"),
    title="3D Electrode Geometry + Equipotential Isosurfaces"
)

st.plotly_chart(fig, use_container_width=True)

# Helpful debug panel
with st.expander("Column mapping (debug)"):
    st.write("Normalized → Original mapping used:", mapping)
    st.write("Sum of segment currents (A):", float(Iseg.sum()))
