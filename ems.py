import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title='BESS Plant Simulator', layout='wide')

st.title('BESS Plant-level Simulator (100 MW / 200 MWh Example)')

st.markdown("""
This Streamlit app simulates a simplified BESS plant EMS for peak shaving.
- Plant example: 100 MW / 200 MWh
- 44 containers × 5 MWh each
- Detects peak windows above baseline and allocates energy
- Computes dispatch and per-container power
""")

# --------------------------------------------------------
# 1. File Upload or Synthetic POI
# --------------------------------------------------------
uploaded = st.file_uploader("Upload POI CSV (columns: time, power_kW)", type=['csv'])

if uploaded:
    poi = pd.read_csv(uploaded, parse_dates=["time"])
else:
    st.info("No POI uploaded → Using synthetic 7-day profile.")
    rng = pd.date_range(start="2025-01-01", periods=24*7, freq="H")
    base = 1000 + 200*np.sin(np.linspace(0, 7*2*np.pi, len(rng)))
    peaks = np.zeros(len(rng))
    for d in range(7):
        peaks[d*24 + 18 : d*24 + 20] += 3000   # two-hour peak
    poi = pd.DataFrame({"time": rng, "power_kW": base + peaks - 1500})

# --------------------------------------------------------
# 2. Sidebar Config Panel
# --------------------------------------------------------
st.sidebar.header("Plant Parameters")

P_site = st.sidebar.number_input("Plant MW rating", value=100.0)
E_site = st.sidebar.number_input("Plant usable MWh", value=200.0)
n_cont = st.sidebar.number_input("Number of containers", value=44)
nameplate = st.sidebar.number_input("Container MWh", value=5.0)
eta = st.sidebar.number_input("Roundtrip efficiency", value=0.92)
soc_min = st.sidebar.number_input("Min SOC (%)", value=10) / 100
soc_max = st.sidebar.number_input("Max SOC (%)", value=100) / 100
threshold = st.sidebar.number_input("Peak detection threshold (kW above baseline)", value=500.0)

st.sidebar.header("SOH settings (optional)")
soh_text = st.sidebar.text_area(
    "SOH list (first year SOH used for simulation):",
    "0.997,0.98,0.96,0.94,0.92"
)

soh_list = [float(x) for x in soh_text.split(",") if x.strip()]
SOH = soh_list[0] if len(soh_list) else 1.0

usable_per_container = nameplate * SOH
usable_total = usable_per_container * n_cont

st.write(f"**Usable Energy (initial)**: {usable_total:.2f} MWh")

# --------------------------------------------------------
# 3. Baseline & Peak Detection
# --------------------------------------------------------
poi = poi.sort_values("time").reset_index(drop=True)
poi["baseline_kW"] = poi["power_kW"].rolling(24, min_periods=1, center=True).median()
poi["excess_kW"] = poi["power_kW"] - poi["baseline_kW"]
poi["is_peak"] = poi["excess_kW"] > threshold

st.subheader("POI Preview")
st.dataframe(poi.head(48))

# Peak block detection
peak_summary = poi.groupby((poi["is_peak"] != poi["is_peak"].shift()).cumsum()).agg(
    start=("time", "first"),
    end=("time", "last"),
    flag=("is_peak", "first"),
    peak_max=("excess_kW", "max")
)

peaks = peak_summary[peak_summary["flag"] == True].reset_index(drop=True)

st.subheader("Detected Peak Windows")
st.dataframe(peaks)

# --------------------------------------------------------
# 4. Energy Feasibility & Allocation
# --------------------------------------------------------
E_available = usable_total * (soc_max - soc_min) * eta
st.write(f"**Available dispatchable energy:** {E_available:.2f} MWh")

E_req_list = []
for _, r in peaks.iterrows():
    mask = (poi["time"] >= r["start"]) & (poi["time"] <= r["end"])
    E_req_MWh = poi.loc[mask, "excess_kW"].sum() / 1000  # kWh → MWh
    E_req_list.append(E_req_MWh)

peaks["E_req_MWh"] = E_req_list

allocated = []
remaining = E_available

for req in E_req_list:
    take = min(req, remaining)
    allocated.append(take)
    remaining -= take

peaks["E_alloc_MWh"] = allocated

st.subheader("Energy Allocation to Peaks")
st.dataframe(peaks)

# --------------------------------------------------------
# 5. Dispatch Time Series
# --------------------------------------------------------
poi["dispatch_kW"] = 0.0

for _, row in peaks.iterrows():
    mask = (poi["time"] >= row["start"]) & (poi["time"] <= row["end"])
    tot = poi.loc[mask, "excess_kW"].sum()
    if tot > 0 and row["E_alloc_MWh"] > 0:
        frac = (row["E_alloc_MWh"] * 1000) / tot
        poi.loc[mask, "dispatch_kW"] = poi.loc[mask, "excess_kW"] * frac

poi["dispatch_per_container_kW"] = poi["dispatch_kW"] / n_cont

# --------------------------------------------------------
# 6. Plotting
# --------------------------------------------------------
st.subheader("POI & Dispatch Plot")
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(poi["time"], poi["power_kW"], label="POI")
ax.plot(poi["time"], poi["baseline_kW"], label="Baseline")
ax.plot(poi["time"], poi["dispatch_kW"], label="BESS Dispatch")
ax.legend()
st.pyplot(fig)

st.subheader("Dispatch Per Container Distribution")
fig2, ax2 = plt.subplots(figsize=(6,3))
ax2.hist(poi["dispatch_per_container_kW"], bins=30)
ax2.set_xlabel("kW per container")
st.pyplot(fig2)

st.success("Simulation complete.")
