import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="BESS Plant Simulator (Realistic)", layout="wide")
st.title("BESS Plant-level Simulator — per-container SOC & realistic dispatch")

st.markdown("""
**This simulator** performs a deterministic plant-level simulation with:
- per-container SOC tracking (N containers),
- opportunistic charging windows and greedy peak shaving,
- C-rate enforcement and BMS-like overrides,
- outputs: BESS power (charge/discharge), net POI, SOC curves, histograms, and energy summary.

Upload a POI CSV (columns: `time` ISO, `power_kW`) or use the built-in synthetic profile.
""")

# ---------------------------
# Input POI
# ---------------------------
uploaded = st.file_uploader("Upload POI CSV (columns: time,power_kW)", type=["csv"])
if uploaded is not None:
    poi = pd.read_csv(uploaded, parse_dates=["time"])
else:
    st.info("No POI uploaded — using synthetic 7-day hourly profile.")
    rng = pd.date_range(start="2025-01-01", periods=24*7, freq="H")
    base = 1000 + 200*np.sin(np.linspace(0, 7*2*np.pi, len(rng)))
    peaks = np.zeros(len(rng))
    for d in range(7):
        peaks[d*24 + 18 : d*24 + 20] += 3000   # 2-hour evening peak
    poi = pd.DataFrame({"time": rng, "power_kW": base + peaks - 1500})

poi = poi.sort_values("time").reset_index(drop=True)
if len(poi) < 2:
    st.error("POI must have at least 2 rows.")
    st.stop()

# timestep in hours
dt_hours = (poi.loc[1, "time"] - poi.loc[0, "time"]).total_seconds() / 3600.0

# ---------------------------
# Sidebar: Plant & Model Params
# ---------------------------
st.sidebar.header("Plant & Container Parameters")
P_site_MW = st.sidebar.number_input("Plant power rating (MW)", value=100.0, min_value=0.1)
E_site_MWh = st.sidebar.number_input("Plant usable energy at t0 (MWh)", value=200.0, min_value=1.0)
n_containers = st.sidebar.number_input("Number of containers", value=44, min_value=1, step=1)
container_MWh = st.sidebar.number_input("Container nameplate MWh", value=5.0, min_value=0.1)
soh_text = st.sidebar.text_area("SOH list (comma-separated years) — first value used for initial run",
                               "0.997,0.98,0.96,0.94,0.92")

st.sidebar.header("Efficiency & Limits")
charge_eff = st.sidebar.number_input("Charge efficiency (0-1)", value=0.96, min_value=0.5, max_value=1.0, step=0.01)
discharge_eff = st.sidebar.number_input("Discharge efficiency (0-1)", value=0.96, min_value=0.5, max_value=1.0, step=0.01)
cont_c_rate = st.sidebar.number_input("Container max C-rate (1/h)", value=0.5, min_value=0.05, max_value=2.0, step=0.01)
soc_min_pct = st.sidebar.number_input("SOC min (%)", value=10.0, min_value=0.0, max_value=50.0) / 100.0
soc_max_pct = st.sidebar.number_input("SOC max (%)", value=100.0, min_value=50.0, max_value=100.0) / 100.0

st.sidebar.header("Dispatch & Detection")
baseline_window = st.sidebar.number_input("Baseline rolling window (hours)", value=24, min_value=1)
peak_threshold_kW = st.sidebar.number_input("Peak detection threshold (kW above baseline)", value=500.0)
charge_threshold_kW = st.sidebar.number_input("Charge opportunity threshold (kW below baseline)", value=-200.0)

# ---------------------------
# Derived params & display
# ---------------------------
P_site_kW = P_site_MW * 1000.0
per_container_pmax = P_site_kW / n_containers
container_nameplate_kWh = container_MWh * 1000.0
soh_list = [float(x.strip()) for x in soh_text.split(",") if x.strip()]
initial_soh = soh_list[0] if len(soh_list) > 0 else 1.0
usable_per_container_kWh = container_nameplate_kWh * initial_soh
usable_total_kWh = usable_per_container_kWh * n_containers

st.write("**Simulation inputs summary**")
st.write(f"- POI rows: {len(poi)}, dt ≈ {dt_hours:.3f} h")
st.write(f"- Plant rating: {P_site_MW} MW | usable (per containers): {usable_total_kWh/1000.0:.2f} MWh")
st.write(f"- Containers: {n_containers} × {container_MWh:.2f} MWh nameplate; initial SOH {initial_soh:.3f}")
st.write(f"- Theoretical per-container pmax (equal split): {per_container_pmax:.1f} kW")
st.write(f"- Per-container C-rate power: {cont_c_rate * container_nameplate_kWh:.1f} kW")

# ---------------------------
# Baseline & detections
# ---------------------------
poi["baseline_kW"] = poi["power_kW"].rolling(window=baseline_window, min_periods=1, center=True).median()
poi["excess_kW"] = poi["power_kW"] - poi["baseline_kW"]
poi["is_peak"] = poi["excess_kW"] > peak_threshold_kW
poi["is_charge_op"] = poi["excess_kW"] < charge_threshold_kW

st.subheader("POI preview (head)")
st.dataframe(poi.head(48))

# ---------------------------
# Simulation button
# ---------------------------
run = st.button("Run simulation")
if not run:
    st.info("Adjust parameters and click **Run simulation** to perform the realistic simulation.")
    st.stop()

# ---------------------------
# On click: perform realistic scheduling + SOC sim
# ---------------------------
progress = st.progress(0)
progress.step(5)

# 1) detect peak blocks (continuous True segments)
blocks = []
in_block = False
start_idx = None
acc_kwh = 0.0
for idx, row in poi.iterrows():
    if row["is_peak"] and not in_block:
        in_block = True
        start_idx = idx
        acc_kwh = row["excess_kW"] * dt_hours
    elif row["is_peak"] and in_block:
        acc_kwh += row["excess_kW"] * dt_hours
    elif not row["is_peak"] and in_block:
        end_idx = idx - 1
        blocks.append({"start_idx": start_idx, "end_idx": end_idx, "E_req_kWh": acc_kwh})
        in_block = False
        start_idx = None
        acc_kwh = 0.0
if in_block and start_idx is not None:
    blocks.append({"start_idx": start_idx, "end_idx": len(poi)-1, "E_req_kWh": acc_kwh})

blocks_df = pd.DataFrame(blocks)
progress.step(15)

# 2) initialize SOCs (start at midpoint of SOC window mapped to usable energy)
# set initial SOC so actual usable energy equals usable_per_container_kWh * (soc_max_pct - soc_min_pct) fraction wise
initial_soc_fraction = (soc_max_pct + soc_min_pct) / 2.0
soc = np.full(shape=(n_containers,), fill_value=initial_soc_fraction)
# ensure per-container usable (kWh) is container_nameplate_kWh * initial_soh * (soc fraction)
# We'll use container_nameplate_kWh * initial_soh as "usable at 100%SOC" baseline

progress.step(25)

# Prepare schedule arrays
poi["scheduled_dispatch_kW"] = 0.0
poi["scheduled_charge_kW"] = 0.0

def total_available_discharge_kwh(soc_array):
    avail = np.sum(np.maximum(soc_array - soc_min_pct, 0.0) * container_nameplate_kWh * initial_soh)
    return avail * discharge_eff

# greedy allocation for each block (chronological)
for _, blk in (blocks_df.iterrows() if (not blocks_df.empty) else []):
    start = int(blk["start_idx"])
    end = int(blk["end_idx"])
    E_req_kWh = blk["E_req_kWh"]
    avail_kWh = total_available_discharge_kwh(soc)
    take_kWh = min(E_req_kWh, avail_kWh)
    if take_kWh <= 0:
        continue
    mask = (poi.index >= start) & (poi.index <= end)
    excess = poi.loc[mask, "excess_kW"].values
    total_excess_kwh = (excess * dt_hours).sum()
    if total_excess_kwh <= 0:
        continue
    frac = take_kWh / total_excess_kwh
    poi.loc[mask, "scheduled_dispatch_kW"] += poi.loc[mask, "excess_kW"] * frac

progress.step(45)

# schedule charging opportunistically to refill energy used
energy_dispatched_kWh = (poi["scheduled_dispatch_kW"].clip(lower=0).sum() * dt_hours)  # kWh dispatched to grid
# to refill battery (accounting for discharge_eff), need more input; approximate required input energy into battery:
# dispatched kWh is energy delivered at AC; battery energy used (pack kWh) = dispatched_kWh / discharge_eff
pack_kwh_used = energy_dispatched_kWh / discharge_eff
# input energy required from grid (AC charging kWh) = pack_kwh_used / charge_eff
required_charge_input_kWh = pack_kwh_used / charge_eff
remaining_charge_kWh = required_charge_input_kWh

# Fill prior and subsequent charge windows (simple greedy left-to-right)
for idx, row in poi.iterrows():
    if remaining_charge_kWh <= 0:
        break
    if not row["is_charge_op"]:
        continue
    available_kW = -row["excess_kW"]
    if available_kW <= 0:
        continue
    per_container_pmax_charge_kW = min(per_container_pmax, cont_c_rate * container_nameplate_kWh)
    plant_max_charge_kW = per_container_pmax_charge_kW * n_containers
    take_kW = min(available_kW, plant_max_charge_kW)
    max_possible_kWh = take_kW * dt_hours
    take_kWh = min(max_possible_kWh, remaining_charge_kWh)
    if take_kWh <= 0:
        continue
    poi.at[idx, "scheduled_charge_kW"] += take_kWh / dt_hours
    remaining_charge_kWh -= take_kWh

progress.step(60)

# Forward SOC simulation applying scheduled charge/discharge
poi["bess_power_kW"] = 0.0
poi["bess_power_per_container_kW"] = 0.0
poi["soc_total_pct"] = 0.0

for t_idx, row in poi.iterrows():
    dispatch_kW = float(row["scheduled_dispatch_kW"])   # positive discharge
    charge_kW = float(row["scheduled_charge_kW"])      # positive charging (kW into battery)
    net_bess_kW = 0.0

    # CHARGE
    if charge_kW > 0:
        need_per_container_kWh = np.maximum(soc_max_pct - soc, 0.0) * container_nameplate_kWh
        total_need_kWh = need_per_container_kWh.sum()
        if total_need_kWh > 0:
            per_container_pmax_charge_kW = min(per_container_pmax, cont_c_rate * container_nameplate_kWh)
            plant_max_charge_kW = per_container_pmax_charge_kW * n_containers
            applied_charge_kW = min(charge_kW, plant_max_charge_kW)
            share = need_per_container_kWh / (total_need_kWh + 1e-12)
            per_cont_power = share * applied_charge_kW
            per_cont_power = np.minimum(per_cont_power, per_container_pmax_charge_kW)
            per_cont_input_kWh = per_cont_power * dt_hours
            # update soc with efficiency
            soc += (per_cont_input_kWh * charge_eff) / container_nameplate_kWh
            soc = np.minimum(soc, soc_max_pct)
            net_bess_kW -= per_cont_power.sum()

    # DISPATCH
    if dispatch_kW > 0:
        avail_per_container_kWh = np.maximum(soc - soc_min_pct, 0.0) * container_nameplate_kWh
        total_avail_kWh = avail_per_container_kWh.sum()
        if total_avail_kWh > 0:
            per_container_pmax_dis_kW = min(per_container_pmax, cont_c_rate * container_nameplate_kWh)
            plant_max_dis_kW = per_container_pmax_dis_kW * n_containers
            planned_dispatch_kW = min(dispatch_kW, plant_max_dis_kW)
            share = avail_per_container_kWh / (total_avail_kWh + 1e-12)
            per_cont_power = share * planned_dispatch_kW
            per_cont_power = np.minimum(per_cont_power, per_container_pmax_dis_kW)
            # compute pack energy removed (kWh)
            energy_out_kWh_at_packs = per_cont_power.sum() * dt_hours / discharge_eff
            soc_decrement = (per_cont_power * dt_hours / discharge_eff) / container_nameplate_kWh
            soc -= soc_decrement
            # if any dropped below min, clamp and recompute applied dispatch
            if (soc < soc_min_pct).any():
                soc = np.maximum(soc, soc_min_pct)
                avail_per_container_kWh = np.maximum(soc - soc_min_pct, 0.0) * container_nameplate_kWh
                total_avail_kWh = avail_per_container_kWh.sum()
                if total_avail_kWh > 0:
                    share = avail_per_container_kWh / (total_avail_kWh + 1e-12)
                    per_cont_power = share * planned_dispatch_kW
                    per_cont_power = np.minimum(per_cont_power, per_container_pmax_dis_kW)
                    applied_dispatch_kW = per_cont_power.sum()
                    soc -= (per_cont_power * dt_hours / discharge_eff) / container_nameplate_kWh
                    soc = np.maximum(soc, soc_min_pct)
                else:
                    applied_dispatch_kW = 0.0
            else:
                applied_dispatch_kW = per_cont_power.sum()
            net_bess_kW += applied_dispatch_kW

    # record
    poi.at[t_idx, "bess_power_kW"] = net_bess_kW
    poi.at[t_idx, "bess_power_per_container_kW"] = (net_bess_kW / n_containers)
    poi.at[t_idx, "soc_total_pct"] = soc.mean() * 100.0

progress.step(85)

# Post-sim metrics
poi["poi_net_kW"] = poi["power_kW"] - poi["bess_power_kW"]

# plots: time-series (POI orig, net POI, BESS power, SOC)
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].plot(poi["time"], poi["power_kW"], label="Original POI (kW)")
axes[0].plot(poi["time"], poi["baseline_kW"], label="Baseline (kW)", alpha=0.6)
axes[0].plot(poi["time"], poi["poi_net_kW"], label="Net POI after BESS (kW)")
axes[0].legend(loc="upper left")
axes[0].set_ylabel("kW")
axes[0].grid(True)

axes[1].plot(poi["time"], poi["bess_power_kW"], label="BESS power (positive discharge, negative charge)")
axes[1].axhline(0, color="k", linewidth=0.6)
axes[1].legend(loc="upper left")
axes[1].set_ylabel("kW")
axes[1].grid(True)

axes[2].plot(poi["time"], poi["soc_total_pct"], label="Average SOC (%)", linewidth=2)
# sample up to 6 container SOC traces from recorded state snapshots (we tracked only average in table, so approximate by distributing average)
axes[2].set_ylabel("SOC (%)")
axes[2].legend(loc="upper left")
axes[2].grid(True)

st.subheader("Time-series: POI, Net POI, BESS & SOC")
st.pyplot(fig)

# histogram and energy summary
fig2, ax2 = plt.subplots(figsize=(9,3))
ax2.hist(poi["bess_power_per_container_kW"].dropna().values, bins=40)
ax2.set_xlabel("kW per container")
ax2.set_ylabel("counts")
st.subheader("Per-container dispatch histogram")
st.pyplot(fig2)

total_dispatched_kWh = (poi["bess_power_kW"].clip(lower=0).sum() * dt_hours)
total_charged_kWh = (-poi["bess_power_kW"].clip(upper=0).sum() * dt_hours)
st.subheader("Energy summary")
st.write(f"- Energy delivered to grid (discharge) : {total_dispatched_kWh/1000.0:.3f} MWh")
st.write(f"- Energy absorbed from grid (charge)   : {total_charged_kWh/1000.0:.3f} MWh")
st.write(f"- Net pack energy used (approx, accounting eff): {(total_charged_kWh - total_dispatched_kWh)/1000.0:.3f} MWh")

progress.step(100)
st.success("Simulation finished.")

# allow download of results CSV
csv_buf = io.StringIO()
poi.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode()
st.download_button("Download simulation results CSV", data=csv_bytes, file_name="bess_sim_results.csv", mime="text/csv")
