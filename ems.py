import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="BESS Plant Simulator (Realistic)", layout="wide")
st.title("BESS Plant-level Simulator — per-container SOC & realistic dispatch")

st.markdown("""
**What this app does (realistic plant-level simulation)**

- Per-container SOC tracking (N containers) with BMS override
- Charge and discharge scheduling based on POI peaks and low-price windows
- Per-container C-rate limits, separate charge/discharge efficiency
- Generates: BESS power (charge/discharge), net POI, SOC curves, per-container histograms
- Not an MPC — deterministic greedy scheduler for teaching/prototyping
""")

# ---------------------------
# Inputs & file upload
# ---------------------------
uploaded = st.file_uploader("Upload POI CSV (must contain 'time' ISO column and 'power_kW' numeric)", type=["csv"])
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

# infer timestep hours
if len(poi) >= 2:
    dt_hours = (poi.loc[1, "time"] - poi.loc[0, "time"]).total_seconds() / 3600.0
else:
    dt_hours = 1.0

# ---------------------------
# Sidebar: plant & model params
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
roundtrip = charge_eff * discharge_eff
cont_c_rate = st.sidebar.number_input("Container max C-rate (1/h)", value=0.5, min_value=0.05, max_value=2.0, step=0.01)
soc_min_pct = st.sidebar.number_input("SOC min (%)", value=10.0, min_value=0.0, max_value=50.0) / 100.0
soc_max_pct = st.sidebar.number_input("SOC max (%)", value=100.0, min_value=50.0, max_value=100.0) / 100.0

st.sidebar.header("Dispatch & detection")
baseline_window = st.sidebar.number_input("Baseline rolling window (hours)", value=24, min_value=1)
peak_threshold_kW = st.sidebar.number_input("Peak detection threshold (kW above baseline)", value=500.0)
charge_threshold_kW = st.sidebar.number_input("Charge opportunity threshold (kW below baseline)", value=-200.0)  # negative

# ---------------------------
# Derived params and validation
# ---------------------------
P_site_kW = P_site_MW * 1000.0
per_container_pmax = P_site_kW / n_containers  # simple equal power rating
per_container_energy_nameplate = container_MWh
soh_list = [float(x.strip()) for x in soh_text.split(",") if x.strip()]
initial_soh = soh_list[0] if len(soh_list) > 0 else 1.0

# compute per-container usable initial MWh from E_site_MWh proportionally if mismatch
# Prefer using explicit container MWh * SOH if provided, else distribute E_site across containers
# We'll use container_MWh * initial_soh as per-container usable; and recompute usable_total (more realistic)
usable_per_container = container_MWh * initial_soh
usable_total = usable_per_container * n_containers

st.write("**Simulation inputs**")
st.write(f"- POI timesteps: {len(poi)} rows, dt (hours) ≈ {dt_hours:.3f}")
st.write(f"- Plant rating: {P_site_MW} MW , usable (initial build based on containers): {usable_total:.2f} MWh")
st.write(f"- Containers: {n_containers} × {container_MWh:.2f} MWh nameplate; initial SOH {initial_soh:.3f}")
st.write(f"- Per-container max power (theoretical equal split): {per_container_pmax:.1f} kW")
st.write(f"- Per-container max power by C-rate: {cont_c_rate * per_container_energy_nameplate * 1000:.1f} kW")
st.write(f"- Charge eff: {charge_eff:.3f}, Disch eff: {discharge_eff:.3f}, Roundtrip ≈ {roundtrip:.3f}")

# ---------------------------
# Compute baseline & detect peaks
# ---------------------------
poi["baseline_kW"] = poi["power_kW"].rolling(window=baseline_window, min_periods=1, center=True).median()
poi["excess_kW"] = poi["power_kW"] - poi["baseline_kW"]
poi["is_peak"] = poi["excess_kW"] > peak_threshold_kW
poi["is_charge_op"] = poi["excess_kW"] < charge_threshold_kW  # good times to charge

# detect peak blocks (continuous True segments)
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
# finalize
if in_block and start_idx is not None:
    blocks.append({"start_idx": start_idx, "end_idx": len(poi)-1, "E_req_kWh": acc_kwh})

# build a DataFrame of blocks
blocks_df = pd.DataFrame(blocks)
if blocks_df.empty:
    st.warning("No peak blocks detected with current threshold. Adjust threshold or baseline window.")
else:
    st.subheader("Detected Peak Blocks")
    # convert to MWh for display
    blocks_df["E_req_MWh"] = blocks_df["E_req_kWh"] / 1000.0
    blocks_df["start_time"] = blocks_df["start_idx"].apply(lambda i: poi.loc[i,"time"])
    blocks_df["end_time"] = blocks_df["end_idx"].apply(lambda i: poi.loc[i,"time"])
    st.dataframe(blocks_df[["start_time","end_time","E_req_MWh"]])

# ---------------------------
# Scheduling: greedy allocation + charge windows
# ---------------------------
# Initialize container SOC arrays (in fraction of nameplate energy)
soc = np.full(shape=(n_containers,), fill_value=(soc_max_pct + soc_min_pct) / 2.0)  # start in mid-window
# But ensure total initial energy matches usable_total distribution: we'll scale soc to match usable_total within soc bounds
# Compute initial total energy from soc * nameplate * soh
initial_total_energy = (container_MWh * initial_soh) * n_containers * ((soc_max_pct + soc_min_pct) / 2.0)
# If initial_total_energy differs significantly from usable_total * midpoint, it's fine — user provided container sizing used above.

# For simulation we'll track per-container SOC (fraction), per-container energy (kWh)
container_nameplate_kWh = container_MWh * 1000.0
container_usable_kWh = container_nameplate_kWh * initial_soh  # usable per container at t0 (kWh)
# enforce soc such that actual container energy equals container_usable_kWh * midpoint fraction
soc[:] = ( (container_usable_kWh * ((soc_max_pct + soc_min_pct)/2.0)) / container_nameplate_kWh )

# time series outputs
poi["bess_power_kW"] = 0.0   # positive discharge, negative charge
poi["bess_power_per_container_kW"] = 0.0
poi["soc_total_pct"] = 0.0

# We'll follow this simple approach:
# 1. For each detected peak block in chronological order:
#    a. Compute E_req (kWh). Compute available discharge energy across containers (sum((soc - soc_min)*nameplate_kWh*soh)*discharge_eff)
#    b. Allocate up to available. Compute dispatch_kW time series within block proportional to excess_kW, scaled to allocated energy.
# 2. After assigning discharges, identify charge opportunities (is_charge_op) before each block (prefer earlier same day), and schedule charging to refill SOC (respecting charge limits and efficiency).
# 3. Simulate SOC forward in time applying dispatch and charge actions with per-container splitting and BMS overrides.

# Pre-allocate arrays for scheduled dispatch (kW) and scheduled charge (kW negative)
poi["scheduled_dispatch_kW"] = 0.0  # positive values for planned discharge
poi["scheduled_charge_kW"] = 0.0    # positive values for planned charge (kW)

# compute available discharge energy right now (kWh) across fleet within SOC window:
def total_available_discharge_kwh(soc_array):
    # energy above soc_min that can be discharged (kWh)
    avail = np.sum(np.maximum(soc_array - soc_min_pct, 0.0) * container_nameplate_kWh * initial_soh)
    # account for discharge efficiency (energy at bus = avail * discharge_eff)
    return avail * discharge_eff

# allocate discharges to blocks greedily
for _, blk in (blocks_df.iterrows() if (not blocks_df.empty) else []):
    start = int(blk["start_idx"])
    end = int(blk["end_idx"])
    E_req_kWh = blk["E_req_kWh"]
    avail_kWh = total_available_discharge_kwh(soc)
    take_kWh = min(E_req_kWh, avail_kWh)
    if take_kWh <= 0:
        continue
    # distribute scheduled dispatch during this block in proportion to excess_kW
    mask = (poi.index >= start) & (poi.index <= end)
    excess = poi.loc[mask, "excess_kW"].values
    total_excess_kwh = (excess * dt_hours).sum()
    if total_excess_kwh <= 0:
        continue
    frac = take_kWh / total_excess_kwh
    poi.loc[mask, "scheduled_dispatch_kW"] += poi.loc[mask, "excess_kW"] * frac

# Now schedule charging opportunistically:
# We'll try to refill energy used by dispatch by using prior negative-excess windows (is_charge_op) in a simple left-to-right pass.
energy_deficit_kWh = poi["scheduled_dispatch_kW"].sum() * dt_hours / 1000.0  # MWh -> careful: scheduled_dispatch_kW in kW, sum*dt_hours -> kWh
energy_deficit_kWh *= 1000.0  # convert to kWh (total energy we scheduled to discharge)
# But note: scheduled dispatch is energy at bus; to charge that amount we need more at AC due to roundtrip
required_charge_input_kWh = energy_deficit_kWh / (charge_eff)  # kWh input into battery (approx)
# We'll fill charge windows in chronological order up to required amount
remaining_charge_kWh = required_charge_input_kWh

for idx, row in poi.iterrows():
    if remaining_charge_kWh <= 0:
        break
    if not row["is_charge_op"]:
        continue
    # available charge power (kW) approximated as -excess_kW (positive when negative)
    available_kW = -row["excess_kW"]
    if available_kW <= 0:
        continue
    # but container charging limit
    per_container_pmax_charge_kW = min(per_container_pmax, cont_c_rate * container_nameplate_kWh)
    plant_max_charge_kW = per_container_pmax_charge_kW * n_containers
    take_kW = min(available_kW, plant_max_charge_kW)
    # limit by remaining energy required (kWh) and dt_hours
    max_possible_kWh = take_kW * dt_hours
    take_kWh = min(max_possible_kWh, remaining_charge_kWh)
    # schedule charge (store as positive in scheduled_charge_kW)
    poi.at[idx, "scheduled_charge_kW"] += take_kW / dt_hours
    remaining_charge_kWh -= take_kWh

# Now run the time-forward SOC simulation applying scheduled charge/discharge and enforce per-container limits and BMS overrides
# We'll allocate per-container power proportionally to available usable energy and enforce per-container C-rate and SOC min/max.
soc_time = []
bess_power_series = []
per_container_power_series = []

# initial SOC already in `soc`
for t_idx, row in poi.iterrows():
    dispatch_kW = float(row["scheduled_dispatch_kW"])   # positive discharge
    charge_kW = float(row["scheduled_charge_kW"])      # positive scheduled charge (kW into battery)
    net_bess_kW = 0.0
    # First apply charge (positive charging reduces SOC if we treat sign convention: we'll use positive -> charging into battery)
    if charge_kW > 0:
        # distribute charge across containers proportional to (soc_max - soc)
        need_per_container_kWh = np.maximum(soc_max_pct - soc, 0.0) * container_nameplate_kWh
        total_need_kWh = need_per_container_kWh.sum()
        if total_need_kWh <= 0:
            # no room to charge: skip
            applied_charge_kW = 0.0
        else:
            # decide actual plant-level max (C-rate & per-PCS)
            per_container_pmax_charge_kW = min(per_container_pmax, cont_c_rate * container_nameplate_kWh)
            plant_max_charge_kW = per_container_pmax_charge_kW * n_containers
            applied_charge_kW = min(charge_kW, plant_max_charge_kW)
            # distribute to containers proportionally to need (kWh)
            share = need_per_container_kWh / (total_need_kWh + 1e-12)
            # per-container charge power
            per_cont_power = share * applied_charge_kW
            # enforce per-container pmax
            per_cont_power = np.minimum(per_cont_power, per_container_pmax_charge_kW)
            # apply to SOC: incoming energy to battery = sum(per_cont_power) * dt_hours * charge_eff (kWh)
            energy_in_kWh = per_cont_power.sum() * dt_hours * charge_eff
            # update per-container SOC by allocated kWh (allocated before eff)
            per_cont_input_kWh = per_cont_power * dt_hours
            # update soc for each container
            soc += (per_cont_input_kWh * charge_eff) / container_nameplate_kWh
            # clip to soc_max
            soc = np.minimum(soc, soc_max_pct)

        net_bess_kW -= applied_charge_kW  # negative at AC (charging, consuming grid)
    # Then apply discharge
    if dispatch_kW > 0:
        # distribute discharge across containers proportional to (soc - soc_min)
        avail_per_container_kWh = np.maximum(soc - soc_min_pct, 0.0) * container_nameplate_kWh
        total_avail_kWh = avail_per_container_kWh.sum()
        if total_avail_kWh <= 0:
            applied_dispatch_kW = 0.0
        else:
            per_container_pmax_dis_kW = min(per_container_pmax, cont_c_rate * container_nameplate_kWh)
            plant_max_dis_kW = per_container_pmax_dis_kW * n_containers
            # dispatch cannot exceed plant max
            planned_dispatch_kW = min(dispatch_kW, plant_max_dis_kW)
            # share by available energy
            share = avail_per_container_kWh / (total_avail_kWh + 1e-12)
            per_cont_power = share * planned_dispatch_kW
            # enforce per-container pmax
            per_cont_power = np.minimum(per_cont_power, per_container_pmax_dis_kW)
            # energy removed from containers = per_cont_power * dt_hours / discharge_eff (because AC energy is less than battery kWh)
            energy_out_kWh_at_packs = per_cont_power.sum() * dt_hours / discharge_eff
            # convert to SOC decrement per container
            soc_decrement = (per_cont_power * dt_hours / discharge_eff) / container_nameplate_kWh
            soc -= soc_decrement
            # clip soc to soc_min
            low_mask = soc < soc_min_pct
            if low_mask.any():
                # zero out any negative SOC and recompute actual applied dispatch by reducing those containers to soc_min
                # compute deficit energy restored to grid that couldn't be provided
                # For simplicity, set soc to soc_min for those containers and recompute applied dispatch from remaining containers
                soc[low_mask] = soc_min_pct
                # recompute available and redo distribution
                avail_per_container_kWh = np.maximum(soc - soc_min_pct, 0.0) * container_nameplate_kWh
                total_avail_kWh = avail_per_container_kWh.sum()
                if total_avail_kWh <= 0:
                    applied_dispatch_kW = 0.0
                else:
                    share = avail_per_container_kWh / (total_avail_kWh + 1e-12)
                    per_cont_power = share * planned_dispatch_kW
                    per_cont_power = np.minimum(per_cont_power, per_container_pmax_dis_kW)
                    applied_dispatch_kW = per_cont_power.sum()
                    # finalize soc decrement with this per_cont_power
                    soc -= (per_cont_power * dt_hours / discharge_eff) / container_nameplate_kWh
                    # ensure no underflow
                    soc = np.maximum(soc, soc_min_pct)
            else:
                applied_dispatch_kW = per_cont_power.sum()
        net_bess_kW += applied_dispatch_kW  # positive delivered to grid

    # record time-step outputs
    poi.at[t_idx, "bess_power_kW"] = net_bess_kW
    poi.at[t_idx, "bess_power_per_container_kW"] = (net_bess_kW / n_containers)
    poi.at[t_idx, "soc_total_pct"] = (soc.mean() * 100.0)
    soc_time.append(soc.copy())
    bess_power_series.append(net_bess_kW)
    per_container_power_series.append(net_bess_kW / n_containers)

# ---------------------------
# Post-simulation metrics and plots
# ---------------------------
poi["bess_power_kW"] = poi["bess_power_kW"].astype(float)
poi["poi_net_kW"] = poi["power_kW"] - poi["bess_power_kW"]

st.subheader("Time-series plots")

fig, axes = plt.subplots(3, 1, figsize=(12,9), sharex=True)
axes[0].plot(poi["time"], poi["power_kW"], label="Original POI (kW)")
axes[0].plot(poi["time"], poi["baseline_kW"], label="Baseline (kW)", alpha=0.6)
axes[0].plot(poi["time"], poi["poi_net_kW"], label="Net POI after BESS (kW)")
axes[0].legend(loc="upper left")
axes[0].set_ylabel("kW")

axes[1].plot(poi["time"], poi["bess_power_kW"], label="BESS power (positive discharge, negative charge)")
axes[1].axhline(0, color="k", linewidth=0.6)
axes[1].legend(loc="upper left")
axes[1].set_ylabel("kW")

axes[2].plot(poi["time"], poi["soc_total_pct"], label="Average SOC (%)")
# show a sample of 6 containers SOC trajectories (if manageable)
sample_idxs = np.linspace(0, n_containers-1, min(6, n_containers)).astype(int)
for sidx in sample_idxs:
    series = [s[sidx]*100.0 for s in soc_time]
    axes[2].plot(poi["time"], series, alpha=0.7, linestyle="--", label=f"Container {sidx} SOC")
axes[2].legend(loc="upper left")
axes[2].set_ylabel("SOC (%)")

for ax in axes:
    ax.grid(True)
st.pyplot(fig)

st.subheader("Per-container dispatch histogram")
fig2, ax2 = plt.subplots(figsize=(8,3))
ax2.hist(poi["bess_power_per_container_kW"].dropna().values, bins=40)
ax2.set_xlabel("kW per container")
ax2.set_ylabel("counts")
st.pyplot(fig2)

# Energy summary
total_dispatched_kWh = (poi["bess_power_kW"].clip(lower=0).sum() * dt_hours)  # kWh delivered to grid
total_charged_kWh = (-poi["bess_power_kW"].clip(upper=0).sum() * dt_hours)  # kWh absorbed from grid
st.subheader("Energy summary")
st.write(f"- Energy delivered to grid (discharge) : {total_dispatched_kWh/1000.0:.3f} MWh")
st.write(f"- Energy absorbed from grid (charge)   : {total_charged_kWh/1000.0:.3f} MWh")
st.write(f"- Net throughput (roundtrip losses influence) : {(total_charged_kWh - total_dispatched_kWh)/1000.0:.3f} MWh")

st.success("Simulation finished. Review the plots and adjust parameters as needed.")
