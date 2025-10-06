# bess_arbitrage_soh_realtime.py
# LIVE (in-place) chart updates for BESS arbitrage + SOH/Degradation.
# No st.rerun loops; we update the charts within a single run using add_rows.

import time
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Utilities
# ----------------------------
def make_synthetic_prices(n=288, market="CAISO", seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)

    base = 35 + 5*np.sin(2*np.pi*(t/n))  # mild diurnal

    if market == "CAISO":
        midday  = -18*np.exp(-0.5*((t - 0.55*n)/(0.10*n))**2)
        evening =  30*np.exp(-0.5*((t - 0.80*n)/(0.06*n))**2)
        noise   = rng.normal(0, 1.5, n)
        p = base + midday + evening + noise
    else:  # ERCOT-like
        spikes = np.zeros(n)
        for _ in range(6):
            ctr = rng.integers(int(0.2*n), int(0.95*n))
            amp = rng.uniform(50, 150)
            wid = rng.uniform(0.01*n, 0.03*n)
            spikes += amp*np.exp(-0.5*((t-ctr)/wid)**2)
        noise = rng.normal(0, 2.2, n)
        p = base + spikes + noise

    return np.clip(p, 0, None)

def quantile_policy_thresholds(price, cheap_q=0.35, rich_q=0.70):
    cheap = float(np.quantile(price, cheap_q))
    rich  = float(np.quantile(price, rich_q))
    if cheap >= rich:
        cheap, rich = rich - 1e-3, rich + 1e-3
    return cheap, rich

def limit_by_c_rate(power_req_mw, e_mwh, c_rate):
    max_abs = float(c_rate) * float(e_mwh)
    return float(np.clip(power_req_mw, -max_abs, max_abs))

def step_dispatch(
    p_t, soc, e_mwh, pmax_mw, eta_c, eta_d,
    cheap_thr, rich_thr, reserve_frac=0.0,
    c_rate=None, dt_h=5/60,
):
    e_free   = e_mwh - soc
    e_stored = soc

    usable_pmax_dis = pmax_mw * (1 - reserve_frac)
    usable_pmax_chg = pmax_mw

    if p_t <= cheap_thr and soc < e_mwh:
        e_in_possible = e_free / eta_c
        p_req = -min(usable_pmax_chg, e_in_possible / dt_h)
    elif p_t >= rich_thr and soc > 0:
        e_out_possible = e_stored * eta_d
        p_req = +min(usable_pmax_dis, e_out_possible / dt_h)
    else:
        p_req = 0.0

    if c_rate is not None and c_rate > 0:
        p_req = limit_by_c_rate(p_req, e_mwh, c_rate)

    if p_req < 0:  # charging
        e_from_grid = -p_req * dt_h
        e_to_store  = min(e_from_grid * eta_c, e_free)
        p_final     = -(e_to_store / eta_c) / dt_h
        soc_new     = soc + e_to_store
    elif p_req > 0:  # discharging
        e_to_grid   = p_req * dt_h
        e_from_store= min(e_to_grid / eta_d, e_stored)
        p_final     = (e_from_store * eta_d) / dt_h
        soc_new     = soc - e_from_store
    else:
        p_final, soc_new = 0.0, soc

    return float(p_final), float(soc_new)

# -------- Degradation (transparent & tweakable) --------
def dod_factor(dod_pu):
    dod_pu = float(np.clip(dod_pu, 0, 1))
    return 0.5 + 0.8 * (dod_pu ** 1.2)

def cycle_deg(delta_e_throughput_mwh, e_mwh, k_cyc):
    if e_mwh <= 1e-9:
        return 0.0
    return 100.0 * float(k_cyc) * (float(delta_e_throughput_mwh) / float(e_mwh))

def calendar_deg(dt_hours, k_cal_per_year, temp_c):
    q10 = 2.0
    scale = q10 ** ((float(temp_c) - 25.0)/10.0)
    return float(k_cal_per_year) * scale * (float(dt_hours) / (24*365))

class DoDTracker:
    def __init__(self):
        self.last_soc_pu = None
    def update(self, soc_mwh, e_mwh):
        if e_mwh <= 1e-9:
            return 0.0
        soc_pu = float(np.clip(soc_mwh / e_mwh, 0, 1))
        if self.last_soc_pu is None:
            self.last_soc_pu = soc_pu
            return 0.0
        delta = abs(soc_pu - self.last_soc_pu)
        self.last_soc_pu = soc_pu
        return float(np.clip(delta, 0, 1))

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="BESS Arbitrage + SOH (Real-Time)", layout="wide")
st.title("BESS Arbitrage + SOH — Real-Time Charts")
st.caption("Live animation via `chart.add_rows()` (no rerun loop).")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Asset & Dispatch")
    e_mwh = st.number_input("Energy (MWh)", 10.0, 5000.0, 400.0, 10.0)
    duration_h = st.number_input("Duration at Pmax (h)", 0.25, 12.0, 4.0, 0.25)
    default_pmax = max(1.0, e_mwh / duration_h)
    pmax_mw = st.number_input("Max Power (MW)", 1.0, 5000.0, default_pmax, 1.0)
    c_rate = st.number_input("C-rate limit (1/h)", 0.05, 5.0, round(pmax_mw/e_mwh, 3), 0.05)
    eta_rt = st.slider("Round-trip efficiency (%)", 50, 98, 90)
    eta_c = float(np.sqrt(eta_rt/100))
    eta_d = float(np.sqrt(eta_rt/100))
    reserve_frac = st.slider("Reserve carve-out (%)", 0, 50, 0) / 100.0

    st.divider()
    st.subheader("Prices")
    mode = st.radio("Source", ["Synthetic (CAISO-like)", "Synthetic (ERCOT-like)", "Upload CSV"], index=0)
    seed = st.number_input("Random seed", 0, 10000, 7, 1)
    n_points = st.select_slider("Horizon (5-min steps)", [144, 288, 576], value=288)
    cheap_q = st.slider("Cheap quantile", 0.05, 0.5, 0.35, 0.05)
    rich_q  = st.slider("Rich quantile", 0.5, 0.95, 0.70, 0.05)

    if mode.startswith("Synthetic"):
        market_kind = "CAISO" if "CAISO" in mode else "ERCOT"
        price = make_synthetic_prices(n_points, market_kind, seed)
    else:
        up = st.file_uploader("CSV with columns: timestamp,price", type=["csv"])
        if up is not None:
            dfp = pd.read_csv(up)
            if "price" not in dfp.columns:
                st.error("CSV must contain 'price' column.")
                st.stop()
            price = dfp["price"].astype(float).to_numpy()
            n_points = len(price)
        else:
            price = make_synthetic_prices(n_points, "CAISO", seed)

    cheap_thr, rich_thr = quantile_policy_thresholds(price, cheap_q, rich_q)

    st.divider()
    st.subheader("Degradation")
    temp_c = st.slider("Cell temperature (°C)", 10, 45, 27)
    k_cal_per_year = st.number_input("Calendar fade @25°C (%/yr)", 0.0, 10.0, 1.2, 0.1)
    k_cyc = st.number_input("Cycle fade coefficient (–)", 0.0, 1.0, 0.08, 0.01)

    st.divider()
    st.subheader("Animation Control")
    dt_min = st.select_slider("Dispatch step (minutes)", [1, 5, 10, 15], value=5)
    dt_h = dt_min / 60.0
    refresh_ms = st.slider("Frame interval (ms)", 50, 1500, 150, 10,
                           help="Lower = faster animation")
    max_steps = st.slider("Steps to run now", 1, 2000, n_points, 1,
                          help="How many time-steps to animate on this start.")

# ---------- State ----------
if "SOC" not in st.session_state:
    st.session_state.SOC = 0.5 * e_mwh
if "SOH" not in st.session_state:
    st.session_state.SOH = 100.0
if "IDX" not in st.session_state:
    st.session_state.IDX = 0
if "THRUPUT" not in st.session_state:
    st.session_state.THRUPUT = 0.0
if "LOG" not in st.session_state:
    st.session_state.LOG = []
if "DOD" not in st.session_state:
    st.session_state.DOD = DoDTracker()

# ---------- Layout scaffolding ----------
left, right = st.columns([2, 1])

with left:
    st.subheader("Live Charts")
    price_chart = st.line_chart(pd.DataFrame({"price": []}))
    power_chart = st.line_chart(pd.DataFrame({"power_mw": []}))
    soc_chart   = st.line_chart(pd.DataFrame({"soc_mwh": []}))
    soh_chart   = st.line_chart(pd.DataFrame({"soh_pct": []}))

with right:
    kpi1 = st.empty()
    kpi2 = st.empty()
    kpi3 = st.empty()
    kpi4 = st.empty()
    st.caption(f"Cheap/Rich thresholds: {cheap_thr:.2f} / {rich_thr:.2f} USD/MWh")
    run = st.button("▶ Start Live Animation")
    reset = st.button("↺ Reset")

# ---------- Reset ----------
if reset:
    st.session_state.SOC = 0.5 * e_mwh
    st.session_state.SOH = 100.0
    st.session_state.IDX = 0
    st.session_state.THRUPUT = 0.0
    st.session_state.LOG = []
    st.session_state.DOD = DoDTracker()
    st.success("State reset.")

# ---------- Live animation (no rerun) ----------
def sim_step(i):
    soh = float(st.session_state.SOH)
    if soh <= 70.0:  # EoL example
        return False

    e_eff = float(e_mwh) * (soh / 100.0)
    soc = float(np.clip(st.session_state.SOC, 0.0, e_eff))

    p_t = float(price[i])
    p_grid_mw, soc_new = step_dispatch(
        p_t=p_t, soc=soc, e_mwh=e_eff, pmax_mw=float(pmax_mw),
        eta_c=float(np.sqrt(eta_rt/100)), eta_d=float(np.sqrt(eta_rt/100)),
        cheap_thr=cheap_thr, rich_thr=rich_thr,
        reserve_frac=float(reserve_frac), c_rate=float(c_rate), dt_h=float(dt_h)
    )

    # Throughput & degradation
    delta_throughput = abs(p_grid_mw) * dt_h
    st.session_state.THRUPUT += delta_throughput

    dod_delta = st.session_state.DOD.update(soc_new, e_eff)
    sev = dod_factor(dod_delta)

    dSOH_cyc = cycle_deg(delta_throughput, e_eff, k_cyc) * sev
    dSOH_cal = calendar_deg(dt_h, k_cal_per_year, temp_c)
    dSOH = dSOH_cyc + dSOH_cal
    soh_new = max(0.0, soh - dSOH)

    # Log row
    st.session_state.LOG.append({
        "t": i, "price": p_t, "power_mw": p_grid_mw, "soc_mwh": soc_new,
        "soh_pct": soh_new, "dsoh_pct": dSOH, "dsoh_cyc_pct": dSOH_cyc,
        "dsoh_cal_pct": dSOH_cal, "throughput_mwh": st.session_state.THRUPUT
    })

    # Commit
    st.session_state.SOC = soc_new
    st.session_state.SOH = soh_new
    st.session_state.IDX = i + 1
    return True

def append_to_charts(i, p_t, p_mw, soc_mwh, soh_pct):
    price_chart.add_rows(pd.DataFrame({"price": [p_t]}, index=[i]))
    power_chart.add_rows(pd.DataFrame({"power_mw": [p_mw]}, index=[i]))
    soc_chart.add_rows(pd.DataFrame({"soc_mwh": [soc_mwh]}, index=[i]))
    soh_chart.add_rows(pd.DataFrame({"soh_pct": [soh_pct]}, index=[i]))

def update_kpis():
    kpi1.metric("Step index", st.session_state.IDX)
    kpi2.metric("SOH (%)", f"{st.session_state.SOH:0.2f}")
    kpi3.metric("Total throughput (MWh)", f"{st.session_state.THRUPUT:0.2f}")
    kpi4.metric("Current SoC (MWh)", f"{st.session_state.SOC:0.2f}")

if run:
    steps_to_run = min(max_steps, n_points - st.session_state.IDX)
    for _ in range(steps_to_run):
        i = st.session_state.IDX
        if i >= n_points:
            break
        # Perform one step
        ok = sim_step(i)
        # Read the just-logged row for plotting
        row = st.session_state.LOG[-1]
        append_to_charts(
            i=row["t"],
            p_t=row["price"],
            p_mw=row["power_mw"],
            soc_mwh=row["soc_mwh"],
            soh_pct=row["soh_pct"],
        )
        update_kpis()
        time.sleep(refresh_ms/1000.0)
        if not ok:
            st.warning("End-of-life SOH reached — stopping animation.")
            break

# Show a download when we have data
log_df = pd.DataFrame(st.session_state.LOG)
if not log_df.empty:
    st.download_button(
        "Download results CSV",
        data=log_df.to_csv(index=False).encode(),
        file_name="bess_arbitrage_soh_results.csv",
        mime="text/csv",
    )

st.caption(
    "This version animates charts in-place using add_rows(). "
    "Dispatch uses a simple quantile policy; SOH blends calendar + throughput with a DoD severity proxy. "
    "Tune coefficients for your chemistry and replace with rainflow + OEM maps for bankable work."
)
