import io
import time
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Utilities
# ----------------------------
def make_synthetic_prices(n=288, market="CAISO"):
    """Generate 5-min prices for 24h (288 points).
    CAISO-like: midday dip + evening ramp. ERCOT-like: more spikes."""
    t = np.arange(n)
    base = 35 + 5*np.sin(2*np.pi*(t/(n)))  # gentle diurnal
    if market == "CAISO":
        midday = -18*np.exp(-0.5*((t - 0.55*n)/(0.10*n))**2)
        evening = 30*np.exp(-0.5*((t - 0.80*n)/(0.06*n))**2)
        noise = np.random.normal(0, 1.5, n)
        p = base + midday + evening + noise
    else:  # ERCOT-like
        spikes = np.zeros(n)
        # add a few random spikes
        for _ in range(6):
            ctr = np.random.randint(int(0.2*n), int(0.95*n))
            amp = np.random.uniform(50, 150)
            wid = np.random.uniform(0.01*n, 0.03*n)
            spikes += amp*np.exp(-0.5*((t-ctr)/wid)**2)
        noise = np.random.normal(0, 2.5, n)
        p = base + spikes + noise
    return np.clip(p, 0, None)

def quantile_policy_thresholds(price, cheap_q=0.35, rich_q=0.70):
    return (np.quantile(price, cheap_q), np.quantile(price, rich_q))

def limit_by_c_rate(power_request_mw, e_mwh, c_rate):
    """Limit absolute power to C-rate * energy capacity."""
    max_abs_mw = c_rate * e_mwh
    return np.clip(power_request_mw, -max_abs_mw, max_abs_mw)

def step_dispatch(p_t, soc, e_mwh, pmax_mw, eta_c, eta_d,
                  cheap_thr, rich_thr, reserve_frac=0.0,
                  c_rate=None, dt_h=5/60):
    """
    Simple price-taking arbitrage:
    - charge if p <= cheap_thr and SoC < 100%
    - discharge if p >= rich_thr and SoC > 0
    - else idle
    Respects SoC, efficiency, reserve carve-out, and optional C-rate limit.
    Returns (p_grid_mw, new_soc)
    """
    # Available headroom/footroom
    e_free = e_mwh - soc
    e_stored = soc

    # Reserve reduces usable discharge headroom
    usable_pmax_dis = pmax_mw * (1 - reserve_frac)
    usable_pmax_chg = pmax_mw

    # Decide raw request
    if p_t <= cheap_thr and soc < e_mwh:
        # charge
        e_in_possible = e_free / eta_c
        p_req = -min(usable_pmax_chg, e_in_possible/dt_h)  # negative = charging
    elif p_t >= rich_thr and soc > 0:
        # discharge
        e_out_possible = e_stored * eta_d
        p_req = +min(usable_pmax_dis, e_out_possible/dt_h)
    else:
        p_req = 0.0

    # Apply C-rate cap
    if c_rate is not None:
        p_req = limit_by_c_rate(p_req, e_mwh, c_rate)

    # Bound by SoC again with efficiencies and dt
    if p_req < 0:  # charging
        e_from_grid = -p_req * dt_h  # MWh
        e_to_store = e_from_grid * eta_c
        e_to_store = min(e_to_store, e_free)
        p_final = -(e_to_store/eta_c) / dt_h
        soc_new = soc + e_to_store
    elif p_req > 0:  # discharging
        e_to_grid = p_req * dt_h
        e_from_store = e_to_grid/eta_d
        e_from_store = min(e_from_store, e_stored)
        p_final = (e_from_store*eta_d) / dt_h
        soc_new = soc - e_from_store
    else:
        p_final = 0.0
        soc_new = soc

    return float(p_final), float(soc_new)

# ----------------------------
# Degradation Models (transparent, tweakable)
# ----------------------------
def dod_factor(dod):
    """Depth-of-Discharge severity factor (heuristic).
    1.0 at 80% DoD, milder below, harsher above."""
    # 0..1 input; returns multiplier >= ~0.5
    return 0.5 + 0.8*(dod**1.2)

def cycle_deg(delta_e_throughput_mwh, e_mwh, k_cyc):
    """
    Throughput-based damage: dSOH (%) = k_cyc * (equivalent throughput / E_cap) * DoD_severity
    Here we pass in delta throughput; DoD severity added in aggregator using DOD windows.
    """
    return 100.0 * k_cyc * (delta_e_throughput_mwh / max(e_mwh, 1e-9))

def calendar_deg(dt_hours, k_cal_per_year, temp_c):
    """
    Calendar fade: dSOH (%) over dt, scaled by temperature.
    k_cal_per_year ~ %/year at 25C; Arrhenius-like scaling per 10C delta.
    """
    q10 = 2.0  # simple Q10 factor
    scale = q10 ** ((temp_c - 25.0)/10.0)
    return k_cal_per_year * scale * (dt_hours / (24*365))

class DoDWindow:
    """Tracks partial cycles to estimate DoD-based severity per interval (simple proxy)."""
    def __init__(self):
        self.last_soc = None
        self.acc = 0.0  # accumulated absolute SoC movement in pu [0..1]

    def update(self, soc, e_mwh):
        soc_pu = soc / max(e_mwh, 1e-9)
        if self.last_soc is None:
            self.last_soc = soc_pu
            return 0.0
        delta = abs(soc_pu - self.last_soc)
        self.acc += delta
        self.last_soc = soc_pu
        # "Equivalent DoD" this step is small; severity applied externally
        return delta

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="BESS Arbitrage + SOH Sandbox", layout="wide")

st.title("BESS Arbitrage + SOH ‘Real-Time’ Sandbox")
st.caption("Configure a battery + prices, then Run to watch dispatch, SoC, SOH, and degradation evolve in near-real-time.")

with st.sidebar:
    st.header("Asset & Market")
    e_mwh = st.number_input("Energy capacity (MWh)", 50.0, 2000.0, 400.0, 10.0)
    duration_h = st.number_input("Duration at Pmax (h)", 0.25, 8.0, 4.0, 0.25)
    pmax_mw = st.number_input("Max power (MW)", 1.0, 2000.0, max(10.0, e_mwh/duration_h), 5.0)
    c_rate = st.number_input("C-rate limit (per-hour)", 0.1, 3.0, round(pmax_mw/e_mwh, 3), 0.05)
    eta_rt = st.slider("Round-trip efficiency (%)", 50, 98, 90)
    eta_c = np.sqrt(eta_rt/100.0)
    eta_d = np.sqrt(eta_rt/100.0)
    reserve_frac = st.slider("Reserve carve-out for ancillary (%)", 0, 50, 0)/100

    st.divider()
    st.subheader("Prices")
    mode = st.radio("Price source", ["Synthetic (CAISO-like)", "Synthetic (ERCOT-like)", "Upload CSV"], index=0)
    if mode.startswith("Synthetic"):
        n_points = st.select_slider("Horizon length", options=[144, 288, 576], value=288,
                                    help="5-min steps: 144=12h, 288=24h, 576=48h")
        price = make_synthetic_prices(n_points, "CAISO" if "CAISO" in mode else "ERCOT")
        cheap_q = st.slider("Cheap quantile", 0.05, 0.5, 0.35, 0.05)
        rich_q = st.slider("Rich quantile", 0.5, 0.95, 0.70, 0.05)
    else:
        up = st.file_uploader("Upload price CSV with columns: timestamp,price", type=["csv"])
        cheap_q = st.slider("Cheap quantile", 0.05, 0.5, 0.35, 0.05)
        rich_q = st.slider("Rich quantile", 0.5, 0.95, 0.70, 0.05)
        if up is not None:
            dfp = pd.read_csv(up)
            assert "price" in dfp.columns, "CSV must contain a 'price' column"
            price = dfp["price"].to_numpy()
            n_points = len(price)
        else:
            price = make_synthetic_prices(288, "CAISO")
            n_points = len(price)

    cheap_thr, rich_thr = quantile_policy_thresholds(price, cheap_q, rich_q)

    st.divider()
    st.subheader("Degradation Model")
    temp_c = st.slider("Average cell temperature (°C)", 10, 45, 27)
    k_cal_per_year = st.number_input("Calendar fade @25°C (%/yr)", 0.0, 10.0, 1.2, 0.1)
    k_cyc = st.number_input("Cycle fade coefficient (–)", 0.0, 1.0, 0.08, 0.01,
                            help="Percent SOH per full-cycle equivalent of throughput (severity scaled by DoD).")
    dt_min = st.select_slider("Dispatch step (minutes)", options=[1, 5, 10, 15], value=5)
    dt_h = dt_min/60.0

    st.divider()
    st.subheader("Run Controls")
    colA, colB = st.columns(2)
    if colA.button("Run / Resume"):
        st.session_state["RUN"] = True
    if colB.button("Pause"):
        st.session_state["RUN"] = False
    if st.button("Reset"):
        for k in ["RUN", "IDX", "SOC", "SOH", "THRUPUT", "LOG"]:
            if k in st.session_state: del st.session_state[k]

# ----------------------------
# Initialize state
# ----------------------------
def init_if_needed():
    if "IDX" not in st.session_state:
        st.session_state["IDX"] = 0
    if "SOC" not in st.session_state:
        st.session_state["SOC"] = 0.5*e_mwh  # start half full
    if "SOH" not in st.session_state:
        st.session_state["SOH"] = 100.0
    if "THRUPUT" not in st.session_state:
        st.session_state["THRUPUT"] = 0.0
    if "LOG" not in st.session_state:
        st.session_state["LOG"] = []  # list of dict rows
    if "RUN" not in st.session_state:
        st.session_state["RUN"] = False

init_if_needed()

# ----------------------------
# One simulation step
# ----------------------------
dod_tracker = DoDWindow()

def sim_step(i):
    soh = st.session_state.SOH
    if soh <= 70.0:
        # Example EoL: 70% capacity
        return False

    # Effective energy capacity shrinks with SOH
    e_eff = e_mwh * (soh/100.0)
    soc = np.clip(st.session_state.SOC, 0.0, e_eff)

    p_t = float(price[i])
    p_grid_mw, soc_new = step_dispatch(
        p_t=p_t, soc=soc, e_mwh=e_eff, pmax_mw=pmax_mw, eta_c=eta_c, eta_d=eta_d,
        cheap_thr=cheap_thr, rich_thr=rich_thr, reserve_frac=reserve_frac,
        c_rate=c_rate, dt_h=dt_h
    )

    # Throughput (absolute energy moved)
    delta_throughput = abs(p_grid_mw) * dt_h
    st.session_state.THRUPUT += delta_throughput

    # DoD severity proxy from SoC motion
    dod_delta = dod_tracker.update(soc_new, e_eff)
    sev = dod_factor(min(1.0, dod_delta))

    dSOH_cyc = cycle_deg(delta_throughput, e_eff, k_cyc) * sev
    dSOH_cal = calendar_deg(dt_h, k_cal_per_year, temp_c)
    dSOH = dSOH_cyc + dSOH_cal
    soh_new = max(0.0, soh - dSOH)

    # Log row
    row = dict(
        t=i, price=p_t, power_mw=p_grid_mw, soc_mwh=soc_new, soh_pct=soh_new,
        dsoh_pct=dSOH, dsoh_cyc_pct=dSOH_cyc, dsoh_cal_pct=dSOH_cal,
        throughput_mwh=st.session_state.THRUPUT,
        cheap_thr=cheap_thr, rich_thr=rich_thr
    )
    st.session_state.LOG.append(row)

    # Commit state
    st.session_state.SOC = soc_new
    st.session_state.SOH = soh_new
    st.session_state.IDX = i + 1
    return True

# ----------------------------
# Real-time runner
# ----------------------------
if st.session_state.RUN and st.session_state.IDX < n_points:
    sim_step(st.session_state.IDX)
    # auto-refresh to animate
    st.experimental_rerun()

# ----------------------------
# Plots & KPIs
# ----------------------------
log = pd.DataFrame(st.session_state.LOG)
left, right = st.columns([2, 1])

with right:
    st.metric("Step index", st.session_state.IDX, help="Advances each refresh while running.")
    st.metric("SOH (%)", f"{st.session_state.SOH:0.2f}")
    st.metric("Total throughput (MWh)", f"{st.session_state.THRUPUT:0.2f}")
    st.metric("Current SoC (MWh)", f"{st.session_state.SOC:0.2f}")
    st.metric("Cheap / Rich (USD/MWh)", f"{cheap_thr:0.2f} / {rich_thr:0.2f}")

    if not log.empty and st.download_button(
        "Download results CSV",
        data=log.to_csv(index=False).encode(),
        file_name="bess_arbitrage_soh_results.csv",
        mime="text/csv",
    ):
        pass

with left:
    st.subheader("Live Charts")
    if not log.empty:
        # Price
        st.line_chart(log.set_index("t")["price"], height=180)
        # Power (positive=discharge, negative=charge)
        st.line_chart(log.set_index("t")["power_mw"], height=180)
        # SoC
        st.line_chart(log.set_index("t")["soc_mwh"], height=180)
        # SOH (%)
        st.line_chart(log.set_index("t")["soh_pct"], height=180)
    else:
        st.info("Press **Run / Resume** to start stepping. Upload a CSV if you prefer real market data.")

st.caption(
    "Notes: Dispatch is a simple quantile policy under SoC, efficiency, reserve, and C-rate constraints. "
    "SOH blends calendar and throughput-based cycle fade with a DoD severity proxy; tune coefficients for your chemistry."
)
