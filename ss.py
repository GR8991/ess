# bess_arbitrage_soh.py
# Streamlit sandbox to explore BESS energy arbitrage vs. C-rate,
# with live step-through dispatch, SoC, SOH, and degradation depth.

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Rerun compatibility (Streamlit 1.28+)
# ----------------------------
def _rerun():
    """Use st.rerun() when available; fall back to experimental API otherwise."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()  # pragma: no cover

# ----------------------------
# Utilities
# ----------------------------
def make_synthetic_prices(n=288, market="CAISO", seed=7):
    """
    Generate 5-minute prices for n steps.
    CAISO-like: midday dip + evening ramp.
    ERCOT-like: base curve + random spikes.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)

    base = 35 + 5*np.sin(2*np.pi*(t/n))  # mild diurnal

    if market == "CAISO":
        midday = -18*np.exp(-0.5*((t - 0.55*n)/(0.10*n))**2)
        evening = 30*np.exp(-0.5*((t - 0.80*n)/(0.06*n))**2)
        noise = rng.normal(0, 1.5, n)
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
    # Ensure cheap < rich
    if cheap >= rich:
        eps = 1e-3
        cheap, rich = rich - eps, rich + eps
    return cheap, rich

def limit_by_c_rate(power_req_mw, e_mwh, c_rate):
    """Limit absolute power to C-rate * energy capacity (MW)."""
    max_abs = float(c_rate) * float(e_mwh)
    return float(np.clip(power_req_mw, -max_abs, max_abs))

def step_dispatch(
    p_t, soc, e_mwh, pmax_mw, eta_c, eta_d,
    cheap_thr, rich_thr, reserve_frac=0.0,
    c_rate=None, dt_h=5/60,
):
    """
    Simple price-taking arbitrage:
      - charge if price <= cheap_thr and SoC < 100%
      - discharge if price >= rich_thr and SoC > 0
      - else idle
    Respects SoC, efficiency, reserve carve-out, and optional C-rate limit.
    Returns (p_grid_mw, new_soc)
    """
    e_free = e_mwh - soc
    e_stored = soc

    usable_pmax_dis = pmax_mw * (1 - reserve_frac)
    usable_pmax_chg = pmax_mw

    # Decide raw power request (MW, + discharging to grid)
    if p_t <= cheap_thr and soc < e_mwh:
        e_in_possible = e_free / eta_c
        p_req = -min(usable_pmax_chg, e_in_possible / dt_h)  # negative = charging
    elif p_t >= rich_thr and soc > 0:
        e_out_possible = e_stored * eta_d
        p_req = +min(usable_pmax_dis, e_out_possible / dt_h)
    else:
        p_req = 0.0

    # Apply C-rate cap
    if c_rate is not None and c_rate > 0:
        p_req = limit_by_c_rate(p_req, e_mwh, c_rate)

    # Enforce SoC bounds after efficiencies
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

# ----------------------------
# Degradation Models (transparent, tweakable)
# ----------------------------
def dod_factor(dod_pu):
    """
    Depth-of-Discharge severity factor (heuristic).
    0..1 input; ~1.0 around 80% DoD; softer below, harsher above.
    """
    dod_pu = float(np.clip(dod_pu, 0, 1))
    return 0.5 + 0.8 * (dod_pu ** 1.2)

def cycle_deg(delta_e_throughput_mwh, e_mwh, k_cyc):
    """
    Throughput-based damage:
      dSOH (%) = k_cyc * (Δthroughput / E_cap) ; DoD severity applied externally.
    """
    if e_mwh <= 1e-9:
        return 0.0
    return 100.0 * float(k_cyc) * (float(delta_e_throughput_mwh) / float(e_mwh))

def calendar_deg(dt_hours, k_cal_per_year, temp_c):
    """
    Calendar fade (% of SOH) over dt_hours, scaled by temperature (Q10).
    k_cal_per_year ~= %/year at 25°C.
    """
    q10 = 2.0
    scale = q10 ** ((float(temp_c) - 25.0)/10.0)
    return float(k_cal_per_year) * scale * (float(dt_hours) / (24*365))

class DoDTracker:
    """
    Very simple SoC-motion proxy to approximate 'cycle severity' per step.
    Tracks absolute SoC movement (per unit) to derive a local DoD-like signal.
    """
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
# Streamlit App
# ----------------------------
st.set_page_config(page_title="BESS Arbitrage + SOH Sandbox", layout="wide")
st.title("BESS Arbitrage + SOH ‘Real-Time’ Sandbox")
st.caption(
    "Configure a battery and price curve, then **Run** to watch dispatch, SoC, SOH, "
    "and degradation evolve step by step."
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Asset & Market")
    e_mwh = st.number_input("Energy capacity (MWh)", 10.0, 5000.0, 400.0, 10.0)
    duration_h = st.number_input("Duration at Pmax (h)", 0.25, 12.0, 4.0, 0.25)
    default_pmax = max(1.0, e_mwh / duration_h)
    pmax_mw = st.number_input("Max power (MW)", 1.0, 5000.0, default_pmax, 1.0)
    # Allow user to explicitly cap C-rate (defaults to pmax/e_mwh)
    c_rate = st.number_input(
        "C-rate limit (1/h)",
        0.05, 5.0,
        round(pmax_mw / max(e_mwh, 1e-9), 3),
        0.05,
        help="Power cap relative to energy capacity. Max |P| = C * E."
    )
    eta_rt = st.slider("Round-trip efficiency (%)", 50, 98, 90)
    eta_c = float(np.sqrt(eta_rt/100.0))
    eta_d = float(np.sqrt(eta_rt/100.0))
    reserve_frac = st.slider("Reserve carve-out for ancillary (%)", 0, 50, 0) / 100.0

    st.divider()
    st.subheader("Prices")
    mode = st.radio("Price source", ["Synthetic (CAISO-like)", "Synthetic (ERCOT-like)", "Upload CSV"], index=0)
    seed = st.number_input("Random seed (synthetic)", 0, 10_000, 7, 1)
    n_points = st.select_slider("Horizon (5-min steps)", [144, 288, 576], value=288)
    cheap_q = st.slider("Cheap quantile", 0.05, 0.5, 0.35, 0.05)
    rich_q  = st.slider("Rich quantile", 0.5, 0.95, 0.70, 0.05)

    if mode.startswith("Synthetic"):
        market_kind = "CAISO" if "CAISO" in mode else "ERCOT"
        price = make_synthetic_prices(n_points, market=market_kind, seed=seed)
    else:
        up = st.file_uploader("Upload CSV with columns: timestamp,price", type=["csv"])
        if up is not None:
            dfp = pd.read_csv(up)
            if "price" not in dfp.columns:
                st.error("CSV must contain a 'price' column.")
                st.stop()
            price = dfp["price"].astype(float).to_numpy()
            n_points = len(price)
        else:
            price = make_synthetic_prices(n_points, market="CAISO", seed=seed)

    cheap_thr, rich_thr = quantile_policy_thresholds(price, cheap_q, rich_q)

    st.divider()
    st.subheader("Degradation Model")
    temp_c = st.slider("Cell temperature (°C)", 10, 45, 27)
    k_cal_per_year = st.number_input("Calendar fade @25°C (%/yr)", 0.0, 10.0, 1.2, 0.1)
    k_cyc = st.number_input(
        "Cycle fade coefficient (–)",
        0.0, 1.0, 0.08, 0.01,
        help="Percent SOH per full-cycle equivalent throughput (severity via DoD proxy)."
    )
    dt_min = st.select_slider("Dispatch step (minutes)", [1, 5, 10, 15], value=5)
    dt_h = dt_min / 60.0

    st.divider()
    st.subheader("Run Controls")
    colA, colB, colC = st.columns(3)
    if colA.button("Run / Resume"):
        st.session_state["RUN"] = True
    if colB.button("Pause"):
        st.session_state["RUN"] = False
    if colC.button("Reset"):
        for k in ("RUN","IDX","SOC","SOH","THRUPUT","LOG","DOD"):
            st.session_state.pop(k, None)

# ---------- Session state init ----------
def init_if_needed():
    st.session_state.setdefault("RUN", False)
    st.session_state.setdefault("IDX", 0)
    st.session_state.setdefault("SOC", 0.5 * e_mwh)  # start half full
    st.session_state.setdefault("SOH", 100.0)
    st.session_state.setdefault("THRUPUT", 0.0)
    st.session_state.setdefault("LOG", [])
    if "DOD" not in st.session_state:
        st.session_state["DOD"] = DoDTracker()

init_if_needed()

# ---------- One step ----------
def sim_step(i: int) -> bool:
    """Advance the simulation by one time-step; returns False if EoL reached."""
    soh = float(st.session_state.SOH)
    if soh <= 70.0:  # example EoL criterion
        return False

    # Effective energy shrinks with SOH
    e_eff = float(e_mwh) * (soh / 100.0)
    soc = float(np.clip(st.session_state.SOC, 0.0, e_eff))

    p_t = float(price[i])
    p_grid_mw, soc_new = step_dispatch(
        p_t=p_t, soc=soc, e_mwh=e_eff, pmax_mw=float(pmax_mw),
        eta_c=eta_c, eta_d=eta_d,
        cheap_thr=cheap_thr, rich_thr=rich_thr,
        reserve_frac=float(reserve_frac), c_rate=float(c_rate), dt_h=float(dt_h)
    )

    # Energy moved this step
    delta_throughput = abs(p_grid_mw) * dt_h
    st.session_state.THRUPUT += delta_throughput

    # DoD severity proxy
    dod_delta = st.session_state.DOD.update(soc_new, e_eff)  # per-unit SoC movement
    severity = dod_factor(dod_delta)

    dSOH_cyc = cycle_deg(delta_throughput, e_eff, k_cyc) * severity
    dSOH_cal = calendar_deg(dt_h, k_cal_per_year, temp_c)
    dSOH     = dSOH_cyc + dSOH_cal
    soh_new  = max(0.0, soh - dSOH)

    # Log the row
    st.session_state.LOG.append({
        "t": i,
        "price": p_t,
        "power_mw": p_grid_mw,
        "soc_mwh": soc_new,
        "soh_pct": soh_new,
        "dsoh_pct": dSOH,
        "dsoh_cyc_pct": dSOH_cyc,
        "dsoh_cal_pct": dSOH_cal,
        "throughput_mwh": st.session_state.THRUPUT,
        "cheap_thr": cheap_thr,
        "rich_thr": rich_thr,
    })

    # Commit state
    st.session_state.SOC = soc_new
    st.session_state.SOH = soh_new
    st.session_state.IDX = i + 1
    return True

# ---------- Auto-advance when running ----------
if st.session_state.RUN and st.session_state.IDX < n_points:
    still_ok = sim_step(st.session_state.IDX)
    if still_ok and st.session_state.IDX < n_points:
        _rerun()

# ---------- Layout ----------
log = pd.DataFrame(st.session_state.LOG)
left, right = st.columns([2, 1])

with right:
    st.metric("Step index", st.session_state.IDX)
    st.metric("SOH (%)", f"{st.session_state.SOH:0.2f}")
    st.metric("Total throughput (MWh)", f"{st.session_state.THRUPUT:0.2f}")
    st.metric("Current SoC (MWh)", f"{st.session_state.SOC:0.2f}")
    st.metric("Cheap / Rich (USD/MWh)", f"{cheap_thr:0.2f} / {rich_thr:0.2f}")

    if not log.empty:
        st.download_button(
            "Download results CSV",
            data=log.to_csv(index=False).encode(),
            file_name="bess_arbitrage_soh_results.csv",
            mime="text/csv",
        )

with left:
    st.subheader("Live Charts")
    if not log.empty:
        st.line_chart(log.set_index("t")["price"], height=180)
        st.line_chart(log.set_index("t")["power_mw"], height=180)
        st.line_chart(log.set_index("t")["soc_mwh"], height=180)
        st.line_chart(log.set_index("t")["soh_pct"], height=180)
    else:
        st.info("Press **Run / Resume** to start stepping. Upload a CSV if you prefer real market data.")

# ---------- Footer ----------
st.caption(
    "Dispatch uses a simple quantile policy under SoC, efficiency, reserve, and C-rate constraints. "
    "SOH blends calendar and throughput-based cycle fade with a DoD severity proxy. "
    "For bankable studies, replace with calibrated rainflow + OEM aging maps."
)
