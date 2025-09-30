
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------- #
# App Config
# --------------------------- #
st.set_page_config(page_title="BESS SOH Curve Generator", layout="centered")

START_SOH = 100.0
END_SOH = 65.0
TOTAL_DROP = START_SOH - END_SOH  # 35%
MAX_DROP_PER_YEAR = 2.5
MIN_YEARS_REQUIRED = int(np.ceil(TOTAL_DROP / MAX_DROP_PER_YEAR))  # 14


# --------------------------- #
# Core Logic
# --------------------------- #
def generate_drops_constrained(years: int, seed: int | None = None) -> np.ndarray:
    """
    Create a random, near-linear set of yearly drops such that:
    - sum(drops) == TOTAL_DROP
    - 0 <= drops[i] < MAX_DROP_PER_YEAR for all i
    - 'Linear-like' with small random variations

    Strategy:
    1) Start with equal drop (TOTAL_DROP / years).
    2) Add zero-sum random noise, then clip and re-normalize under the cap.
    3) Iterate clipping/redistribution to guarantee the per-year cap and exact sum.

    Returns
    -------
    drops : np.ndarray of shape (years,)
    """
    if years < MIN_YEARS_REQUIRED:
        raise ValueError(
            f"Years too few ({years}) to reach {END_SOH}% while keeping "
            f"per-year drop < {MAX_DROP_PER_YEAR}% (need >= {MIN_YEARS_REQUIRED})."
        )

    rng = np.random.default_rng(seed if seed not in (None, 0) else None)

    # 1) Start near-linear
    base = TOTAL_DROP / years  # average drop per year (<= 2.5% when years >= 14)
    drops = np.full(years, base, dtype=float)

    # 2) Add zero-sum noise for randomness while keeping near-linear feel
    #    Scale noise so it won't easily violate the cap; we'll still enforce strictly below.
    noise = rng.normal(loc=0.0, scale=min(0.2, 0.5 * (MAX_DROP_PER_YEAR - base)), size=years)
    noise -= noise.mean()  # zero-sum
    drops = drops + noise

    # Ensure non-negative preliminary (we'll refine with clipping & redistribute)
    drops = np.clip(drops, 0.0, None)

    # 3) Iteratively enforce cap and exact sum
    #    We maintain sum(drops) == TOTAL_DROP by rebalancing headroom.
    for _ in range(1000):  # generous safety iterations
        # Clip to strictly below the cap (e.g., cap - tiny epsilon)
        eps = 1e-6
        over_mask = drops > (MAX_DROP_PER_YEAR - eps)
        excess = (drops[over_mask] - (MAX_DROP_PER_YEAR - eps)).sum()
        drops[over_mask] = MAX_DROP_PER_YEAR - eps  # enforce < 2.5%

        # If no excess, just normalize total to exactly TOTAL_DROP and exit
        if excess <= 1e-12:
            current_sum = drops.sum()
            diff = TOTAL_DROP - current_sum
            if abs(diff) <= 1e-9:
                break

            # Distribute tiny diff into entries with headroom
            headroom = (MAX_DROP_PER_YEAR - eps) - drops
            mask = headroom > 1e-12
            if not np.any(mask):
                # Fallback: spread uniformly (should be extremely rare)
                drops += diff / years
                continue

            alloc = np.zeros_like(drops)
            alloc[mask] = diff * (headroom[mask] / headroom[mask].sum())
            drops += alloc
            # Loop again to ensure no cap violation from numerical noise
            continue

        # We have excess due to clipping. Redistribute into headroom.
        headroom = (MAX_DROP_PER_YEAR - eps) - drops
        mask = headroom > 1e-12
        if not np.any(mask):
            # Should not happen when years >= 14; but just in case, slightly relax by uniform spread
            drops += excess / years
            continue

        alloc = np.zeros_like(drops)
        alloc[mask] = excess * (headroom[mask] / headroom[mask].sum())
        drops += alloc

    # Final tidy: exact sum and cap compliance
    drops = np.clip(drops, 0.0, MAX_DROP_PER_YEAR - 1e-6)
    correction = TOTAL_DROP - drops.sum()
    if abs(correction) > 1e-9:
        # Put correction where there is most headroom
        headroom = (MAX_DROP_PER_YEAR - 1e-6) - drops
        idx = int(np.argmax(headroom))
        drops[idx] = np.clip(drops[idx] + correction, 0.0, MAX_DROP_PER_YEAR - 1e-6)

    # Sanity checks
    assert drops.sum() == pytest_close(TOTAL_DROP), "Sum constraint not met."
    assert np.all(drops < MAX_DROP_PER_YEAR), "Per-year cap violated."
    assert np.all(drops >= 0.0), "Negative drop encountered."
    return drops


def pytest_close(val: float, target: float = None, tol: float = 1e-6) -> float:
    """
    Helper for 'exact enough' equality to avoid float drift.
    Returns the target for clean equality checks in assertions.
    """
    if target is None:
        target = val
    if abs(val - target) <= tol:
        return target
    return val  # assertion will fail if not within tolerance


def build_soh_series(drops: np.ndarray) -> pd.DataFrame:
    """Build the SOH trajectory from yearly drops, starting at 100% down to 65%."""
    soh = [START_SOH]
    for d in drops:
        soh.append(soh[-1] - d)
    years_idx = np.arange(0, len(drops) + 1)  # Year 0..N
    df = pd.DataFrame({"Year": years_idx, "SOH_%": np.round(soh, 4)})
    return df


# --------------------------- #
# UI
# --------------------------- #
st.title("Random BESS SOH Curve Generator")
st.caption(
    f"Generates a near-linear SOH curve from **{START_SOH:.0f}%** to **{END_SOH:.0f}%** "
    f"with each annual drop **< {MAX_DROP_PER_YEAR:.1f}%**."
)

with st.sidebar:
    st.header("Parameters")
    years = st.number_input("Number of years", min_value=1, max_value=100, value=20, step=1)
    seed = st.number_input("Random seed (optional)", min_value=0, max_value=10_000, value=0, step=1)
    st.markdown(
        f"- Constraint: need **≥ {MIN_YEARS_REQUIRED} years** "
        f"to reach {END_SOH:.0f}% with < {MAX_DROP_PER_YEAR:.1f}% annual drop."
    )

gen = st.button("Generate SOH Curve")

if gen:
    if years < MIN_YEARS_REQUIRED:
        st.error(
            f"Cannot satisfy both constraints with {years} years. "
            f"Please use at least {MIN_YEARS_REQUIRED} years."
        )
    else:
        try:
            drops = generate_drops_constrained(years=int(years), seed=int(seed))
            df = build_soh_series(drops)

            st.success(
                f"Generated {int(years)}-year SOH curve from {START_SOH:.0f}% to "
                f"{END_SOH:.0f}% with annual drops < {MAX_DROP_PER_YEAR:.1f}%."
            )

            st.line_chart(df.set_index("Year")["SOH_%"])
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"soh_curve_{int(years)}y_seed{int(seed)}.csv",
                mime="text/csv",
            )

            with st.expander("Details"):
                annual = pd.DataFrame(
                    {
                        "Year": np.arange(1, int(years) + 1),
                        "Annual_Drop_%": np.round(drops, 4),
                    }
                )
                st.markdown("**Annual degradation (%/year)**")
                st.dataframe(annual, use_container_width=True)

        except Exception as e:
            st.error(f"Generation failed: {e}")

with st.expander("Notes"):
    st.markdown(
        "- Curve is **monotonic decreasing**, ending exactly at **65%** (± tiny float tolerance).\n"
        "- Each year’s drop is **strictly less than 2.5%**.\n"
        "- Use the **random seed** to reproduce a specific curve.\n"
        "- The distribution is near-linear with small random variation."
    )
