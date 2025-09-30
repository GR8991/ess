
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Random BESS SOH Curve", layout="centered")

st.title("Random BESS SOH Curve Generator")
st.caption("Generates a linear-like SOH curve from 100% to 65% with < 2.5% annual drop.")

# ---- Parameters ----
START_SOH = 100.0
END_SOH = 65.0
TOTAL_DROP = START_SOH - END_SOH  # 35%
MAX_DROP_PER_YEAR = 2.5
MIN_YEARS_REQUIRED = int(np.ceil(TOTAL_DROP / MAX_DROP_PER_YEAR))  # 14 years

col1, col2 = st.columns(2)
with col1:
    years = st.number_input("Number of years", min_value=1, max_value=100, value=20, step=1)
with col2:
    seed = st.number_input("Random seed (optional)", min_value=0, max_value=10_000, value=0, step=1)

st.write(f"**Constraint**: To drop from {START_SOH:.0f}% to {END_SOH:.0f}% with < {MAX_DROP_PER_YEAR:.1f}% per year, "
         f"you need at least **{MIN_YEARS_REQUIRED} years**.")

def generate_soh(years:int, seed:int|None=None):
    """
    Generate yearly SOH drops that:
    - Sum exactly to TOTAL_DROP (35%)
    - Each year drop <= MAX_DROP_PER_YEAR (2.5%)
    - Non-negative (monotonic decreasing SOH)
    Uses a Dirichlet-based randomization + redistribution to respect per-year cap.
    """
    if years < MIN_YEARS_REQUIRED:
        return None, "Years too few to reach 65% while keeping per-year drop < 2.5%."

    rng = np.random.default_rng(seed if seed else None)

    # Start with a random partition of TOTAL_DROP
    drops = rng.dirichlet(np.ones(years)) * TOTAL_DROP

    # If any drop exceeds the cap, iteratively clip & redistribute the excess
    # There is guaranteed feasible solution because average drop <= cap.
    for _ in range(1000):  # safety iterations
        over = drops - MAX_DROP_PER_YEAR
        over[over < 0] = 0.0
        excess = over.sum()
        if excess <= 1e-9:
            break

        # Clip to cap
        drops = np.minimum(drops, MAX_DROP_PER_YEAR)

        # Distribute excess to entries with headroom
        headroom = MAX_DROP_PER_YEAR - drops
        mask = headroom > 1e-12
        if not np.any(mask):
            # Shouldn't happen when years >= MIN_YEARS_REQUIRED, but guard anyway
            # Even distribution to all (will be very small oscillation)
            drops += excess / years
            continue

        # Allocate proportional to headroom
        alloc = np.zeros_like(drops)
        alloc[mask] = excess * (headroom[mask] / headroom[mask].sum())
        drops += alloc

    # Numeric tidy-up to ensure exact sum
    correction = TOTAL_DROP - drops.sum()
    # Push tiny correction to the largest headroom slot (or last)
    idx = int(np.argmax(MAX_DROP_PER_YEAR - drops))
    drops[idx] = np.clip(drops[idx] + correction, 0.0, MAX_DROP_PER_YEAR)

    # Build SOH trajectory
    soh = [START_SOH]
    for d in drops:
        soh.append(soh[-1] - d)
    soh = np.array(soh)

    # Final sanity checks
    if np.any(drops < -1e-9) or np.any(drops - MAX_DROP_PER_YEAR > 1e-6):
        return None, "Failed to enforce per-year cap; try a different seed."
    if abs(soh[-1] - END_SOH) > 1e-6:
        return None, "Failed to hit final 65%; try a different seed."

    years_idx = np.arange(0, years + 1)  # Year 0 to Year N
    df = pd.DataFrame({
        "Year": years_idx,
        "SOH_%": np.round(soh, 4)
    })
    return df, None

# ---- Generate button ----
if st.button("Generate SOH Curve"):
    if years < MIN_YEARS_REQUIRED:
        st.error(f"Cannot satisfy both constraints with {years} years. "
                 f"Please use at least {MIN_YEARS_REQUIRED} years.")
    else:
        df, err = generate_soh(years, seed)
        if err:
            st.error(err)
        else:
            st.success(f"Generated {years}-year SOH curve from {START_SOH:.0f}% to {END_SOH:.0f}% "
                       f"with each annual drop < {MAX_DROP_PER_YEAR}%.")

            st.line_chart(df.set_index("Year")["SOH_%"])
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"soh_curve_{years}y_seed{seed}.csv",
                mime="text/csv"
            )

# Helpful notes
with st.expander("Notes & Tips"):
    st.markdown(
        "- The curve is **monotonic decreasing** and ends exactly at **65%**.\n"
        "- Each year’s drop is **< 2.5%**, so you need **≥ 14 years** to reach 65%.\n"
        "- Use the **random seed** to reproduce a particular curve.\n"
        "- The pattern is near-linear with small random variation year-to-year."
    )
