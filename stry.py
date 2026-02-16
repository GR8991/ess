import math
import streamlit as st

# ==================================================
# EMAIL-ONLY LOGIN AUTH
# ==================================================

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_email = None

    if st.session_state.authenticated:
        return

    st.title("🔒 Access Restricted")
    st.caption("Please enter your authorized email ID")

    email = st.text_input("Email ID")

    # ---- ALLOWED EMAIL IDS (EDIT THIS LIST) ----
    ALLOWED_EMAILS = [
        "gangaraju.p@sunstripe.com",
        "s.kt@sunstripe.com",
        "deva@gmail.com",
        "kartik.v@sunstripe.",
    ]

    if st.button("Sign In"):
        if email.lower().strip() in ALLOWED_EMAILS:
            st.session_state.authenticated = True
            st.session_state.user_email = email.lower().strip()
            st.success("Access granted")
            st.rerun()
        else:
            st.error("Unauthorized email ID")

    st.stop()


# ==================================================
# CORE SIZING ENGINE (INSTALLED MWH ONLY)
# ==================================================

def size_bess_installed_mwh(
    module_kwh,
    modules_per_rack,
    inverter_mw,
    project_mw,
    installed_mwh_target,
):
    # Energy per rack (MWh)
    rack_mwh = (module_kwh * modules_per_rack) / 1000.0

    # Inverters required (gross, nameplate-based)
    n_inverters = math.ceil(project_mw / inverter_mw)

    # Continuous rack requirement
    total_racks_required = installed_mwh_target / rack_mwh

    # Integer rack options
    racks_per_inv_floor = max(1, math.floor(total_racks_required / n_inverters))
    racks_per_inv_ceil = math.ceil(total_racks_required / n_inverters)

    def evaluate(racks_per_inv):
        total_racks = racks_per_inv * n_inverters
        installed_mwh_actual = total_racks * rack_mwh
        energy_per_inv = racks_per_inv * rack_mwh

        return {
            "racks_per_inverter": racks_per_inv,
            "total_racks": total_racks,
            "installed_mwh": round(installed_mwh_actual, 2),
            "dc_ac_duration_hr": round(installed_mwh_actual / project_mw, 2),
            "system_c_rate": round(project_mw / installed_mwh_actual, 2),
            "block_c_rate": round(inverter_mw / energy_per_inv, 2),
        }

    return {
        "rack_mwh": round(rack_mwh, 4),
        "number_of_inverters": n_inverters,
        "lean_option": evaluate(racks_per_inv_floor),
        "conservative_option": evaluate(racks_per_inv_ceil),
    }


# ==================================================
# STREAMLIT APP
# ==================================================

st.set_page_config(
    page_title="BESS Installed-MWh Hardware Sizing",
    layout="centered"
)

# ---- LOGIN GATE ----
check_login()

# ---- MAIN APP ----
st.title("BESS Installed-MWh Hardware Sizing Tool")
st.caption(
    f"Logged in as: {st.session_state.user_email}"
)

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

with st.sidebar:
    st.header("Project Definition")

    project_mw = st.number_input(
        "Project Power (MW)",
        min_value=1.0,
        max_value=2000.0,
        value=150.0,
        step=1.0
    )

    installed_mwh_target = st.number_input(
        "Installed Energy (MWh) — FIXED BY USER",
        min_value=1.0,
        max_value=20000.0,
        value=447.9,
        step=1.0
    )

    inverter_mw = st.number_input(
        "Inverter Rating (MW)",
        min_value=0.5,
        max_value=10.0,
        value=2.5,
        step=0.1
    )

    st.divider()
    st.header("Battery Building Blocks")

    module_kwh = st.number_input(
        "Module Energy (kWh)",
        min_value=10.0,
        max_value=1000.0,
        value=117.0,
        step=1.0
    )

    modules_per_rack = st.number_input(
        "Modules per Rack",
        min_value=1,
        max_value=50,
        value=3,
        step=1
    )

    st.divider()

    if st.button("Sign Out"):
        st.session_state.authenticated = False
        st.session_state.user_email = None
        st.rerun()

# --------------------------------------------------
# RUN SIZING
# --------------------------------------------------

st.divider()

if st.button("Calculate Hardware Sizing"):
    result = size_bess_installed_mwh(
        module_kwh,
        modules_per_rack,
        inverter_mw,
        project_mw,
        installed_mwh_target,
    )

    st.subheader("Derived Hardware Configuration")

    st.write(f"**Energy per Rack:** {result['rack_mwh']} MWh")
    st.write(f"**Number of Inverters:** {result['number_of_inverters']}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Lean Option (Below / Closest)")
        for k, v in result["lean_option"].items():
            st.write(f"{k.replace('_',' ').title()}: **{v}**")

    with col2:
        st.markdown("### Conservative Option (Above / Closest)")
        for k, v in result["conservative_option"].items():
            st.write(f"{k.replace('_',' ').title()}: **{v}**")

    # --------------------------------------------------
    # ENGINEERING FLAGS
    # --------------------------------------------------

    lean_mwh = result["lean_option"]["installed_mwh"]

    if lean_mwh < installed_mwh_target:
        st.warning(
            "Lean option results in LOWER installed energy than requested. "
            "Use Conservative option if exact or higher MWh is required."
        )

    if result["lean_option"]["system_c_rate"] > 0.5:
        st.warning(
            "System C-rate exceeds 0.5C. Verify battery power capability."
        )

    st.info(
        "Apply these values directly in Storlytics:\n\n"
        "• Modules per Rack\n"
        "• Racks per Inverter\n"
        "• Number of Inverters\n\n"
        "System C-rate = Project MW / Installed MWh\n"
        "Block C-rate = Inverter MW / DC energy per inverter"
    )





