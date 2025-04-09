import streamlit as st
import math
from fpdf import FPDF
from datetime import datetime
import tempfile

st.set_page_config(page_title="Earthing Calculation for Wind", layout="wide")
st.title("Wind Farm Earthing Design Calculator")

st.header("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    c = st.number_input("Earth resistance (Ohm)", min_value=0.0)
    g = st.selectbox("Surface Type", ["Crushed", "Asphalt"])
    b = st.number_input("Layer fault duration (s)", min_value=0.0)
    d = st.number_input("Layer thickness (m)", min_value=0.0)
    df = st.number_input("Df - Decrement factor", min_value=0.0)
    sf = st.number_input("Sf - Current division factor", min_value=0.0)
    If = st.number_input("If - Symmetrical ground fault current (kA)", min_value=0.0)
    lc = st.number_input("Lc - Total earth conductor length (m)", min_value=0.0)
    ne = st.number_input("No. of electrodes", min_value=1)
    le = st.number_input("Length of each electrode (m)", min_value=0.0)

with col2:
    D = st.number_input("Grid spacing (m)", min_value=0.0)
    area = st.number_input("Area of earth mat (sq.m)", min_value=0.0)
    h = st.number_input("Depth of buried conductor (m)", min_value=0.0)
    p = st.number_input("Perimeter of the grid (m)", min_value=0.0)
    lx = st.number_input("Length in x direction (m)", min_value=0.0)
    ly = st.number_input("Length in y direction (m)", min_value=0.0)
    width = st.number_input("Electrode width (mm)", min_value=0.0)
    Thick = st.number_input("Electrode thickness (mm)", min_value=0.0)

if st.button("Calculate"):
    Lr = ne * le
    Lt = lc * Lr
    Ls = 0.75 * lc + 0.85 * Lr
    na = 2 * lc / p
    nb = math.sqrt(p / (4 * math.sqrt(area)))
    n = na * nb
    ki = 0.644 + 0.148 * n
    s = 1 / math.pi * (1 / (2 * h) + 1 / (le + h) + 1 / 3 * 1 - 0.5 ** (n - 2))
    Ks_rounded = round(s, 4)
    lm = lc + (1.55 + 1.22 * (le / math.sqrt(lx ** 2 + ly ** 2))) * Lr
    Area_electrode = (width / 1000) * (Thick / 1000)
    dia = math.sqrt(4 * Area_electrode / math.pi)
    km = 1 / (2 * math.pi) * (
        math.log((D ** 2) / (16 * h * dia) + ((D + 2 * h) ** 2) / (8 * D * dia) - (h / (4 * dia))) +
        1 / 1.27 * math.log(8 / (math.pi * (2 * n - 1)))
    )

    a = 3000 if g == "Crushed" else 10000 if g == "Asphalt" else 0

    if a != 0:
        e = 1 - (0.09 * (1 - (c / a))) / ((2 * d) + 0.09)
        f = round(e, 3)

        p_70 = ((1000 + (1.5 * f * a)) * 0.157) / math.sqrt(b)
        z_50 = ((1000 + (1.5 * f * a)) * 0.116) / math.sqrt(b)
        x_70 = ((1000 + (6 * f * a)) * 0.157) / math.sqrt(b)
        y_50 = ((1000 + (6 * f * a)) * 0.116) / math.sqrt(b)

        IG = df * sf * If

        Rg = (c * (1 / Lt + 1 / math.sqrt(20 * area) * (1 + 1 / (1 + h * math.sqrt(20 / area)))) *
              1.52 * (2 * math.log(p_70 * math.sqrt(2 / area) - 1) * math.sqrt(area) / p_70))

        es = (c * IG * Ks_rounded * ki) / Ls * 1000
        em = (c * IG * km * ki) / lm * 1000

        st.subheader("Calculation Results")
        st.markdown(f"**Touch Voltage (50kg):** {round(z_50, 3)} V")
        st.markdown(f"**Touch Voltage (70kg):** {round(p_70, 3)} V")
        st.markdown(f"**Step Voltage (50kg):** {round(y_50, 3)} V")
        st.markdown(f"**Step Voltage (70kg):** {round(x_70, 3)} V")
        st.markdown(f"**IG (kA):** {IG}")
        st.markdown(f"**Rg (Ohm):** {Rg}")
        st.markdown(f"**Step Potential:** {es} V")
        st.markdown(f"**Touch Potential:** {em} V")

        if st.checkbox("Download results as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, txt="Wind Farm Earthing Calculation Report", ln=True, align="C")
            pdf.ln(10)

            # Add table of results
            pdf.set_font("Arial", size=11)
            results = [
                ("Touch Voltage (50kg)", f"{round(z_50, 3)} V"),
                ("Touch Voltage (70kg)", f"{round(p_70, 3)} V"),
                ("Step Voltage (50kg)", f"{round(y_50, 3)} V"),
                ("Step Voltage (70kg)", f"{round(x_70, 3)} V"),
                ("IG (kA)", f"{IG}"),
                ("Rg (Ohm)", f"{Rg}"),
                ("Step Potential", f"{es} V"),
                ("Touch Potential", f"{em} V")
            ]

            for name, value in results:
                pdf.cell(90, 10, name, 1)
                pdf.cell(90, 10, value, 1, ln=True)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pdf.ln(10)
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, f"Generated on: {timestamp}", ln=True, align="R")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                pdf.output(tmpfile.name)
                st.download_button("Download PDF", data=open(tmpfile.name, "rb").read(), file_name="earthing_results.pdf")

    else:
        st.warning("Calculation skipped due to invalid surface selection or input.")
