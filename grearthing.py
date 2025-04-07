

import streamlit as st
import math
from fpdf import FPDF
import tempfile

st.title("*PGR* - Earth Touch & Step Voltage Calculator")

# User Inputs
c = st.number_input("Enter Earth Resistance (Ohm):", value=1.0)
g = st.selectbox("Select Ground Type:", ["Crushed", "Asphalt"])
b = st.number_input("Enter Layer Fault Duration (s):", value=1.0)
d = st.number_input("Enter Layer Thickness (m):", value=0.1)
df = st.number_input("Enter Df (Decrement Factor):", value=1.0)
sf = st.number_input("Enter Sf (Current Division Factor):", value=1.0)
If = st.number_input("Enter If (Symmetrical Ground Fault Current in kA):", value=1.0)
lc = st.number_input("Enter Lc (Total Earth Conductor for WTG & USS in m):", value=1.0)
ne = st.number_input("Enter Number of Electrodes:", step=1, value=1)
le = st.number_input("Enter Length of Each Electrode (m):", value=1.0)
area = st.number_input("Enter Area of Earth Mat (m²):", value=1.0)
h = st.number_input("Enter h (Depth of Buried Conductor in m):", value=0.5)
Lp = st.number_input("Enter Perimeter of Grid (Lp in m):", value=1.0)

# Ground type factor 'a'
a = 3000 if g == "Crushed" else 10000 if g == "Asphalt" else 0

if a == 0:
    st.error("Invalid Ground Type selected. Aborting calculation.")
else:
    # Calculate Cs
    Cs = 1 - (0.09 * (1 - (c / a))) / ((2 * d) + 0.09)
    f = round(Cs, 3)

    # Voltage Calculations
    V_70kg_touch = ((1000 + (1.5 * f * a)) * 0.157) / math.sqrt(b)
    V_50kg_touch = ((1000 + (1.5 * f * a)) * 0.116) / math.sqrt(b)
    V_70kg_step = ((1000 + (6 * f * a)) * 0.157) / math.sqrt(b)
    V_50kg_step = ((1000 + (6 * f * a)) * 0.116) / math.sqrt(b)

    # IG Calculation
    IG = df * sf * If

    # Grid Resistance Rg
    Lr = ne * le
    Lt = lc * Lr
    try:
        Rg = (
            c *
            (1 / Lt + 1 / math.sqrt(20 * area) * (1 + 1 / (1 + h * math.sqrt(20 / area)))) *
            1.52 *
            (2 * math.log(Lp * math.sqrt(2 / area) - 1) * math.sqrt(area) / Lp)
        )
    except:
        Rg = float('nan')
        st.warning("Error in Rg calculation (check inputs for valid values).")

    # Display Outputs
    st.subheader("Results")
    st.markdown("### Tolerable Touch Voltage")
    st.write(f"50 kg touch voltage: **{round(V_50kg_touch, 3)} V**")
    st.write(f"70 kg touch voltage: **{round(V_70kg_touch, 3)} V**")

    st.markdown("### Step Voltage")
    st.write(f"50 kg step voltage: **{round(V_50kg_step, 3)} V**")
    st.write(f"70 kg step voltage: **{round(V_70kg_step, 3)} V**")

    st.markdown("### Other Results")
    st.write(f"IG value: **{IG:.2f} kA**")
    st.write(f"Rg value: **{Rg:.2f} Ω**")

    # PDF Generation Option
    if st.checkbox("Do you want to generate a PDF report?"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="PGR - Earth Touch & Step Voltage Calculator Report", ln=True, align='C')
        pdf.ln(5)

        # Inputs
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Inputs:", ln=True)
        pdf.set_font("Arial", size=11)
        inputs = [
            f"Earth Resistance: {c} Ohm",
            f"Ground Type: {g}",
            f"Layer Fault Duration: {b} s",
            f"Layer Thickness: {d} m",
            f"Df: {df}",
            f"Sf: {sf}",
            f"If: {If} kA",
            f"Lc: {lc} m",
            f"Number of Electrodes: {ne}",
            f"Length of Each Electrode: {le} m",
            f"Area of Earth Mat: {area} m²",
            f"Depth of Buried Conductor (h): {h} m",
            f"Perimeter of Grid (Lp): {Lp} m"
        ]
        for line in inputs:
            pdf.cell(200, 8, txt=line, ln=True)

        # Results Table
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Results (in Table):", ln=True)

        pdf.set_font("Arial", size=11)
        table_data = [
            ["Parameter", "Value", "Unit"],
            ["Touch Voltage (50kg)", round(V_50kg_touch, 3), "V"],
            ["Touch Voltage (70kg)", round(V_70kg_touch, 3), "V"],
            ["Step Voltage (50kg)", round(V_50kg_step, 3), "V"],
            ["Step Voltage (70kg)", round(V_70kg_step, 3), "V"],
            ["IG (Ground Fault Current)", round(IG, 3), "kA"],
            ["Rg (Grid Resistance)", round(Rg, 3), "Ohm"]
        ]

        col_widths = [70, 40, 30]
        for row in table_data:
            for i, datum in enumerate(row):
                pdf.cell(col_widths[i], 10, txt=str(datum), border=1)
            pdf.ln()

        # Save and offer download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            st.success("PDF report generated.")
            with open(tmp_file.name, "rb") as f:
                st.download_button("Download PDF", f, file_name="PGR_touch_step_voltage_report.pdf")


