

''''import streamlit as st
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
                st.download_button("Download PDF", f, file_name="PGR_touch_step_voltage_report.pdf")'''

import streamlit as st
import math
import tempfile
import datetime
import requests
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from pathlib import Path
import pandas as pd
import os

# Constants
LOGO_PATH = "logo.png"
FONT_URL = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"
FONT_NAME = "Roboto"
CSV_LOG = "report_log.csv"

# Download Google Font if not already present
font_path = Path("Roboto-Regular.ttf")
if not font_path.exists():
    response = requests.get(FONT_URL)
    font_path.write_bytes(response.content)

# Streamlit UI
st.title("*PGR* - Earth Touch & Step Voltage Calculator")

author = st.text_input("Enter your name (for report):", value="Engineer")
report_note = st.text_area("Add additional notes to the report (optional):", value="")

# Inputs
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
    Cs = 1 - (0.09 * (1 - (c / a))) / ((2 * d) + 0.09)
    f = round(Cs, 3)

    V_70kg_touch = ((1000 + (1.5 * f * a)) * 0.157) / math.sqrt(b)
    V_50kg_touch = ((1000 + (1.5 * f * a)) * 0.116) / math.sqrt(b)
    V_70kg_step = ((1000 + (6 * f * a)) * 0.157) / math.sqrt(b)
    V_50kg_step = ((1000 + (6 * f * a)) * 0.116) / math.sqrt(b)

    IG = df * sf * If

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

    # Show results
    st.subheader("Results")
    st.write(f"50 kg touch voltage: **{round(V_50kg_touch, 3)} V**")
    st.write(f"70 kg touch voltage: **{round(V_70kg_touch, 3)} V**")
    st.write(f"50 kg step voltage: **{round(V_50kg_step, 3)} V**")
    st.write(f"70 kg step voltage: **{round(V_70kg_step, 3)} V**")
    st.write(f"IG: **{IG:.2f} kA**")
    st.write(f"Rg: **{Rg:.2f} Ω**")

    if st.checkbox("Generate PDF Report with Logo & Footer"):
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.add_page()

        # Register and set custom Google Font
        pdf.add_font(FONT_NAME, style="", fname=str(font_path))
        pdf.set_font(FONT_NAME, "", 12)

        # Logo
        try:
            pdf.image(LOGO_PATH, x=10, y=8, w=30)
        except Exception as e:
            st.warning(f"Could not load logo: {e}")

        # Title
        pdf.set_font(FONT_NAME, size=16)
        pdf.cell(0, 15, "PGR - Touch & Step Voltage Report", align="C", new_x=XPos.LEFT, new_y=YPos.NEXT)
        pdf.ln(5)

        # Inputs
        pdf.set_font(FONT_NAME, "B", 13)
        pdf.cell(0, 8, "Input Parameters", new_x=XPos.LEFT, new_y=YPos.NEXT)
        pdf.set_font(FONT_NAME, size=11)
        inputs = [
            ("Earth Resistance", f"{c} Ω"), ("Ground Type", g), ("Fault Duration", f"{b} s"),
            ("Layer Thickness", f"{d} m"), ("Df", df), ("Sf", sf),
            ("If", f"{If} kA"), ("Lc", f"{lc} m"), ("No. of Electrodes", ne),
            ("Electrode Length", f"{le} m"), ("Area of Earth Mat", f"{area} m²"),
            ("Buried Conductor Depth", f"{h} m"), ("Perimeter", f"{Lp} m")
        ]
        for name, val in inputs:
            pdf.cell(70, 8, name + ":", border=0)
            pdf.cell(60, 8, str(val), new_x=XPos.LEFT, new_y=YPos.NEXT)

        # Results Table
        pdf.ln(5)
        pdf.set_font(FONT_NAME, "B", 12)
        pdf.cell(0, 8, "Calculated Results", new_x=XPos.LEFT, new_y=YPos.NEXT)
        pdf.set_fill_color(230, 230, 250)
        pdf.set_font(FONT_NAME, "", 11)

        headers = ["Parameter", "Value", "Unit"]
        results = [
            ["Touch Voltage (50kg)", round(V_50kg_touch, 3), "V"],
            ["Touch Voltage (70kg)", round(V_70kg_touch, 3), "V"],
            ["Step Voltage (50kg)", round(V_50kg_step, 3), "V"],
            ["Step Voltage (70kg)", round(V_70kg_step, 3), "V"],
            ["IG (Ground Fault Current)", round(IG, 3), "kA"],
            ["Rg (Grid Resistance)", round(Rg, 3), "Ω"]
        ]

        col_widths = [70, 40, 30]
        for i, head in enumerate(headers):
            pdf.cell(col_widths[i], 8, head, border=1, fill=True)
        pdf.ln()

        for row in results:
            for i, item in enumerate(row):
                pdf.cell(col_widths[i], 8, str(item), border=1)
            pdf.ln()

        # Notes Section
        if report_note.strip():
            pdf.ln(5)
            pdf.set_font(FONT_NAME, "B", 12)
            pdf.cell(0, 8, "Additional Notes", new_x=XPos.LEFT, new_y=YPos.NEXT)
            pdf.set_font(FONT_NAME, "", 11)
            pdf.multi_cell(0, 8, report_note)

        # Footer
        pdf.set_y(-30)
        pdf.set_font(FONT_NAME, "I", 9)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 10, f"Generated on: {timestamp}", ln=True)
        pdf.cell(0, 10, f"Report Author: {author}", ln=True)

        # Save PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            st.success("PDF report generated successfully!")
            with open(tmp_file.name, "rb") as f:
                st.download_button("📄 Download PDF", f, file_name="touch_step_voltage_report.pdf")

        # --- CSV Logging ---
        log_entry = {
            "Timestamp": timestamp,
            "Author": author,
            "Note": report_note,
            "IG (kA)": round(IG, 3),
            "Rg (Ohm)": round(Rg, 3),
            "V50_Touch (V)": round(V_50kg_touch, 3),
            "V70_Touch (V)": round(V_70kg_touch, 3),
            "V50_Step (V)": round(V_50kg_step, 3),
            "V70_Step (V)": round(V_70kg_step, 3)
        }

        if os.path.exists(CSV_LOG):
            df = pd.read_csv(CSV_LOG)
            df = df.append(log_entry, ignore_index=True)
        else:
            df = pd.DataFrame([log_entry])

        df.to_csv(CSV_LOG, index=False)
        st.info("📁 Report details logged to `report_log.csv`")



