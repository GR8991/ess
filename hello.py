import streamlit as st
import math
from docx import Document
from io import BytesIO
import smtplib
from email.message import EmailMessage

st.set_page_config(page_title="PGR-Earthing Design Calculator", layout="centered")
st.title("⚡ PGR - Earthing Design Calculator")

# Input Section
st.sidebar.header("Input Parameters")
inputs = {
    "Earth Resistance (Ω)": st.sidebar.number_input("Earth Resistance (Ω)", value=1.0),
    "Surface Type": st.sidebar.selectbox("Surface Type", ["Crushed", "Asphalt"]),
    "Layer Fault Duration (s)": st.sidebar.number_input("Layer Fault Duration (s)", value=1.0),
    "Layer Thickness (m)": st.sidebar.number_input("Layer Thickness (m)", value=0.15),
    "Df - Decrement Factor": st.sidebar.number_input("Df - Decrement Factor", value=1.3),
    "Sf - Current Division Factor": st.sidebar.number_input("Sf - Current Division Factor", value=0.5),
    "If - Ground Fault Current (kA)": st.sidebar.number_input("If - Symmetrical Ground Fault Current (kA)", value=10.0),
    "Lc - Total Earth Conductor (m)": st.sidebar.number_input("Lc - Total Earth Conductor (m)", value=300.0),
    "No. of Electrodes": st.sidebar.number_input("No. of Electrodes", min_value=1, step=1, value=30),
    "Length of Each Electrode (m)": st.sidebar.number_input("Length of Each Electrode (m)", value=3),
    "Selected Grid Spacing (m)": st.sidebar.number_input("Selected Grid Spacing (m)", value=0.1),
    "Area of Earth Mat (m²)": st.sidebar.number_input("Area of Earth Mat (m²)", value=400.0),
    "Depth of Buried Conductor (m)": st.sidebar.number_input("Depth of Buried Conductor (m)", value=0.6),
    "Perimeter of the Grid (m)": st.sidebar.number_input("Perimeter of the Grid (m)", value=80.0),
    "Max Length in X (m)": st.sidebar.number_input("Max Length in X (m)", value=20.0),
    "Max Length in Y (m)": st.sidebar.number_input("Max Length in Y (m)", value=20.0),
    "Electrode Width (mm)": st.sidebar.number_input("Electrode Width (mm)", value=50.0),
    "Electrode Thickness (mm)": st.sidebar.number_input("Electrode Thickness (mm)", value=6.0)
}

# Assign variables
c = inputs["Earth Resistance (Ω)"]
g = inputs["Surface Type"]
b = inputs["Layer Fault Duration (s)"]
d = inputs["Layer Thickness (m)"]
df = inputs["Df - Decrement Factor"]
sf = inputs["Sf - Current Division Factor"]
If = inputs["If - Ground Fault Current (kA)"]
lc = inputs["Lc - Total Earth Conductor (m)"]
ne = inputs["No. of Electrodes"]
le = inputs["Length of Each Electrode (m)"]
D = inputs["Selected Grid Spacing (m)"]
area = inputs["Area of Earth Mat (m²)"]
h = inputs["Depth of Buried Conductor (m)"]
p = inputs["Perimeter of the Grid (m)"]
lx = inputs["Max Length in X (m)"]
ly = inputs["Max Length in Y (m)"]
width = inputs["Electrode Width (mm)"]
Thick = inputs["Electrode Thickness (mm)"]

# Intermediate Calculations
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

a = 3000 if g == "Crushed" else 10000
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

# Results dictionary
results = {
    "Touch Voltage (50kg)": f"{round(z_50, 2)} V",
    "Touch Voltage (70kg)": f"{round(p_70, 2)} V",
    "Step Voltage (50kg)": f"{round(y_50, 2)} V",
    "Step Voltage (70kg)": f"{round(x_70, 2)} V",
    "IG (kA)": f"{IG:.3f}",
    "Rg (Ω)": f"{Rg:.3f}",
    "Step Potential": f"{es:.3f} V",
    "Touch Potential": f"{em:.3f} V"
}

# Display results in Streamlit
st.subheader("📊 Results Summary")
st.table(results)

# Email option
st.subheader("📧 Send Report to Email (Optional)")
receiver_email = st.text_input("Enter email to receive report (leave blank to skip):")

# Create document
doc = Document()
doc.add_heading("PGR-Earthing Design Report", level=0)

doc.add_heading("Input Parameters", level=1)
table1 = doc.add_table(rows=1, cols=2)
table1.style = 'Light List Accent 1'
table1.rows[0].cells[0].text = "Parameter"
table1.rows[0].cells[1].text = "Value"

for key, val in inputs.items():
    row = table1.add_row().cells
    row[0].text = str(key)
    row[1].text = str(val)

doc.add_heading("Calculation Results", level=1)
table2 = doc.add_table(rows=1, cols=2)
table2.style = 'Light Grid Accent 1'
table2.rows[0].cells[0].text = "Parameter"
table2.rows[0].cells[1].text = "Value"

for key, val in results.items():
    row = table2.add_row().cells
    row[0].text = str(key)
    row[1].text = str(val)

buffer = BytesIO()
doc.save(buffer)
buffer.seek(0)

# Download button
st.download_button(
    label="📥 Download Word Report",
    data=buffer,
    file_name="earthing_report.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

# Send email if address is provided
if receiver_email:
    sender_email = "youremail@gmail.com"  # replace with your email
    sender_password = "yourapppassword"   # replace with app password

    try:
        msg = EmailMessage()
        msg["Subject"] = "Earthing Design Report"
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg.set_content("Please find the attached earthing design report.")

        buffer.seek(0)
        msg.add_attachment(buffer.read(), maintype="application",
                           subtype="vnd.openxmlformats-officedocument.wordprocessingml.document",
                           filename="earthing_report.docx")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)

        st.success(f"Report successfully sent to {receiver_email} 📬")

    except Exception as e:
        st.error(f"Failed to send email: {e}")
