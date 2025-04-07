import streamlit as st
import math

st.title("Touch & Step Voltage Calculator")

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
    st.write(f"Rg value: **{Rg:.2f} Ω**")
