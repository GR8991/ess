# streamlit_app.py
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

st.set_page_config("BESS 115 kV Ring-Bus Fault Simulator")

st.title("🔌 115 kV Ring-Bus Switching Station – Fault Simulation")
st.caption("BESS → GSU → Gen-Tie → Ring-Bus → Utility")

# --- Define electrical network ---
G = nx.Graph()
edges = [
    ("BESS", "GSU"),
    ("GSU", "CB1"),
    ("CB1", "BusA"),
    ("BusA", "CB2"),
    ("CB2", "Utility"),
    ("BusA", "CB3"),
    ("CB3", "BusB"),
    ("BusB", "CB1")
]
G.add_edges_from(edges)

# approximate coordinates for plotting
pos = {
    "BESS": (0, 0),
    "GSU": (1, 0),
    "CB1": (2, 0.5),
    "BusA": (3, 0.5),
    "CB2": (4, 0.5),
    "Utility": (5, 0),
    "CB3": (3, -0.5),
    "BusB": (2, -0.5),
}

# --- Initialize session state ---
if "fault" not in st.session_state:
    st.session_state.fault = None
if "breakers" not in st.session_state:
    st.session_state.breakers = {"CB1": True, "CB2": True, "CB3": True}

st.sidebar.header("Simulation Controls")

# --- Fault selection ---
fault_type = st.sidebar.selectbox(
    "Select Fault Location",
    ["No Fault", "Gen-Tie Line", "Utility Line", "Bus-A Section", "Bus-B Section"]
)

# --- Fault injection logic ---
if fault_type == "No Fault":
    st.session_state.fault = None
    st.session_state.breakers = {"CB1": True, "CB2": True, "CB3": True}

elif fault_type == "Gen-Tie Line":
    st.session_state.fault = "Gen-Tie"
    st.session_state.breakers["CB1"] = False
    st.session_state.breakers["CB2"] = True
    st.session_state.breakers["CB3"] = True

elif fault_type == "Utility Line":
    st.session_state.fault = "Utility"
    st.session_state.breakers["CB1"] = True
    st.session_state.breakers["CB2"] = False
    st.session_state.breakers["CB3"] = True

elif fault_type == "Bus-A Section":
    st.session_state.fault = "BusA"
    # Bus-A fault clears both connected breakers
    st.session_state.breakers["CB1"] = False
    st.session_state.breakers["CB2"] = False
    st.session_state.breakers["CB3"] = True

elif fault_type == "Bus-B Section":
    st.session_state.fault = "BusB"
    # Bus-B fault isolates CB3 only
    st.session_state.breakers["CB1"] = True
    st.session_state.breakers["CB2"] = True
    st.session_state.breakers["CB3"] = False

# --- Draw network using Plotly ---
fig = go.Figure()
for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    color = "green"
    width = 3

    # color rules
    fault = st.session_state.fault
    brk = st.session_state.breakers

    if fault == "Gen-Tie" and ("GSU" in (u, v) or "CB1" in (u, v)):
        color, width = "red", 6
    elif fault == "Utility" and ("Utility" in (u, v) or "CB2" in (u, v)):
        color, width = "red", 6
    elif fault == "BusA" and ("BusA" in (u, v)):
        color, width = "red", 6
    elif fault == "BusB" and ("BusB" in (u, v)):
        color, width = "red", 6
    # breaker open = gray
    if ("CB1" in (u, v) and not brk["CB1"]) \
       or ("CB2" in (u, v) and not brk["CB2"]) \
       or ("CB3" in (u, v) and not brk["CB3"]):
        color, width = "lightgray", 2

    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1],
        mode="lines+text",
        text=[u, v],
        textposition="top center",
        line=dict(color=color, width=width),
        hoverinfo="text",
        showlegend=False
    ))

fig.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    plot_bgcolor="white",
    height=550,
    title=f"Current Fault: {fault_type}"
)
st.plotly_chart(fig, use_container_width=True)

# --- Status summary ---
st.subheader("Breaker Status")
st.write(st.session_state.breakers)

st.subheader("Interpretation")
if fault_type == "No Fault":
    st.success("System healthy: all breakers closed; power flows BESS → CB1 → Bus-A → CB2 → Utility. CB3 closed but carries no load.")
elif fault_type == "Gen-Tie Line":
    st.warning("Gen-Tie fault: CB1 opened to isolate fault; CB2 & CB3 remain closed keeping buses energized from utility side.")
elif fault_type == "Utility Line":
    st.warning("Utility fault: CB2 opened; BESS & buses remain energized through CB1-CB3 path.")
elif fault_type == "Bus-A Section":
    st.error("Bus-A fault: CB1 & CB2 opened; Bus-B and CB3 remain energized; ring integrity preserved.")
elif fault_type == "Bus-B Section":
    st.error("Bus-B fault: CB3 opened; Bus-A continues service via CB1-CB2 path.")

st.caption("🟢 Green = energized, ⚪ Gray = breaker open, 🔴 Red = faulted path")
