# streamlit_app.py
import streamlit as st
import plotly.graph_objects as go
import networkx as nx

st.set_page_config("BESS Ring-Bus Fault Simulator")

# --- Define electrical model ---
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
pos = {
    "BESS":(0,0), "GSU":(1,0), "CB1":(2,0.5),
    "BusA":(3,0.5),"CB2":(4,0.5),"Utility":(5,0),
    "CB3":(3,-0.5),"BusB":(2,-0.5)
}

# --- Initialize session state ---
if "fault" not in st.session_state:
    st.session_state.fault = None
if "breakers" not in st.session_state:
    st.session_state.breakers = {"CB1":True,"CB2":True,"CB3":True}

st.sidebar.write("Click on an element to simulate a fault.")
st.sidebar.write("Current breaker states:", st.session_state.breakers)

# --- Draw network with Plotly ---
x, y = zip(*[pos[n] for n in G.nodes])
fig = go.Figure()

for u,v in G.edges():
    x0,y0 = pos[u]
    x1,y1 = pos[v]
    color = "green"
    width = 3
    if st.session_state.fault == (u,v) or st.session_state.fault == (v,u):
        color = "red"
        width = 6
    if ("CB1" in (u,v) and not st.session_state.breakers["CB1"]) \
       or ("CB2" in (u,v) and not st.session_state.breakers["CB2"]) \
       or ("CB3" in (u,v) and not st.session_state.breakers["CB3"]):
        color = "gray"
    fig.add_trace(go.Scatter(
        x=[x0,x1], y=[y0,y1],
        mode="lines+markers+text",
        line=dict(color=color,width=width),
        text=[u,v], textposition="top center",
        hoverinfo="text",
        customdata=[(u,v)],
        name=f"{u}-{v}"
    ))

fig.update_layout(
    showlegend=False,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    height=500
)
clicked = st.plotly_chart(fig, on_click="ignore")  # Placeholder: Streamlit Cloud blocks JS callbacks

# --- Handle fault simulation manually ---
fault_choice = st.selectbox("Select fault location", list(G.edges()))
if st.button("Inject Fault"):
    st.session_state.fault = fault_choice
    # Determine which breaker opens
    if "CB1" in fault_choice:
        st.session_state.breakers["CB1"] = False
    elif "CB2" in fault_choice:
        st.session_state.breakers["CB2"] = False
    elif "CB3" in fault_choice:
        st.session_state.breakers["CB3"] = False
    else:
        st.session_state.breakers = {"CB1":True,"CB2":True,"CB3":True}

st.experimental_rerun()
