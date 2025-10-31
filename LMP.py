import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Synthetic grid and LMP data for demonstration
# Replace this with data from ERCOT/ISO APIs or Python grid analytics libraries
grid_nodes = pd.DataFrame({
    'node': ['A', 'B', 'C', 'D', 'E'],
    'lat': [30.26, 30.28, 30.31, 30.33, 30.35],
    'lon': [-97.74, -97.76, -97.77, -97.78, -97.79],
    'energy': [50, 53, 51, 55, 52],
    'congestion': [10, 12, 19, 7, 14],
    'loss': [1, 2, 1.5, 1.7, 1.2]
})
grid_nodes['LMP'] = grid_nodes['energy'] + grid_nodes['congestion'] + grid_nodes['loss']

st.title("Grid Analytics & LMP Dashboard")
st.markdown("""
This dashboard visualizes synthetic Locational Marginal Prices (LMP) and grid congestion.
Replace sample data with real values via grid APIs or analytics libraries for production deployment.
""")

st.header("Grid Node Summary")
st.dataframe(grid_nodes)

st.header("LMP Components Visualization")
lmp_plot = px.bar(
    grid_nodes, x='node', y=['energy', 'congestion', 'loss', 'LMP'],
    title="LMP and Components by Node",
    barmode='group',
    labels={'value': 'Price ($/MWh)', 'node': 'Grid Node'}
)
st.plotly_chart(lmp_plot)

st.header("Grid Map Visualization")
st.map(grid_nodes[['lat', 'lon']])

st.header("Congestion Analysis")
max_congestion = grid_nodes.loc[grid_nodes['congestion'].idxmax()]
st.write(f"The most congested node is {max_congestion['node']} with congestion price {max_congestion['congestion']} $/MWh.")

st.write("""
### Instructions:
- To connect to real ISO data, replace the sample dataset with API data.
- To perform actual LMP calculations, integrate with PYPOWER or pandapower and feed results here.
- Enhance map with path flows using libraries like Folium or Plotly.
- Add time series support for temporal analysis.
""")
