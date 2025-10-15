# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Load waveform data
df = pd.read_csv('ac_waveform.csv')

st.title("Circuit Breaker Operation Visualization")

st.markdown("""
This app shows a 60 Hz AC waveform and demonstrates where the breaker interrupts in terms of waveform cycles.
""")

# User input: number of interrupting cycles
cycles = st.slider("Interrupting Time (cycles)", min_value=1, max_value=6, value=3, step=1)
freq = 60  # Hz
ms_per_cycle = 1000 / freq  # ms per cycle

# Compute interruption time in ms
interrupt_ms = cycles * ms_per_cycle

# Prepare highlighted data
df['interrupted'] = df['time_ms'] >= interrupt_ms

# Base waveform chart
base = alt.Chart(df).mark_line().encode(
    x=alt.X('time_ms', title='Time (ms)'),
    y=alt.Y('current', title='Current (pu)')
)

# Highlight region post-interruption
highlight = alt.Chart(df[df['interrupted']]).mark_area(color='red', opacity=0.3).encode(
    x='time_ms',
    y='current'
)

# Vertical line at interruption point
vline = alt.Chart(pd.DataFrame({'interrupt_ms': [interrupt_ms]})).mark_rule(color='red').encode(
    x='interrupt_ms'
)

# Combine charts
chart = (base + highlight + vline).properties(
    width=700,
    height=400,
    title=f'Breaker Interrupt at {cycles} Cycles ({interrupt_ms:.1f} ms)'
)

st.altair_chart(chart, use_container_width=True)
