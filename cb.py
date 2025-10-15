import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title("Circuit Breaker Operation Visualization")

st.markdown("""
This app shows a 60 Hz AC waveform and demonstrates where the breaker interrupts in terms of waveform cycles.
""")

# Parameters for waveform generation
fs = 10000  # sampling frequency (Hz)
t_end = 0.1  # duration in seconds (100 ms)
freq = 60  # frequency (Hz)

# Generate time and waveform
t = np.linspace(0, t_end, int(fs * t_end))
wave = np.sin(2 * np.pi * freq * t)
df = pd.DataFrame({'time_ms': t * 1000, 'current': wave})

# User input for interrupting cycles
cycles = st.slider("Interrupting Time (cycles)", min_value=1, max_value=6, value=3, step=1)
ms_per_cycle = 1000 / freq  # milliseconds per cycle
interrupt_ms = cycles * ms_per_cycle

# Create a column to mark interrupted region
df['interrupted'] = df['time_ms'] >= interrupt_ms

# Base waveform chart
base = alt.Chart(df).mark_line().encode(
    x=alt.X('time_ms', title='Time (ms)'),
    y=alt.Y('current', title='Current (pu)')
)

# Highlight interrupted portion
highlight = alt.Chart(df[df['interrupted']]).mark_area(color='red', opacity=0.3).encode(
    x='time_ms',
    y='current'
)

# Vertical line at interruption time
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
