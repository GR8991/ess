import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Title
st.title("Extrapolation Prediction Tool")

# User inputs
x_values = st.text_input("Enter X values (comma separated)", "0,14")
y_values = st.text_input("Enter Y values (comma separated)", "100,96.78")
future_intervals = st.number_input("Enter how many future intervals to predict", min_value=1, value=12)

# Process when user clicks
if st.button("Predict"):
    # Convert inputs to numpy arrays
    X = np.array([float(i) for i in x_values.split(",")]).reshape(-1, 1)
    Y = np.array([float(i) for i in y_values.split(",")])

    # Train linear regression model
    model = LinearRegression()
    model.fit(X, Y)

    # Generate future predictions
    last_x = X[-1][0]
    future_X = np.array([last_x + i for i in range(1, future_intervals + 1)]).reshape(-1, 1)
    future_Y = model.predict(future_X)

    # Combine all data for plotting
    all_X = np.vstack([X, future_X])
    all_Y = np.concatenate([Y, future_Y])

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(X, Y, 'bo-', label="Input Values")
    plt.plot(future_X, future_Y, 'ro--', label="Predicted Values")

    # Add labels on data points
    for (x, y) in zip(X.flatten(), Y):
        plt.text(x, y, f"{y:.2f}", fontsize=9, ha='right', va='bottom', color='blue')
    for (x, y) in zip(future_X.flatten(), future_Y):
        plt.text(x, y, f"{y:.2f}", fontsize=9, ha='left', va='bottom', color='red')

    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Extrapolation Prediction with Data Labels")
    plt.legend()
    st.pyplot(plt)

    # Show predicted values below graph
    st.write("### Predicted Future Values")
    for i, val in enumerate(future_Y, start=1):
        st.write(f"X = {last_x + i}, Predicted Y = {val:.2f}")
