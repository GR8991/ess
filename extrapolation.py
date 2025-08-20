import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from io import BytesIO

st.title("Prediction Tool: Interpolation & Extrapolation")

# User inputs
x_values = st.text_input("Enter X values (comma separated)", "0,14")
y_values = st.text_input("Enter Y values (comma separated)", "100,96.78")

mode = st.radio("Choose Mode", ["Extrapolation", "Interpolation"])

if mode == "Extrapolation":
    future_intervals = st.number_input("Enter how many future intervals to predict", min_value=1, value=12)

if mode == "Interpolation":
    interp_points = st.text_input("Enter X value(s) to interpolate (comma separated)", "7")

if st.button("Run"):
    X = np.array([float(i) for i in x_values.split(",")])
    Y = np.array([float(i) for i in y_values.split(",")])

    plt.figure(figsize=(8, 5))
    plt.plot(X, Y, 'bo-', label="Input Values")

    if mode == "Extrapolation":
        # Linear regression for extrapolation
        X_reshaped = X.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_reshaped, Y)

        last_x = X[-1]
        future_X = np.array([last_x + i for i in range(1, future_intervals + 1)]).reshape(-1, 1)
        future_Y = model.predict(future_X)

        plt.plot(future_X, future_Y, 'ro--', label="Extrapolated Values")

        # Labels
        for (x, y) in zip(X, Y):
            plt.text(x, y, f"{y:.2f}", fontsize=9, ha='right', va='bottom', color='blue')
        for (x, y) in zip(future_X.flatten(), future_Y):
            plt.text(x, y, f"{y:.2f}", fontsize=9, ha='left', va='bottom', color='red')

        # Show predictions
        st.write("### Predicted Future Values")
        for i, val in enumerate(future_Y, start=1):
            st.write(f"X = {last_x + i}, Predicted Y = {val:.2f}")

    elif mode == "Interpolation":
        # Decide method automatically
        method = "linear" if len(X) <= 2 else "cubic"
        f = interp1d(X, Y, kind=method)

        interp_X = np.array([float(i) for i in interp_points.split(",")])
        interp_Y = f(interp_X)

        plt.plot(interp_X, interp_Y, 'gs', label="Interpolated Values")

        # Labels
        for (x, y) in zip(X, Y):
            plt.text(x, y, f"{y:.2f}", fontsize=9, ha='right', va='bottom', color='blue')
        for (x, y) in zip(interp_X, interp_Y):
            plt.text(x, y, f"{y:.2f}", fontsize=9, ha='left', va='bottom', color='green')

        # Show interpolated results
        st.write("### Interpolated Values")
        for x, val in zip(interp_X, interp_Y):
            st.write(f"X = {x}, Interpolated Y = {val:.2f}")

    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title(f"{mode} with Data Labels")
    plt.legend()

    # Show plot in Streamlit
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title(f"{mode} with Data Labels")
    plt.legend()
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # Light grid
    st.pyplot(plt)

    # Save as JPG
    buf = BytesIO()
    plt.savefig(buf, format="jpg")
    st.download_button("Download Graph as JPG", data=buf.getvalue(),
                       file_name="prediction_graph.jpg", mime="image/jpeg")
