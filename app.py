import numpy as np
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from SimulationLois import loi_uniforme, loi_exponentielle, loi_cauchy, loi_bernoulli, loi_normale


st.set_page_config(layout="wide")
st.markdown("""
<style>
.stApp { max-width: 1200000px; margin: 0 auto; }
.distribution-controls { background: white; padding: 2rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("Statistical Distributions Simulator")

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown('<div class="distribution-controls">', unsafe_allow_html=True)
    distribution = st.selectbox("Distribution", ["Uniform", "Exponential", "Cauchy", "Bernoulli", "Normal"])
    n_samples = st.slider("Number of samples", 100, 10000, 1000)

    if distribution == "Uniform":
        a = st.number_input("Min (a)", value=0.0)
        b = st.number_input("Max (b)", value=1.0)
        samples = loi_uniforme(a, b, n_samples)
    elif distribution == "Exponential":
        lambda_exp = st.number_input("λ (lambda)", min_value=0.1, value=1.0)
        samples = loi_exponentielle(lambda_exp, n_samples)
    elif distribution == "Cauchy":
        c = st.number_input("c (scale)", min_value=0.1, value=1.0)
        samples = loi_cauchy(c, n_samples)
    elif distribution == "Bernoulli":
        p = st.number_input("p (probability)", min_value=0.0, max_value=1.0, value=0.5)
        samples = loi_bernoulli(p, n_samples)
    else:
        samples = loi_normale(n_samples)

    if st.button("Generate", type="primary"):
        st.session_state.samples = samples
        st.session_state.stats = {
            "Mean": np.mean(samples),
            "Std Dev": np.std(samples),
            "Min": np.min(samples),
            "Max": np.max(samples)
        }

with col2:
    if "samples" in st.session_state:
        samples = st.session_state.samples

        # Vérifier si les échantillons sont continus ou discrets
        if distribution in ["Bernoulli"]:
            # Graphique pour des distributions discrètes
            unique, counts = np.unique(samples, return_counts=True)
            fig = go.Figure(data=[go.Bar(x=unique, y=counts)])
            fig.update_layout(
                title=f"Distribution of {distribution} Samples",
                xaxis_title="Values",
                yaxis_title="Counts",
                showlegend=False
            )
        else:
            # Graphique pour des distributions continues
            density = gaussian_kde(samples)
            x_vals = np.linspace(min(samples), max(samples), 500)
            y_vals = density(x_vals)

            fig = go.Figure(data=[go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(color='blue', width=2)
            )])
            fig.update_layout(
                title=f"Probability Density Function (PDF) of {distribution} Samples",
                xaxis_title="Values",
                yaxis_title="Density",
                showlegend=False
            )

        st.plotly_chart(fig)

        st.write("Statistics:")
        st.json(st.session_state.stats)

