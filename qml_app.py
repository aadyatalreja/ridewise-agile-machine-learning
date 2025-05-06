import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# --- UI Setup ---
st.set_page_config(page_title="RideWise QML Analysis", layout="centered")
st.markdown("""
    <style>
    .gradient-text {
        font-size: 2.2em;
        font-weight: bold;
        background: linear-gradient(90deg, #11998e, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    div.stButton > button {
        background: linear-gradient(to right, #11998e, #38ef7d);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75em 1em;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="gradient-text">RideWise: Quantum ML Membership Prediction</div>', unsafe_allow_html=True)

# --- Load & Clean Data ---
try:
    df = pd.read_csv("2010-capitalbikeshare-tripdata.csv")
except FileNotFoundError:
    st.error("❌ File not found: 2010-capitalbikeshare-tripdata.csv")
    st.stop()

# ✅ USE ONLY NUMERIC COLUMNS
station_encoder = LabelEncoder()
df['Start station'] = station_encoder.fit_transform(df['Start station'])
df = df[['Duration', 'Start station', 'Member type']].dropna()
le = LabelEncoder()
df['Member type'] = le.fit_transform(df['Member type'])

X = df[['Duration', 'Start station']].values
y = df['Member type'].values

# Handle binary classification requirement
if len(np.unique(y)) > 2:
    allowed = np.unique(y)[:2]
    mask = np.isin(y, allowed)
    X = X[mask]
    y = y[mask]

try:
    X = X.astype(np.float64)
except ValueError as e:
    st.error(f"Conversion Error: {e}")
    st.stop()

X = np.nan_to_num(X)

# --- Split & Scale ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train_mapped = 2 * y_train - 1
y_test_mapped = 2 * y_test - 1

# --- QML Models ---
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def basic_qml(weights, x):
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def deep_qml(weights, x):
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[1, 0])
    qml.RX(weights[2], wires=0)
    qml.RX(weights[3], wires=1)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def entangled_qml(weights, x):
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[0], wires=0)
    qml.RZ(weights[1], wires=1)
    qml.CNOT(wires=[1, 0])
    qml.RY(weights[2], wires=0)
    qml.RY(weights[3], wires=1)
    return qml.expval(qml.PauliZ(0))

def cost_fn(model, weights, X, Y):
    return np.mean([(model(weights, x) - y) ** 2 for x, y in zip(X, Y)])

def train_qml(model, X, Y, weight_len=2, steps=40, lr=0.1):
    opt = NesterovMomentumOptimizer(stepsize=lr)
    weights = qml.numpy.array(np.random.random(weight_len), requires_grad=True)
    for _ in range(steps):
        weights = opt.step(lambda w: cost_fn(model, w, X, Y), weights)
    return weights

# --- Sidebar Model Select ---
model_choice = st.sidebar.selectbox("Choose QML Model", ["NONE", "Basic QML", "Deep QML", "Entangled QML"])
if model_choice == "NONE":
    st.info("⬅️ Select a Quantum ML model from the sidebar to begin.")
    st.stop()

model_map = {
    "Basic QML": (basic_qml, 2),
    "Deep QML": (deep_qml, 4),
    "Entangled QML": (entangled_qml, 4),
}

qml_model, weight_len = model_map[model_choice]

# --- Train Selected Model ---
if st.button("🔮 Train Selected QML Model"):
    with st.spinner("Training..."):
        weights = train_qml(qml_model, X_train, y_train_mapped, weight_len)
        preds = [1 if qml_model(weights, x) > 0 else 0 for x in X_test]
        acc = accuracy_score(y_test, preds)
        st.success(f"✅ {model_choice} Accuracy: {acc * 100:.2f}%")

        st.subheader("📊 Make a Custom Prediction")
        d = st.number_input("Enter Duration", value=500.0)
        s = st.number_input("Enter Start Station", value=300.0)
        user_input = scaler.transform([[d, s]])
        pred = np.sign(qml_model(weights, user_input[0]))
        label = 1 if pred > 0 else 0
        st.write(f"🔎 Prediction: **{le.inverse_transform([label])[0]}**")

# --- Compare All Models ---
if st.button("📊 Compare All QML Models"):
    acc_results = {}
    for name, (model_fn, size) in model_map.items():
        weights = train_qml(model_fn, X_train, y_train_mapped, size)
        preds = [1 if model_fn(weights, x) > 0 else 0 for x in X_test]
        acc_results[name] = accuracy_score(y_test, preds) * 100

    df_acc = pd.DataFrame(list(acc_results.items()), columns=["QML Model", "Accuracy (%)"]).sort_values(by="Accuracy (%)", ascending=False)
    st.subheader("QML Model Accuracy Comparison")
    st.dataframe(df_acc, use_container_width=True)

    fig = px.bar(df_acc, x="QML Model", y="Accuracy (%)", color="Accuracy (%)", text="Accuracy (%)", title="QML Model Comparison")
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.info("Built with ❤️ using PennyLane + Streamlit. QML supports only 2 classes currently.")