import streamlit as st
import pandas as pd
import requests
import shap
import joblib
import plotly.graph_objects as go

# ------------------------
# CONFIGURATION DE LA PAGE
# ------------------------

st.set_page_config(page_title="Scoring Conseiller IA", layout="wide")
st.title("Scoring Crédit - Conseiller IA")

# ------------------------
# CHARGEMENT DES DONNÉES
# ------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_conseiller.csv", usecols=["SK_ID_CURR", "revenu", "age", "anciennete", "nb_incidents"])
    df = df[df["anciennete"] >= 0]
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(str)
    return df

df = load_data()

# ------------------------
# SÉLECTION CLIENT (TOP N)
# ------------------------

df_top = df.sample(n=500, random_state=42)  # sous-échantillon

col_select, col_gauge, col_infos = st.columns([1, 2, 1])

with col_select:
    st.markdown("### 🔍 Sélection du client")
    selected_id = st.selectbox("Choisir un client (Top 100)", df_top["SK_ID_CURR"].tolist())

client_data = df[df["SK_ID_CURR"] == selected_id].iloc[0]

# ------------------------
# SCORE API + JAUGE
# ------------------------

payload = {
    "revenu": float(client_data["revenu"]),
    "age": int(client_data["age"]),
    "anciennete": int(client_data["anciennete"]),
    "nb_incidents": float(client_data["nb_incidents"])
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)

if response.status_code == 200:
    score = response.json()["score_credit"]
    score_percent = round(score * 100)

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score_percent,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#6ecf7e"},
                'bgcolor': "#e6f2ea",
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Erreur lors de l'appel à l'API.")
    st.stop()

# ------------------------
# INFOS CLIENT
# ------------------------

with col_infos:
    st.markdown("### Informations client")
    st.markdown(f"**Âge :** {int(client_data['age'])} ans")
    st.markdown(f"**Ancienneté :** {client_data['anciennete']} an(s)")
    st.markdown(f"**Revenu :** {int(client_data['revenu']):,} €".replace(",", " "))
    st.markdown(f"**Incidents bancaires :** {int(client_data['nb_incidents'])}")

# ------------------------
# SHAP - EXPLICATION DU SCORE
# ------------------------

st.markdown("---")
st.markdown("### Interprétation du score")

model = joblib.load("model/pipeline_simple.pkl")
explainer = shap.Explainer(model.named_steps["xgb"])

X_client = pd.DataFrame([[
    client_data["revenu"],
    client_data["age"],
    client_data["anciennete"],
    client_data["nb_incidents"]
]], columns=["revenu", "age", "anciennete", "nb_incidents"])

X_client = X_client.apply(pd.to_numeric, errors="coerce").fillna(0)
shap_values = explainer(X_client)

contributions = shap_values[0].values
features = X_client.columns
colors = ["green" if val > 0 else "red" for val in contributions]

for feature, val, color in zip(features, contributions, colors):
    sign = "+" if val > 0 else ""
    st.markdown(f"- <span style='color:{color}'>**{feature}**</span> : {sign}{val:.2f}", unsafe_allow_html=True)
