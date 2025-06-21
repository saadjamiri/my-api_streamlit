import streamlit as st
import shap
import pandas as pd
import numpy as np
import plotly.graph_objects as go 
import joblib
import matplotlib.pyplot as plt

# ----------------------------
# Charger le modèle et les colonnes
# ----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.joblib")
        columns = joblib.load("columns.joblib")
        return model, columns
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        raise

model, model_columns = load_model()

# ----------------------------
# Fonction : Cadrant (compteur visuel)
# ----------------------------
def plot_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=score,
        mode="gauge+number",
        title={'text': "Probabilité de défaut", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.3], 'color': 'green'},
                {'range': [0.3, 0.7], 'color': 'yellow'},
                {'range': [0.7, 1], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }}))

    fig.update_layout(font={'color': "black", 'family': "Arial"},
                      paper_bgcolor='rgba(255,255,255,1)',
                      plot_bgcolor='rgba(255,255,255,1)',
                      margin=dict(l=20, r=20, t=80, b=20))
    
    return fig


# ----------------------------
# Interface utilisateur
# ----------------------------
st.title(" Dashboard de Prédiction")
st.write("Entrez un identifiant client pour estimer la probabilité de défaut.")

# Charger les données clients
df_clients = pd.read_csv("data/data_cleaned.csv")
client_ids = df_clients["SK_ID_CURR"].tolist()
client_id = st.selectbox("Sélectionnez un client :", client_ids)

# ----------------------------
# Préparer les données du client
# ----------------------------
client_data = df_clients[df_clients["SK_ID_CURR"] == client_id]
X_client = client_data.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")

# Affichage des features du client
st.subheader(" Informations du client sélectionné")
st.dataframe(client_data.drop(columns=["TARGET"], errors="ignore"))

# Encodage
X_client_encoded = pd.get_dummies(X_client)

# Aligner avec les colonnes du modèle
X_client_encoded = X_client_encoded.reindex(columns=model_columns, fill_value=0)

# Optionnel : afficher les features utilisées
with st.expander(" Voir les features utilisées pour la prédiction"):
    st.dataframe(X_client_encoded.T.rename(columns={0: "Valeur"}))

# ----------------------------
# Prédiction
# ----------------------------
proba = model.predict_proba(X_client_encoded)[0, 1]
st.subheader(" Résultat de la prédiction :")
st.metric(label="Probabilité de défaut", value=f"{proba:.2%}")
fig = plot_gauge_chart(proba)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Interprétation locale avec SHAP
# ----------------------------
with st.expander(" Interprétation locale des features (SHAP)"):
    try:
        # Créer l'explainer SHAP
        explainer = shap.Explainer(model)

        # Calculer les valeurs SHAP du client
        shap_values = explainer(X_client_encoded)

        # Afficher le graphe waterfall
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.error(f"Erreur lors de l'explication SHAP : {e}")
