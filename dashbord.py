import streamlit as st
import pandas as pd
import numpy as np
import joblib

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