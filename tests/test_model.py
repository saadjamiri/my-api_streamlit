import pytest
import pandas as pd
import joblib
from xgboost import XGBClassifier
import os

# Charger les composants nécessaires
MODEL_PATH = "model.joblib"
COLUMNS_PATH = "columns.joblib"
DATA_PATH = "data/data_cleaned.csv"


def test_model_and_columns_exist():
    assert os.path.exists(MODEL_PATH), "Le fichier model.joblib est manquant"
    assert os.path.exists(COLUMNS_PATH), "Le fichier columns.joblib est manquant"


def test_model_can_predict_proba():
    # Charger modèle et colonnes
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)

    # Charger données brutes
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")
    
    # Encodage et alignement
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

    # Prédiction
    probas = model.predict_proba(df_encoded)
    assert probas.shape[1] == 2, "Le modèle ne retourne pas deux probabilités"
    assert (probas >= 0).all() and (probas <= 1).all(), "Les probabilités ne sont pas dans [0,1]"

