import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ----------------------------
# Préparation des données (à adapter selon ton dataset)
# ----------------------------
df = pd.read_csv("data/data_cleaned.csv")  # Remplace par ton vrai fichier
y = df["TARGET"]
X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

# Encodage des variables catégorielles
X_encoded = pd.get_dummies(X)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)

# ----------------------------
# 3. Entraînement du modèle
# ----------------------------
model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# ----------------------------
# 4. Evaluation du modèle
# ----------------------------
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC : {auc:.4f}")

# ----------------------------
# 5. Sauvegarde du modèle et des colonnes
# ----------------------------

# Sauvegarder le modèle
joblib.dump(model, "model.joblib")

# Sauvegarder les colonnes
joblib.dump(X_train.columns.tolist(), "columns.joblib")