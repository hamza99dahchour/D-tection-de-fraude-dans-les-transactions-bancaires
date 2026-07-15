"""
Détection de fraude dans les transactions bancaires.

Pipeline simple :
1. Chargement des données (générées via data/generate_data.py)
2. Prétraitement (mise à l'échelle des variables numériques)
3. Entraînement de deux modèles : Régression Logistique et Random Forest
4. Évaluation adaptée aux données déséquilibrées (precision, recall, F1, ROC-AUC)
5. Sauvegarde des courbes ROC et de la matrice de confusion dans outputs/
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/transactions.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' introuvable. Lancez d'abord : python data/generate_data.py"
        )
    return pd.read_csv(path)


def prepare_data(df):
    features = [
        "amount",
        "hour",
        "distance_from_home_km",
        "nb_transactions_24h",
        "merchant_risk_score",
        "is_foreign_transaction",
    ]
    X = df[features]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, features


def train_models(X_train, y_train):
    models = {
        "Régression Logistique": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_models(models, X_test, y_test):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, digits=3, target_names=["Légitime", "Fraude"]))
        print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.3f}")

        RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=ax_roc)

        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4.5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["Légitime", "Fraude"], ax=ax_cm, cmap="Oranges"
        )
        ax_cm.set_title(f"Matrice de confusion — {name}")
        fig_cm.tight_layout()
        fig_cm.savefig(f"{OUTPUT_DIR}/confusion_matrix_{name.replace(' ', '_')}.png", dpi=150)
        plt.close(fig_cm)

    ax_roc.set_title("Courbes ROC")
    fig_roc.tight_layout()
    fig_roc.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=150)
    plt.close(fig_roc)
    print(f"\nGraphiques enregistrés dans {OUTPUT_DIR}/")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test, features = prepare_data(df)
    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)


if __name__ == "__main__":
    main()
