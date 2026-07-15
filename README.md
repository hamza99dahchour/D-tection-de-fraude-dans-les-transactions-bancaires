# Détection de fraude dans les transactions bancaires

Projet Machine Learning simple : détecter les transactions frauduleuses dans un jeu de données déséquilibré (~2% de fraude), avec prétraitement des données et comparaison de deux modèles de classification.

## Contenu

```
.
├── data/
│   └── generate_data.py   # génère un jeu de données synthétique réaliste
├── fraud_detection.py     # entraînement + évaluation des modèles
├── requirements.txt
└── outputs/                # généré à l'exécution (matrices de confusion, courbes ROC)
```

## Approche

- **Données** : transactions synthétiques (montant, heure, distance au domicile, nombre de transactions sur 24h, score de risque du marchand, transaction à l'étranger), avec une classe minoritaire "fraude" volontairement rare pour reproduire un cas réel.
- **Prétraitement** : split train/test stratifié, mise à l'échelle des variables numériques (`StandardScaler`).
- **Modèles** : Régression Logistique et Random Forest, tous deux avec `class_weight="balanced"` pour compenser le déséquilibre de classes.
- **Évaluation** : precision / recall / F1 par classe (l'accuracy seule n'a pas de sens sur des données déséquilibrées), ROC-AUC, matrice de confusion.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# 1. Générer les données
python data/generate_data.py

# 2. Entraîner et évaluer les modèles
python fraud_detection.py
```

Les résultats (rapport de classification, ROC-AUC) s'affichent dans la console, et les graphiques (courbes ROC, matrices de confusion) sont enregistrés dans `outputs/`.

## Pistes d'amélioration

- Tester des techniques de rééchantillonnage (SMOTE, sous-échantillonnage)
- Ajouter une validation croisée et une recherche d'hyperparamètres
- Remplacer les données synthétiques par un jeu de données réel (ex. Kaggle Credit Card Fraud Detection)
