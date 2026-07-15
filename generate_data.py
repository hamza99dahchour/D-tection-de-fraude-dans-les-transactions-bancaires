"""
Génère un jeu de données synthétique de transactions bancaires,
avec un déséquilibre de classes réaliste (fraude ~ 2% des transactions).
"""
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_SAMPLES = 20_000
FRAUD_RATIO = 0.02


def generate_transactions(n_samples=N_SAMPLES, fraud_ratio=FRAUD_RATIO, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    def make_block(n, is_fraud):
        amount = rng.gamma(shape=2.0, scale=60 if not is_fraud else 170, size=n)
        hour = rng.normal(loc=14 if not is_fraud else 6, scale=6, size=n) % 24
        distance_km = np.abs(rng.normal(loc=6 if not is_fraud else 50, scale=22, size=n))
        nb_transactions_24h = rng.poisson(lam=3 if not is_fraud else 6, size=n)
        merchant_risk_score = rng.beta(a=2 if not is_fraud else 5, b=8 if not is_fraud else 4, size=n)
        is_foreign = rng.binomial(1, 0.05 if not is_fraud else 0.3, size=n)

        return pd.DataFrame({
            "amount": amount.round(2),
            "hour": hour.round(2),
            "distance_from_home_km": distance_km.round(2),
            "nb_transactions_24h": nb_transactions_24h,
            "merchant_risk_score": merchant_risk_score.round(3),
            "is_foreign_transaction": is_foreign,
            "is_fraud": is_fraud,
        })

    df = pd.concat([
        make_block(n_legit, is_fraud=0),
        make_block(n_fraud, is_fraud=1),
    ], ignore_index=True)

    # Bruit d'étiquetage réaliste : ~1% des labels sont inversés
    # (erreurs de déclaration, fraudes non détectées au moment des faits, etc.)
    flip_mask = rng.random(len(df)) < 0.01
    df.loc[flip_mask, "is_fraud"] = 1 - df.loc[flip_mask, "is_fraud"]

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df["transaction_id"] = [f"TX{100000 + i}" for i in range(len(df))]
    cols = ["transaction_id"] + [c for c in df.columns if c != "transaction_id"]
    return df[cols]


if __name__ == "__main__":
    df = generate_transactions()
    df.to_csv("data/transactions.csv", index=False)
    print(f"{len(df)} transactions générées -> data/transactions.csv")
    print(df["is_fraud"].value_counts(normalize=True).rename("proportion"))
