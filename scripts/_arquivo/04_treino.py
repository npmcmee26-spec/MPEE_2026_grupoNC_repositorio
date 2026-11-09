"""04 - Treino do baseline."""
import json
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

with open("config/experimento.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["seed"]

treino = pd.read_parquet("data/working/treino.parquet")
val = pd.read_parquet("data/working/validacao.parquet")
teste = pd.read_parquet("data/working/teste.parquet")

IGNORAR = ["Serial Number", "Test Time", "Repair Time", "State Flag", "target"]
features = [c for c in treino.columns if c not in IGNORAR]
features = [c for c in features if pd.api.types.is_numeric_dtype(treino[c])]

X_tr, y_tr = treino[features], treino["target"]
X_va, y_va = val[features], val["target"]
X_te, y_te = teste[features], teste["target"]

scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_va = scaler.transform(X_va)
X_te = scaler.transform(X_te)

modelo = LogisticRegression(
    max_iter=cfg["modelo"]["max_iter"],
    class_weight=cfg["modelo"]["class_weight"],
    random_state=SEED,
)
modelo.fit(X_tr, y_tr)

def avaliar(nome, X, y):
    pred = modelo.predict(X)
    prob = modelo.predict_proba(X)[:, 1]
    m = {
        "acuracia": round(accuracy_score(y, pred), 4),
        "precisao": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y, prob), 4),
    }
    print(f"{nome:8s} " + " | ".join(f"{k}={v}" for k, v in m.items()))
    return m

resultado = {
    "seed": SEED,
    "n_features": len(features),
    "validacao": avaliar("Val", X_va, y_va),
    "teste": avaliar("Teste", X_te, y_te),
}

with open("resultados/metricas.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2)

print("\nMetricas salvas em resultados/metricas.json")
