"""03 - Divisao sem vazamento (itens 25 a 27)."""
import pandas as pd
import yaml

with open("config/experimento.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PROP_TREINO = cfg["split"]["prop_treino"]
PROP_VAL = cfg["split"]["prop_validacao"]

df = pd.read_parquet("data/working/features.parquet")
df["Test Time"] = pd.to_datetime(df["Test Time"])
df = df.sort_values("Test Time").reset_index(drop=True)

primeiro = (
    df.groupby("Serial Number")["Test Time"].min()
    .sort_values().reset_index()
)
n = len(primeiro)
corte_treino = int(n * PROP_TREINO)
corte_val = int(n * (PROP_TREINO + PROP_VAL))

placas_treino = set(primeiro.loc[:corte_treino - 1, "Serial Number"])
placas_val = set(primeiro.loc[corte_treino:corte_val - 1, "Serial Number"])
placas_teste = set(primeiro.loc[corte_val:, "Serial Number"])

treino = df[df["Serial Number"].isin(placas_treino)]
val = df[df["Serial Number"].isin(placas_val)]
teste = df[df["Serial Number"].isin(placas_teste)]

assert not (set(treino["Serial Number"]) & set(teste["Serial Number"])), "VAZAMENTO!"
assert not (set(treino["Serial Number"]) & set(val["Serial Number"])), "VAZAMENTO!"

print(f"Treino : {len(treino):5d} linhas | {len(placas_treino):4d} placas")
print(f"Val    : {len(val):5d} linhas | {len(placas_val):4d} placas")
print(f"Teste  : {len(teste):5d} linhas | {len(placas_teste):4d} placas")
print("Sem sobreposicao de placas entre os conjuntos.")

treino.to_parquet("data/working/treino.parquet", index=False)
val.to_parquet("data/working/validacao.parquet", index=False)
teste.to_parquet("data/working/teste.parquet", index=False)
