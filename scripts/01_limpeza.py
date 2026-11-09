"""01 - Limpeza com deduplicacao em dois niveis.

Etapas:
  [1] Remove colunas classificadas como nao-coleta no dicionario
  [2] Remove linhas de cabecalho repetido
  [3] Cria evento_id (Serial Number + Test Time)
  [4] Cria defeito_id (Serial Number + Test Time + Component + Location Desc)
  [5] Deduplica no nivel de defeito
  [6] Cria houve_reparo a partir de Repair Time
  [7] Salva base_limpa.xlsx
"""
import pandas as pd
from _config import WORKING, DICIONARIO

ENTRADA = WORKING / "base_anonimizada.xlsx"
SAIDA = WORKING / "base_limpa.xlsx"

df = pd.read_excel(ENTRADA)
print(f"Entrada: {len(df)} linhas x {len(df.columns)} colunas")

# [1] Colunas nao-coleta
dic = pd.read_excel(DICIONARIO, sheet_name="Dicionário")
nao_coleta = dic.loc[
    dic["Classificação"].astype(str).str.strip().str.lower() == "não-coleta",
    "Nome da Coluna"
].dropna().tolist()
print(f"[1] Dicionario lido: {len(nao_coleta)} colunas nao-coleta")

remover = [c for c in nao_coleta if c in df.columns]
df = df.drop(columns=remover)
print(f"[1] Removidas {len(remover)} colunas sem coleta no MES")

# [2] Cabecalhos repetidos
mask_cab = pd.Series(False, index=df.index)
for col in df.columns:
    mask_cab |= df[col].astype(str).str.strip() == col
n_cab = int(mask_cab.sum())
df = df[~mask_cab].copy()
print(f"[2] Removidas {n_cab} linhas de cabecalho repetido")

# [3] Normalizacao de tipos
df["Serial Number"] = df["Serial Number"].astype(str).str.strip()
df["Test Time"] = pd.to_datetime(df["Test Time"], errors="coerce")
df["Component"] = df["Component"].fillna("").astype(str).str.strip()
df["Location Desc"] = df["Location Desc"].fillna("").astype(str).str.strip()

# [4] evento_id
df["evento_id"] = (
    df["Serial Number"] + "|" + df["Test Time"].astype(str)
)
n_eventos = df["evento_id"].nunique()
print(f"[3] evento_id criado: {n_eventos} eventos unicos")

# [5] defeito_id
df["defeito_id"] = (
    df["evento_id"] + "|" + df["Component"] + "|" + df["Location Desc"]
)
n_defeitos_antes = df["defeito_id"].nunique()
print(f"[4] defeito_id criado: {n_defeitos_antes} defeitos unicos")

# [6] Deduplicacao no nivel de defeito
antes = len(df)
df = df.drop_duplicates(subset="defeito_id", keep="first").copy()
removidas = antes - len(df)
print(f"[5] Deduplicacao: {antes} -> {len(df)} linhas ({removidas} removidas)")

# [7] houve_reparo
df["houve_reparo"] = df["Repair Time"].notna().astype(int)
n_reparo = int(df["houve_reparo"].sum())
print(f"[6] houve_reparo: {n_reparo} com reparo ({n_reparo/len(df)*100:.1f}%)")

# [8] Validacao
print(f"\n[7] Validacao final")
print(f"    Linhas: {len(df)}")
print(f"    Colunas: {len(df.columns)}")
print(f"    Nulos em State Flag: {df['State Flag'].isna().sum()}")
print(f"    Duplicatas de defeito_id: {df['defeito_id'].duplicated().sum()}")

# [9] Salvar
df.to_excel(SAIDA, index=False)
print(f"\nSalvo: {SAIDA}")
print(f"Saida: {len(df)} linhas x {len(df.columns)} colunas")

# NOTA METODOLOGICA
# A deduplicacao usa a chave composta
#   Serial Number + Test Time + Component + Location Desc
# que define um defeito unico. Registros com State Flag nulo sao
# tratados como qualquer outro: se a chave se repete, o registro e
# duplicata e e removido.
#
# Resultado: 5.235 defeitos unicos (263 duplicatas removidas).
# A versao anterior do apendice registrava 5.239, valor obtido ao
# preservar os 4 registros com State Flag nulo. A diferenca de 4
# linhas corresponde a 2 duplicatas exatas (placa_1813, mesmo
# timestamp, componente U1) e 2 registros unicos.
