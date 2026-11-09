"""00 - Pseudonimizacao das colunas identificadoras."""
import pandas as pd
from _config import RAW, PRIVATE, WORKING, ABA

SAIDA = WORKING / "base_anonimizada.xlsx"

df = pd.read_excel(RAW, sheet_name=ABA)
print(f"Entrada: {len(df)} linhas x {len(df.columns)} colunas")

# Colunas a pseudonimizar: prefixo -> coluna original
COLUNAS = {
    "placa": "Serial Number",
    "mo": "MO",
    "part": "Key Part No",
    "op": "OP",
    "rep": "Repair Order",
    "ship": "Shipment",
    "custpart": "Customer Part No",
}

for prefixo, coluna in COLUNAS.items():
    if coluna not in df.columns:
        print(f"[AVISO] Coluna ausente: {coluna}")
        continue
    valores = df[coluna].astype(str).str.strip()
    unicos = valores.unique()
    mapa = {v: f"{prefixo}_{i:04d}" for i, v in enumerate(unicos, start=1)}
    df[coluna] = valores.map(mapa)
    pd.DataFrame({"original": list(mapa.keys()), "pseudonimo": list(mapa.values())}) \
        .to_csv(PRIVATE / f"mapa_{prefixo}.csv", index=False)
    print(f"[OK] {coluna} -> {prefixo}_XXXX ({len(mapa)} valores)")

df.to_excel(SAIDA, index=False)
print(f"\nSalvo: {SAIDA}")
print(f"Saida: {len(df)} linhas x {len(df.columns)} colunas")
