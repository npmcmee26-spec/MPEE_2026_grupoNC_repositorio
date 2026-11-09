"""Valida a chave: remove cabecalhos repetidos e checa duplicidade real."""
import hashlib
import pandas as pd
from _config import RAW, ABA, HASH_ESPERADO

# [0] Verificacao de integridade do arquivo bruto
hash_real = hashlib.sha256(RAW.read_bytes()).hexdigest()
assert hash_real == HASH_ESPERADO, (
    f"Hash divergente!\nEsperado: {HASH_ESPERADO}\nObtido:   {hash_real}"
)
print(f"[OK] Hash SHA-256 confere: {hash_real[:16]}...\n")

df = pd.read_excel(RAW, sheet_name=ABA)
print(f"Linhas brutas: {len(df)}")

mask_cab = pd.Series(False, index=df.index)
for col in df.columns:
    mask_cab |= df[col].astype(str).str.strip() == col
n_cab = int(mask_cab.sum())
df = df[~mask_cab].copy()
print(f"[0] Cabecalhos repetidos removidos: {n_cab}")
print(f"    Linhas validas: {len(df)}\n")

df["Serial Number"] = df["Serial Number"].astype(str).str.strip()
df["Test Time"] = pd.to_datetime(df["Test Time"], errors="coerce")

print(f"Test Time invalidos apos limpeza: {df['Test Time'].isna().sum()}")
print(f"Seriais unicos: {df['Serial Number'].nunique()}")
print(f"Linhas de placa retestada: {len(df) - df['Serial Number'].nunique()}\n")

CHAVE_EVENTO = ["Serial Number", "Test Time"]
dup_evento = int(df.duplicated(subset=CHAVE_EVENTO).sum())
print(f"[1] Duplicatas na chave de evento: {dup_evento}")
print("    -> Esperado > 0: um evento pode ter varios defeitos.\n")

antes = len(df)
