"""Diagnostico da chave composta."""
import pandas as pd
from _config import RAW, ABA

df = pd.read_excel(RAW, sheet_name=ABA)
print(f"Linhas: {len(df)}")
print(f"Colunas: {len(df.columns)}")
print(f"Seriais unicos: {df['Serial Number'].nunique()}")
print(f"Eventos unicos (Serial + Time): {df[['Serial Number', 'Test Time']].drop_duplicates().shape[0]}")
