"""Diagnostico da limpeza: compara antes e depois."""
import pandas as pd
from _config import WORKING

base = pd.read_excel(WORKING / "base_anonimizada.xlsx")
print(f"base_anonimizada: {base.shape[0]} linhas x {base.shape[1]} colunas")

limpa = pd.read_excel(WORKING / "base_limpa.xlsx")
print(f"base_limpa:       {limpa.shape[0]} linhas x {limpa.shape[1]} colunas")

print(f"\nLinhas removidas: {base.shape[0] - limpa.shape[0]}")
print(f"Colunas removidas: {base.shape[1] - limpa.shape[1]}")
