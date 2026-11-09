"""Carrega a configuracao central. Importado por todos os scripts."""
from pathlib import Path
import yaml

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_CFG = RAIZ / "config" / "experimento.yaml"

with open(CAMINHO_CFG, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# Atalhos usados pelos scripts
SEED = CFG["reprodutibilidade"]["seed"]
RAW = RAIZ / CFG["caminhos"]["raw"]
PRIVATE = RAIZ / CFG["caminhos"]["private"]
WORKING = RAIZ / CFG["caminhos"]["working"]
OUTPUTS = RAIZ / CFG["caminhos"]["outputs"]
DICIONARIO = RAIZ / CFG["caminhos"]["dicionario"]
ABA = CFG["dados"]["aba"]
HASH_ESPERADO = CFG["dados"]["hash_sha256"]

# Garante que as pastas de saida existem
for pasta in (PRIVATE, WORKING, OUTPUTS):
    pasta.mkdir(parents=True, exist_ok=True)
