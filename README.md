# Analise de Causa-Raiz em Defeitos de Placas-Mae

Pipeline de preparacao e analise de dados de defeitos provenientes do sistema
MES SMOM 2.29.02, referentes ao periodo de janeiro a marco de 2026.

## Requisitos

- Python 3.11 ou superior
- 4 GB de memoria RAM

## Instalacao

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Estrutura

| Pasta | Conteudo |
|---|---|
| config/ | Parametros centralizados (YAML) |
| data/raw/ | Dado bruto original (nao versionado) |
| data/private/ | Mapas de pseudonimizacao (nao versionado) |
| data/working/ | Bases intermediarias (nao versionado) |
| scripts/ | Pipeline executavel |
| notebooks/ | Exploracao e analise |
| docs/ | Dicionario de dados e apendice |
| outputs/ | Tabelas e figuras geradas |

## Execucao

O pipeline deve ser executado na ordem abaixo:

    python3 scripts/validar_chave.py
    python3 scripts/00_anonimizacao.py
    python3 scripts/01_limpeza.py
    python3 scripts/02_features.py
    python3 scripts/03_analise.py

## Dados

O dado bruto nao e redistribuido por conter informacoes de processo produtivo
de terceiros. A integridade do arquivo original e verificavel pelo hash
SHA-256 registrado em config/experimento.yaml:

    772ea1fd6689da71ab312222fe501a9b2e8518acd7be16fc73091d94ade01b63

## Reprodutibilidade

- Semente aleatoria fixa (42)
- Parametros centralizados em config/experimento.yaml
- Versoes de bibliotecas fixadas em requirements.txt
- Hash SHA-256 do dado bruto registrado

## Licenca

Uso academico. Ver autorizacao institucional em docs/.
