#!/usr/bin/env python3
"""04_dataset.py - Construcao do dataset de classificacao de causa raiz.

Projeto: MPEE_2026 - Aprendizado Profundo
Pergunta: Dado um defeito confirmado e o componente afetado identificado,
          prever a FAMILIA de causa raiz.

Entrada : data/working/features.parquet
Saida   : data/working/{treino,val,teste}.parquet
          data/working/mapeamento_familias.csv
          data/working/dataset_resumo.txt

Decisao metodologica (Opcao 2):
  Features de processo + contexto do componente.
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "data" / "working" / "features.parquet"
SAIDA = RAIZ / "data" / "working"
SAIDA.mkdir(parents=True, exist_ok=True)

FAMILIAS = {
    "COMPONENT SHIFT": "Posicionamento componente",
    "Shifted Component": "Posicionamento componente",
    "COMPONENT MISSING": "Posicionamento componente",
    "COMPONENT FLOATING": "Posicionamento componente",
    "ROTATED COMPONENT": "Posicionamento componente",
    "UPSIDE DOWN": "Posicionamento componente",
    "WRONG POLARITY": "Posicionamento componente",
    "WRONG COMPONENT": "Posicionamento componente",
    "SKIPPED COMPONENT": "Posicionamento componente",
    "UNRECORDED COMPONENT": "Posicionamento componente",
    "EXTRA COMPONENT": "Posicionamento componente",
    "TOMBSTONE": "Posicionamento componente",
    "BILLBOARDING": "Posicionamento componente",
    "COPLANARITY FAIL": "Posicionamento componente",
    "COMPONENT WITHOUT PROJECTION": "Posicionamento componente",
    "NON-WETTING": "Soldagem",
    "SOLDER SHORT": "Soldagem",
    "INSUFFICIENT SOLDER": "Soldagem",
    "EXCESSIVE SOLDER": "Soldagem",
    "SOLDER BALL": "Soldagem",
    "SOLDER LACK": "Soldagem",
    "HEAD-IN-PILLOW": "Soldagem",
    "SHORT COMPONENT": "Soldagem",
    "COMPONENT DAMAGE": "Integridade componente",
    "COMPONENT FAIL": "Integridade componente",
    "COMPONENT BURN": "Integridade componente",
    "ELECTRICAL FAILURE": "Integridade componente",
    "IMPEDANCE FAIL": "Integridade componente",
    "DAMAGE": "Integridade componente",
    "FOREIGN MATERIAL": "Material e contaminacao",
    "MATERIAL ISSUE": "Material e contaminacao",
    "CONTAMINATED PAD": "Material e contaminacao",
    "IMPURITY": "Material e contaminacao",
    "EXCESSIVE ADHESIVE": "Material e contaminacao",
    "INSUFFICIENT ADHESIVE": "Material e contaminacao",
    "NO ADHESIVE": "Material e contaminacao",
    "Dirty": "Material e contaminacao",
    "DAMAGED PAD": "Material e contaminacao",
    "EXPOSED COPPER": "Material e contaminacao",
    "EXPOSED COPPER SUPPLIER": "Material e contaminacao",
    "CONNECTER PIN DIRTY ": "Material e contaminacao",
    "CONNECTER PIN DIRTY": "Material e contaminacao",
    "MYLAR MISSING": "Filme e acessorios",
    "MYLAR SHIFT": "Filme e acessorios",
    "MYLAR DANIFICADO": "Filme e acessorios",
    "MYLAR FALTANDO": "Filme e acessorios",
    "FALTA DE MYLAR5": "Filme e acessorios",
    "LOST BRACKET": "Filme e acessorios",
    "LOST SHIELD": "Filme e acessorios",
    "LOST LABEL": "Filme e acessorios",
    "LOST PCB": "Filme e acessorios",
    "MISSING RUBBER": "Filme e acessorios",
    "WORKMANSHIP": "Processo e trabalho",
    "WORKMANSCRAP": "Processo e trabalho",
    "REPAIR FAILURE": "Processo e trabalho",
    "TESTE DE ENGENHARIA": "Processo e trabalho",
    "RECORDING PROBLEMS": "Processo e trabalho",
    "BIOS INCORRECT RECORDING": "Processo e trabalho",
    "CORRUPTED BIOS": "Processo e trabalho",
    "RECORD BIOS": "Processo e trabalho",
    "CMOS RESET OK": "Processo e trabalho",
    "UNDER ANALYSIS": "Outros",
}

GRUPOS_ONEHOT = {
    "Test Line": "Test Line",
    "Test Station": "Test Station",
    "Test Group": "Test Group",
    "Model Name": "Model Name",
    "Duty Type": "Duty Type",
    "component_familia": "component_familia",
}

COLUNAS_VAZAMENTO = [
    "Reason Code", "Repair Time", "Repairer", "houve_reparo",
    "defeito_seq", "defeito_id", "target",
    "Serial Number", "MO Error Code", "Error Desc",
    "Part No", "Component", "Shipping SN", "Cust Part No",
    "State Flag", "Reason DescTester", "Data Code",
    "Location DescTester", "evento_id", "loc_n_posicoes",
]


def reverter_onehot(df, prefixo, nome_saida):
    cols = [c for c in df.columns if c.startswith(prefixo + "_")]
    if not cols:
        print("  [AVISO] prefixo %s_ nao encontrado" % prefixo)
        df[nome_saida] = "desconhecido"
        return df
    bloco = df[cols]
    idx = bloco.values.argmax(axis=1)
    rotulos = np.array([c.replace(prefixo + "_", "", 1) for c in cols])
    todos_zero = bloco.values.sum(axis=1) == 0
    valores = rotulos[idx]
    valores[todos_zero] = "desconhecido"
    df[nome_saida] = valores
    df = df.drop(columns=cols)
    print("  %-22s <- %3d cols -> %3d categorias" % (nome_saida, len(cols), df[nome_saida].nunique()))
    return df


def extrair_reason_desc(row, cols_rd):
    valores = row[cols_rd].values.astype(float)
    if valores.sum() == 0:
        return "UNDER ANALYSIS"
    i = valores.argmax()
    return cols_rd[i].replace("Reason Desc_", "", 1)


def main():
    print("=" * 70)
    print("04_dataset.py - classificacao de familia de causa raiz")
    print("=" * 70)
    print("Inicio: %s" % datetime.now().isoformat(timespec="seconds"))
    print()

    print("[1] Carregando features.parquet ...")
    df = pd.read_parquet(ENTRADA)
    print("    Shape original: %s" % (df.shape,))
    print()

    print("[2] Reconstruindo descricao da causa raiz ...")
    cols_rd = [c for c in df.columns if c.startswith("Reason Desc_")]
    print("    Colunas Reason Desc: %d" % len(cols_rd))
    df["causa_desc"] = df[cols_rd].apply(extrair_reason_desc, axis=1, cols_rd=cols_rd)
    df = df.drop(columns=cols_rd)
    print("    Categorias distintas: %d" % df["causa_desc"].nunique())

    print()
    print("[3] Mapeando descricoes -> familias ...")
    df["familia"] = df["causa_desc"].map(FAMILIAS)
    nao_mapeadas = df.loc[df["familia"].isna(), "causa_desc"].unique()
    if len(nao_mapeadas):
        print("    [ATENCAO] %d descricoes sem mapeamento:" % len(nao_mapeadas))
        for d in nao_mapeadas:
            print("      - %r" % d)
        df["familia"] = df["familia"].fillna("Outros")
    else:
        print("    Todas as descricoes mapeadas.")

    print()
    print("    Distribuicao das familias:")
    dist = df["familia"].value_counts()
    for fam, n in dist.items():
        print("      %-30s %5d  (%5.1f%%)" % (fam, n, n / len(df) * 100))

    print()
    print("[4] Revertendo one-hot -> categoricas ...")
    for prefixo, nome in GRUPOS_ONEHOT.items():
        df = reverter_onehot(df, prefixo, nome)

    print()
    print("[5] Derivando features temporais ...")
    tt = pd.to_datetime(df["Test Time"])
    df["hora_sin"] = np.sin(2 * np.pi * tt.dt.hour / 24)
    df["hora_cos"] = np.cos(2 * np.pi * tt.dt.hour / 24)
    df["dia_semana"] = tt.dt.dayofweek
    df["is_fim_semana"] = (tt.dt.dayofweek >= 5).astype(int)
    df["turno"] = pd.cut(
        tt.dt.hour,
        bins=[-1, 5, 13, 21, 23],
        labels=["madrugada", "manha", "tarde", "noite"],
    ).astype(str)
    df["turno"] = df["turno"].replace("nan", "noite")
    print("    + hora_sin, hora_cos, dia_semana, is_fim_semana, turno")

    print()
    print("[6] Removendo colunas de vazamento ...")
    presentes = [c for c in COLUNAS_VAZAMENTO if c in df.columns]
    df = df.drop(columns=presentes)
    print("    Removidas %d colunas" % len(presentes))

    CATEGORICAS = ["Test Line", "Test Station", "Test Group",
                   "Model Name", "Duty Type", "component_familia", "turno"]
    NUMERICAS = ["hora_sin", "hora_cos", "dia_semana", "is_fim_semana"]
    TARGET = "familia"
    GRUPO = "Serial Number"

    df_original = pd.read_parquet(ENTRADA, columns=["Serial Number", "Test Time"])
    df[GRUPO] = df_original[GRUPO].values
    df["Test Time"] = df_original["Test Time"].values

    faltando = [c for c in CATEGORICAS + NUMERICAS + [TARGET, GRUPO] if c not in df.columns]
    if faltando:
        raise KeyError("Colunas ausentes: %s" % faltando)

    final = df[CATEGORICAS + NUMERICAS + [TARGET, GRUPO, "Test Time"]].copy()
    print()
    print("[7] Dataset final: %s" % (final.shape,))
    print("    Categoricas: %s" % CATEGORICAS)
    print("    Numericas  : %s" % NUMERICAS)

    print()
    print("[8] Split temporal por placa (70/15/15) ...")
    final = final.sort_values("Test Time").reset_index(drop=True)
    placas = final[GRUPO].unique()
    n = len(placas)
    corte1, corte2 = int(0.70 * n), int(0.85 * n)
    placas_tr = placas[:corte1]
    placas_va = placas[corte1:corte2]
    placas_te = placas[corte2:]

    treino = final[final[GRUPO].isin(placas_tr)].reset_index(drop=True)
    val = final[final[GRUPO].isin(placas_va)].reset_index(drop=True)
    teste = final[final[GRUPO].isin(placas_te)].reset_index(drop=True)

    print()
    print("[9] Verificacao de vazamento entre conjuntos ...")
    s_tr, s_va, s_te = set(placas_tr), set(placas_va), set(placas_te)
    assert not (s_tr & s_va), "VAZAMENTO treino/val: %d placas" % len(s_tr & s_va)
    assert not (s_tr & s_te), "VAZAMENTO treino/teste: %d placas" % len(s_tr & s_te)
    assert not (s_va & s_te), "VAZAMENTO val/teste: %d placas" % len(s_va & s_te)
    print("    OK - nenhuma placa em dois conjuntos")

    print()
    print("[10] Distribuicao das familias por conjunto ...")
    for nome, parte in [("Treino", treino), ("Val", val), ("Teste", teste)]:
        print()
        print("    %s: %d eventos, %d placas" % (nome, len(parte), parte[GRUPO].nunique()))
        vc = parte[TARGET].value_counts()
        for fam, n_fam in vc.items():
            print("      %-30s %5d (%5.1f%%)" % (fam, n_fam, n_fam / len(parte) * 100))

    print()
    print("[11] Salvando ...")
    treino.to_parquet(SAIDA / "treino.parquet", index=False)
    val.to_parquet(SAIDA / "val.parquet", index=False)
    teste.to_parquet(SAIDA / "teste.parquet", index=False)
    pd.DataFrame(
        list(FAMILIAS.items()),
        columns=["causa_desc_original", "familia"],
    ).sort_values("familia").to_csv(SAIDA / "mapeamento_familias.csv", index=False)

    with open(SAIDA / "dataset_resumo.txt", "w") as f:
        f.write("Gerado em: %s\n" % datetime.now().isoformat(timespec="seconds"))
        f.write("Semente: %d\n" % SEED)
        f.write("Fonte: %s\n\n" % ENTRADA)
        f.write("Shape original: %s\n" % (df.shape,))
        f.write("Treino: %d | Val: %d | Teste: %d\n" % (len(treino), len(val), len(teste)))
        f.write("Placas: %d | %d | %d\n\n" % (len(placas_tr), len(placas_va), len(placas_te)))
        f.write("Categoricas: %s\n" % CATEGORICAS)
        f.write("Numericas: %s\n" % NUMERICAS)
        f.write("Target: %s (%d classes)\n" % (TARGET, final[TARGET].nunique()))
        f.write("Grupo: %s\n\n" % GRUPO)
        f.write("Distribuicao do target (treino):\n")
        f.write(treino[TARGET].value_counts().to_string() + "\n")

    print("    treino.parquet        %5d linhas" % len(treino))
    print("    val.parquet           %5d linhas" % len(val))
    print("    teste.parquet         %5d linhas" % len(teste))
    print("    mapeamento_familias.csv")
    print("    dataset_resumo.txt")
    print()
    print("Fim: %s" % datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
