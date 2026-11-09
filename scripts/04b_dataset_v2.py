#!/usr/bin/env python3
"""04b_dataset_v2.py - dataset v2: familias corrigidas + componente por prefixo.

Correcoes em relacao ao 04_dataset.py:
  1. Mapeia "BGA" -> Soldagem e "FALTA DE MYLAR 5" (com espaco) -> Filme
  2. component_familia: 260 nomes brutos -> ~20 grupos por prefixo (IPC-7351)
  3. Salva tambem Serial Number e Test Time nos tres parquets
"""

from pathlib import Path
from datetime import datetime
import re
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "data" / "working" / "features.parquet"
SAIDA = RAIZ / "data" / "working"

# ---------------- familias de causa raiz ----------------
FAMILIAS = {
    # Posicionamento
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
    # Soldagem
    "NON-WETTING": "Soldagem",
    "SOLDER SHORT": "Soldagem",
    "INSUFFICIENT SOLDER": "Soldagem",
    "EXCESSIVE SOLDER": "Soldagem",
    "SOLDER BALL": "Soldagem",
    "SOLDER LACK": "Soldagem",
    "HEAD-IN-PILLOW": "Soldagem",
    "SHORT COMPONENT": "Soldagem",
    "BGA": "Soldagem",
    # Integridade
    "COMPONENT DAMAGE": "Integridade componente",
    "COMPONENT FAIL": "Integridade componente",
    "COMPONENT BURN": "Integridade componente",
    "ELECTRICAL FAILURE": "Integridade componente",
    "IMPEDANCE FAIL": "Integridade componente",
    "DAMAGE": "Integridade componente",
    # Material
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
    "CONNECTER PIN DIRTY": "Material e contaminacao",
    "CONNECTER PIN DIRTY ": "Material e contaminacao",
    # Filme
    "MYLAR MISSING": "Filme e acessorios",
    "MYLAR SHIFT": "Filme e acessorios",
    "MYLAR DANIFICADO": "Filme e acessorios",
    "MYLAR FALTANDO": "Filme e acessorios",
    "FALTA DE MYLAR5": "Filme e acessorios",
    "FALTA DE MYLAR 5": "Filme e acessorios",
    "LOST BRACKET": "Filme e acessorios",
    "LOST SHIELD": "Filme e acessorios",
    "LOST LABEL": "Filme e acessorios",
    "LOST PCB": "Filme e acessorios",
    "MISSING RUBBER": "Filme e acessorios",
    # Processo
    "WORKMANSHIP": "Processo e trabalho",
    "WORKMANSCRAP": "Processo e trabalho",
    "REPAIR FAILURE": "Processo e trabalho",
    "TESTE DE ENGENHARIA": "Processo e trabalho",
    "RECORDING PROBLEMS": "Processo e trabalho",
    "BIOS INCORRECT RECORDING": "Processo e trabalho",
    "CORRUPTED BIOS": "Processo e trabalho",
    "RECORD BIOS": "Processo e trabalho",
    "CMOS RESET OK": "Processo e trabalho",
    "UNDER ANALYSIS": "Processo e trabalho",
}

# ---------------- componente -> grupo por prefixo ----------------
# Regras aplicadas em ordem. O primeiro padrao que casar define o grupo.
REGRAS_COMPONENTE = [
    # acessorios e pecas nao-eletronicas
    (r"^SHIELD", "Acessorio"),
    (r"^MYLAR", "Acessorio"),
    (r"^GASKET", "Acessorio"),
    (r"^RUBBER", "Acessorio"),
    (r"^BRACK", "Acessorio"),
    (r"^ETIQUETA", "Acessorio"),
    (r"^COLA$", "Acessorio"),
    (r"^COBRE$", "Acessorio"),
    (r"^SHDTOP", "Acessorio"),
    (r"^NUT", "Acessorio"),
    # placas
    (r"^PCB", "Placa"),
    (r"^PCG", "Placa"),
    (r"^PCA", "Placa"),
    (r"^PCXB", "Placa"),
    # conectores (checar antes de C generico)
    (r"^CN", "Conector"),
    (r"^CC", "Conector"),
    (r"^J[A-Z0-9]", "Conector"),
    (r"^HDMI", "Conector"),
    (r"^USB", "Conector"),
    (r"^LAN", "Conector"),
    (r"^RJ", "Conector"),
    (r"^DCIN", "Conector"),
    # pontos de teste
    (r"^TP", "Ponto de teste"),
    (r"^TSL", "Ponto de teste"),
    # semicondutores
    (r"^U[A-Z0-9]", "Circuito integrado"),
    (r"^Q[A-Z0-9]", "Transistor"),
    (r"^D[A-Z0-9]", "Diodo"),
    (r"^LED", "LED"),
    (r"^BIOS", "Circuito integrado"),
    (r"^CMOS", "Circuito integrado"),
    (r"^CPU", "Circuito integrado"),
    (r"^RESET", "Circuito integrado"),
    (r"^PWRSW", "Switch"),
    (r"^SW[A-Z0-9]", "Switch"),
    # passivos
    (r"^C[A-Z0-9]", "Capacitor"),
    (r"^R[A-Z0-9]", "Resistor"),
    (r"^L[A-Z0-9]", "Indutor"),
    (r"^F[A-Z0-9]", "Fusivel"),
    (r"^CK", "Cristal"),
    (r"^SPR", "Acessorio"),
    # modulos e perifericos
    (r"^BAT", "Bateria"),
    (r"^BATT", "Bateria"),
    (r"^PBATT", "Bateria"),
    (r"^SPK", "Alto-falante"),
    (r"^FAN", "Cooler"),
    (r"^CAM", "Camera"),
    (r"^LCD", "Display"),
    (r"^DIMM", "Modulo memoria"),
    (r"^SSD", "Armazenamento"),
    (r"^HDD", "Armazenamento"),
    (r"^CARD", "Armazenamento"),
    (r"^MB$", "Placa"),
    (r"^OPTION", "Modulo"),
]


def grupo_componente(nome):
    """Mapeia nome bruto do componente para grupo por prefixo."""
    if not isinstance(nome, str) or nome.strip() == "":
        return "Outros"
    nome = nome.strip().upper()
    for padrao, grupo in REGRAS_COMPONENTE:
        if re.match(padrao, nome):
            return grupo
    return "Outros"


GRUPOS_ONEHOT = {
    "Test Line": "Test Line",
    "Test Station": "Test Station",
    "Test Group": "Test Group",
    "Model Name": "Model Name",
    "Duty Type": "Duty Type",
    "component_familia": "componente_bruto",
}

COLUNAS_VAZAMENTO = [
    "Reason Code", "Repair Time", "Repairer", "houve_reparo",
    "defeito_seq", "defeito_id", "target",
    "MO Error Code", "Error Desc", "Part No", "Component",
    "Shipping SN", "Cust Part No", "State Flag",
    "Reason DescTester", "Data Code", "Location DescTester",
    "evento_id", "loc_n_posicoes",
]


def reverter_onehot(df, prefixo, nome_saida):
    cols = [c for c in df.columns if c.startswith(prefixo + "_")]
    if not cols:
        print("  [AVISO] prefixo %s_ nao encontrado" % prefixo)
        df[nome_saida] = "desconhecido"
        return df
    bloco = df[cols].values
    idx = bloco.argmax(axis=1)
    rotulos = np.array([c.replace(prefixo + "_", "", 1) for c in cols])
    todos_zero = bloco.sum(axis=1) == 0
    valores = rotulos[idx].astype(object)
    valores[todos_zero] = "desconhecido"
    df[nome_saida] = valores
    df = df.drop(columns=cols)
    print("  %-22s <- %3d cols -> %3d valores" % (nome_saida, len(cols), pd.Series(valores).nunique()))
    return df


def extrair_reason_desc(row, cols_rd):
    valores = row[cols_rd].values.astype(float)
    if valores.sum() == 0:
        return "UNDER ANALYSIS"
    return cols_rd[int(valores.argmax())].replace("Reason Desc_", "", 1)


def main():
    print("=" * 70)
    print("04b_dataset_v2.py - dataset v2")
    print("=" * 70)
    print("Inicio: %s" % datetime.now().isoformat(timespec="seconds"))
    print()

    print("[1] Carregando features.parquet ...")
    df = pd.read_parquet(ENTRADA)
    meta = pd.read_parquet(ENTRADA, columns=["Serial Number", "Test Time"])
    print("    Shape: %s" % (df.shape,))

    print()
    print("[2] Reconstruindo causa_desc ...")
    cols_rd = [c for c in df.columns if c.startswith("Reason Desc_")]
    df["causa_desc"] = df[cols_rd].apply(extrair_reason_desc, axis=1, cols_rd=cols_rd)
    df = df.drop(columns=cols_rd)
    print("    61 descricoes reconstruidas")

    print()
    print("[3] Mapeando causa_desc -> familia ...")
    df["familia"] = df["causa_desc"].map(FAMILIAS)
    nao = df.loc[df["familia"].isna(), "causa_desc"].unique()
    if len(nao):
        print("    [ATENCAO] %d sem mapeamento: %s" % (len(nao), list(nao)))
        df["familia"] = df["familia"].fillna("Processo e trabalho")
    else:
        print("    Todas mapeadas. Sem classe residual.")

    print()
    print("    Distribuicao das familias:")
    for fam, n in df["familia"].value_counts().items():
        print("      %-30s %5d (%5.1f%%)" % (fam, n, n / len(df) * 100))

    print()
    print("[4] Revertendo one-hot ...")
    for prefixo, nome in GRUPOS_ONEHOT.items():
        df = reverter_onehot(df, prefixo, nome)

    print()
    print("[5] Agrupando componente bruto -> grupo por prefixo ...")
    df["componente_grupo"] = df["componente_bruto"].map(grupo_componente)
    df = df.drop(columns=["componente_bruto"])
    print("    %d nomes brutos -> %d grupos" % (260, df["componente_grupo"].nunique()))
    print()
    for g, n in df["componente_grupo"].value_counts().items():
        print("      %-24s %5d (%5.1f%%)" % (g, n, n / len(df) * 100))

    print()
    print("[6] Features temporais ...")
    tt = pd.to_datetime(meta["Test Time"])
    df["hora_sin"] = np.sin(2 * np.pi * tt.dt.hour / 24)
    df["hora_cos"] = np.cos(2 * np.pi * tt.dt.hour / 24)
    df["dia_semana"] = tt.dt.day_of_week
    df["is_fim_semana"] = (tt.dt.day_of_week >= 5).astype(int)
    df["turno"] = pd.cut(
        tt.dt.hour, bins=[-1, 5, 13, 21, 23],
        labels=["madrugada", "manha", "tarde", "noite"],
    ).astype(str).replace("nan", "noite")
    print("    + hora_sin, hora_cos, dia_semana, is_fim_semana, turno")

    print()
    print("[7] Removendo vazamento ...")
    presentes = [c for c in COLUNAS_VAZAMENTO if c in df.columns]
    df = df.drop(columns=presentes)
    print("    %d colunas removidas" % len(presentes))

    CATEGORICAS = ["Test Line", "Test Station", "Test Group", "Model Name",
                   "Duty Type", "componente_grupo", "turno"]
    NUMERICAS = ["hora_sin", "hora_cos", "dia_semana", "is_fim_semana"]

    df["Serial Number"] = meta["Serial Number"].values
    df["Test Time"] = meta["Test Time"].values

    final = df[CATEGORICAS + NUMERICAS + ["familia", "Serial Number", "Test Time"]].copy()
    final = final.rename(columns={"familia": "target"})
    print()
    print("[8] Dataset final: %s" % (final.shape,))
    for c in CATEGORICAS:
        print("    %-22s %3d categorias" % (c, final[c].nunique()))

    print()
    print("[9] Split temporal por placa (70/15/15) ...")
    final = final.sort_values("Test Time").reset_index(drop=True)
    placas = final["Serial Number"].unique()
    n = len(placas)
    c1, c2 = int(0.70 * n), int(0.85 * n)
    p_tr, p_va, p_te = placas[:c1], placas[c1:c2], placas[c2:]

    treino = final[final["Serial Number"].isin(p_tr)].reset_index(drop=True)
    val = final[final["Serial Number"].isin(p_va)].reset_index(drop=True)
    teste = final[final["Serial Number"].isin(p_te)].reset_index(drop=True)

    assert not (set(p_tr) & set(p_va)), "vazamento treino/val"
    assert not (set(p_tr) & set(p_te)), "vazamento treino/teste"
    assert not (set(p_va) & set(p_te)), "vazamento val/teste"
    print("    OK - sem vazamento entre conjuntos")

    print()
    print("[10] Distribuicao do target ...")
    for nome, parte in [("Treino", treino), ("Val", val), ("Teste", teste)]:
        print()
        print("    %s: %d eventos, %d placas" % (nome, len(parte), parte["Serial Number"].nunique()))
        for fam, cnt in parte["target"].value_counts().items():
            print("      %-30s %5d (%5.1f%%)" % (fam, cnt, cnt / len(parte) * 100))

    print()
    print("[11] Salvando ...")
    treino.to_parquet(SAIDA / "treino.parquet", index=False)
    val.to_parquet(SAIDA / "val.parquet", index=False)
    teste.to_parquet(SAIDA / "teste.parquet", index=False)
    pd.DataFrame(
        list(FAMILIAS.items()), columns=["causa_desc_original", "familia"]
    ).sort_values("familia").to_csv(SAIDA / "mapeamento_familias.csv", index=False)
    pd.DataFrame(
        [(g,) for g in sorted(final["componente_grupo"].unique())],
        columns=["grupo_componente"],
    ).to_csv(SAIDA / "grupos_componente.csv", index=False)

    with open(SAIDA / "dataset_resumo.txt", "w") as f:
        f.write("Gerado: %s\n" % datetime.now().isoformat(timespec="seconds"))
        f.write("Semente: %d\n" % SEED)
        f.write("Fonte: %s\n\n" % ENTRADA)
        f.write("Treino: %d | Val: %d | Teste: %d\n" % (len(treino), len(val), len(teste)))
        f.write("Placas: %d | %d | %d\n\n" % (len(p_tr), len(p_va), len(p_te)))
        f.write("Categoricas: %s\n" % CATEGORICAS)
        f.write("Numericas: %s\n" % NUMERICAS)
        f.write("Target: %d classes\n" % final["target"].nunique())
        f.write("Grupos de componente: %d\n\n" % final["componente_grupo"].nunique())
        f.write("Distribuicao (treino):\n")
        f.write(treino["target"].value_counts().to_string() + "\n")

    print("    treino.parquet  %5d" % len(treino))
    print("    val.parquet     %5d" % len(val))
    print("    teste.parquet   %5d" % len(teste))
    print("    mapeamento_familias.csv")
    print("    grupos_componente.csv")
    print("    dataset_resumo.txt")
    print()
    print("Fim: %s" % datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
