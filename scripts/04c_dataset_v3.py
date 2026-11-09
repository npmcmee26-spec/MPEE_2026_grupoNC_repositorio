#!/usr/bin/env python3
"""04c_dataset_v3.py - dataset v3.

Correcao critica: component_familia_* contem apenas o PREFIXO do componente
(ex.: "C", "U", "PC"), nao o nome completo. O agrupamento passa a ser um
dicionario direto prefixo -> grupo, cobrindo os 73 prefixos do dataset.
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
    "BGA": "Soldagem",
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
    "CONNECTER PIN DIRTY": "Material e contaminacao",
    "CONNECTER PIN DIRTY ": "Material e contaminacao",
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

# ---------------- PREFIXO DO COMPONENTE -> GRUPO ----------------
# A coluna component_familia_* contem apenas o prefixo alfabetico.
PREFIXOS = {
    # passivos
    "C": "Capacitor",
    "R": "Resistor",
    "L": "Indutor",
    "F": "Fusivel",
    "CK": "Cristal",
    # semicondutores
    "U": "Circuito integrado",
    "Q": "Transistor",
    "D": "Diodo",
    "LED": "LED",
    "BIOS": "Circuito integrado",
    "CMOS": "Circuito integrado",
    "CPU": "Circuito integrado",
    "NAND": "Circuito integrado",
    # componentes montados em PCB (prefixo P)
    "P": "Componente de PCB",
    "PC": "Componente de PCB",
    "PCB": "Componente de PCB",
    "PCA": "Componente de PCB",
    "PCG": "Componente de PCB",
    "PCH": "Componente de PCB",
    "PCM": "Componente de PCB",
    "PCXB": "Componente de PCB",
    "PCXZ": "Componente de PCB",
    "PCZ": "Componente de PCB",
    "PD": "Componente de PCB",
    "PDB": "Componente de PCB",
    "PHZ": "Componente de PCB",
    "PJPDC": "Componente de PCB",
    "PL": "Componente de PCB",
    "PLA": "Componente de PCB",
    "PLM": "Componente de PCB",
    "PPT": "Componente de PCB",
    "PQ": "Componente de PCB",
    "PQB": "Componente de PCB",
    "PQM": "Componente de PCB",
    "PQS": "Componente de PCB",
    "PQZ": "Componente de PCB",
    "PR": "Componente de PCB",
    "PRB": "Componente de PCB",
    "PRG": "Componente de PCB",
    "PRM": "Componente de PCB",
    "PRZ": "Componente de PCB",
    "PT": "Componente de PCB",
    "PU": "Componente de PCB",
    "PUA": "Componente de PCB",
    "PUB": "Componente de PCB",
    "PUG": "Componente de PCB",
    "PUM": "Componente de PCB",
    "PUS": "Componente de PCB",
    "PUZ": "Componente de PCB",
    "PWRSW": "Switch",
    # conectores
    "J": "Conector",
    "CN": "Conector",
    "CC": "Conector",
    "HDMI": "Conector",
    "USB": "Conector",
    "LAN": "Conector",
    "RJ": "Conector",
    "DCIN": "Conector",
    "XF": "Conector",
    "X": "Conector",
    "BT": "Conector",
    # solda e metal
    "EL": "Solda e metal",
    "ED": "Solda e metal",
    "EC": "Solda e metal",
    "ESD": "Solda e metal",
    # acessorios e ferragens
    "EU": "Acessorio",
    "AD": "Acessorio",
    "H": "Acessorio",
    "HS": "Acessorio",
    "BRACKET": "Acessorio",
    "BRACKT": "Acessorio",
    "GASKET": "Acessorio",
    "MYLAR": "Acessorio",
    "RUBBER": "Acessorio",
    "SHIELD": "Acessorio",
    "SHDTOP": "Acessorio",
    "ETIQUETA": "Acessorio",
    "COLA": "Acessorio",
    "COBRE": "Acessorio",
    "NTF": "Acessorio",
    "SN": "Sensor",
    "YC": "Cabo/Yoke",
    # modulos e perifericos
    "KMCOM": "Modulo",
    "WLAN": "Modulo wireless",
    "SPK": "Alto-falante",
    "FAN": "Cooler",
    "CAM": "Camera",
    "LCD": "Display",
    "DIMM": "Modulo memoria",
    "SSD": "Armazenamento",
    "HDD": "Armazenamento",
    "CARD": "Armazenamento",
    "OPTION": "Modulo",
    "SW": "Switch",
    "MB": "Placa-mae",
    "NUT": "Acessorio",
    "TP": "Ponto de teste",
    "TSL": "Ponto de teste",
}
# valores fora do dicionario caem no generico abaixo
GRUPO_FALLBACK = "Outros"

GRUPOS_ONEHOT = {
    "Test Line": "Test Line",
    "Test Station": "Test Station",
    "Test Group": "Test Group",
    "Model Name": "Model Name",
    "Duty Type": "Duty Type",
    "component_familia": "componente_prefixo",
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




# fallback: primeira letra indica a familia quando o prefixo nao esta no dicionario
_PRIMEIRA_LETRA = {
    "C": "Capacitor",
    "R": "Resistor",
    "L": "Indutor",
    "F": "Fusivel",
    "U": "Circuito integrado",
    "Q": "Transistor",
    "D": "Diodo",
    "J": "Conector",
    "P": "Componente de PCB",
    "E": "Solda e metal",
    "H": "Acessorio",
    "X": "Conector",
    "S": "Sensor",
    "K": "Modulo",
    "B": "Bateria",
    "Y": "Cabo/Yoke",
    "W": "Modulo wireless",
    "A": "Acessorio",
    "T": "Ponto de teste",
    "M": "Placa-mae",
}


def _grupo_de(prefixo):
    """Classifica o prefixo do componente em um grupo.

    Ordem: (1) prefixo exato no dicionario; (2) prefixo mais longo conhecido
    contido no nome; (3) primeira letra do prefixo; (4) fallback generico.
    """
    if not isinstance(prefixo, str) or not prefixo.strip():
        return GRUPO_FALLBACK
    p = prefixo.strip().upper()

    # 1) exato
    if p in PREFIXOS:
        return PREFIXOS[p]

    # 2) o nome comeca com um prefixo conhecido (o mais longo tem prioridade)
    candidatos = sorted(
        [k for k in PREFIXOS if p.startswith(k)],
        key=len,
        reverse=True,
    )
    for k in candidatos:
        return PREFIXOS[k]

    # 3) primeira letra
    if p[0] in _PRIMEIRA_LETRA:
        return _PRIMEIRA_LETRA[p[0]]

    return GRUPO_FALLBACK


def main():
    print("=" * 70)
    print("04c_dataset_v3.py - componente por prefixo (correcao)")
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
    for fam, n in df["familia"].value_counts().items():
        print("      %-30s %5d (%5.1f%%)" % (fam, n, n / len(df) * 100))

    print()
    print("[4] Revertendo one-hot ...")
    for prefixo, nome in GRUPOS_ONEHOT.items():
        df = reverter_onehot(df, prefixo, nome)

    print()
    print("[5] Mapeando prefixo do componente -> grupo ...")
    df["componente_grupo"] = df["componente_prefixo"].map(lambda x: _grupo_de(x))
    nao_map = df.loc[df["componente_grupo"] == GRUPO_FALLBACK, "componente_prefixo"].value_counts()
    print("    %d prefixos -> %d grupos" % (df["componente_prefixo"].nunique(),
                                            df["componente_grupo"].nunique()))
    if len(nao_map):
        print("    Prefixos no fallback (%d distintos):" % len(nao_map))
        for p, n in nao_map.head(10).items():
            print("      %-10s %d" % (p, n))
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
        list(PREFIXOS.items()), columns=["prefixo_componente", "grupo"]
    ).sort_values(["grupo", "prefixo_componente"]).to_csv(
        SAIDA / "mapeamento_prefixos_componente.csv", index=False)

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
    print("    mapeamento_prefixos_componente.csv")
    print("    dataset_resumo.txt")
    print()
    print("Fim: %s" % datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
