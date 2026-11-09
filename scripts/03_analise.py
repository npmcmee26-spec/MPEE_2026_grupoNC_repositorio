"""03 - Analise de causa-raiz dos defeitos de placa-mae.

Executa quatro analises sobre a base limpa:
  [1] Estatistica descritiva das variaveis de defeito
  [2] Teste qui-quadrado e V de Cramer para associacao com o estagio do defeito
  [3] Regras de associacao (apriori) entre atributos categoricos
  [4] Detector de desvios de processo por intervalo de confianca de Wilson

Saidas:
  outputs/tabelas/01_descritiva.csv
  outputs/tabelas/02_qui_quadrado.csv
  outputs/tabelas/03_regras_associacao.csv
  outputs/tabelas/04_desvios_processo.csv
  outputs/figuras/02_v_cramer.png
  outputs/figuras/03_regras.png
  outputs/figuras/04_desvios.png
"""
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from _config import WORKING, OUTPUTS, CFG

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- configuracao
ALPHA = CFG["analise"]["alpha"]
MIN_SUPORTE = CFG["analise"]["min_suporte"]
MIN_CONFIANCA = CFG["analise"]["min_confianca"]
MIN_LIFT = CFG["analise"]["min_lift"]
TOP_N_REGRAS = CFG["analise"]["top_n_regras"]

DIR_TABELAS = OUTPUTS / "tabelas"
DIR_FIGURAS = OUTPUTS / "figuras"
DIR_TABELAS.mkdir(parents=True, exist_ok=True)
DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


def secao(titulo):
    print()
    print("=" * 72)
    print(titulo)
    print("=" * 72)


def salvar_figura(nome):
    caminho = DIR_FIGURAS / f"{nome}.png"
    plt.savefig(caminho)
    plt.close()
    print(f"    figura salva: {caminho}")


# ------------------------------------------------------------------ carregamento
ARQUIVO = WORKING / "base_limpa.xlsx"
if not ARQUIVO.exists():
    print(f"[ERRO] Base nao encontrada: {ARQUIVO}")
    print("Execute o pipeline antes: 00_anonimizacao.py, 01_limpeza.py, 02_features.py")
    sys.exit(1)

df = pd.read_excel(ARQUIVO)
print(f"Base carregada: {len(df)} linhas x {len(df.columns)} colunas")

df["estagio_defeito"] = (
    pd.to_numeric(df["State Flag"], errors="coerce").fillna(1).astype(int)
)

MAPA_PREFIXOS = [
    ("BRACKET", "MECANICO"), ("BRACKT", "MECANICO"), ("MYLAR", "MECANICO"),
    ("GASKET", "MECANICO"), ("ETIQUETA", "MECANICO"), ("NUT", "MECANICO"),
    ("HS", "MECANICO"), ("BATT", "MECANICO"), ("BAT", "MECANICO"),
    ("BT", "MECANICO"), ("H", "FURO_MONTAGEM"),
    ("PCB", "PLACA"), ("MB", "PLACA"), ("SN", "SERIAL"),
    ("HDMI", "CONECTOR"), ("OPTION", "CONECTOR"), ("WLAN", "CONECTOR"),
    ("WL", "CONECTOR"), ("KMCOM", "CONECTOR"), ("KBBL", "CONECTOR"),
    ("AUD", "CONECTOR"), ("J", "CONECTOR"), ("P", "CONECTOR"),
    ("BIOS", "CI"), ("EU", "CI"), ("AD", "CI"), ("U", "CI"), ("IC", "CI"),
    ("EC", "CAPACITOR"), ("C", "CAPACITOR"),
    ("EL", "INDUTOR"), ("L", "INDUTOR"),
    ("ED", "DIODO"), ("D", "DIODO"), ("LED", "LED"),
    ("R", "RESISTOR"), ("TR", "RESISTOR"),
    ("Q", "TRANSISTOR"), ("TQ", "TRANSISTOR"),
    ("Y", "CRISTAL"), ("X", "CRISTAL"),
    ("F", "FUSIVEL"),
    ("TP", "PONTO_TESTE"), ("TSL", "PONTO_TESTE"),
    ("SW", "CHAVE"), ("S", "CHAVE"),
    ("MH", "FURO_MONTAGEM"),
    ("ER", "OUTROS_TIPO"), ("AR", "OUTROS_TIPO"), ("EQ", "OUTROS_TIPO"),
    ("HE", "OUTROS_TIPO"), ("NTF", "OUTROS_TIPO"),
    ("T", "OUTROS_TIPO"),
]


def tipo_componente(s):
    s = str(s).strip().upper()
    if s in ("", "NAN", "NONE"):
        return "SEM_LOCAL"
    for prefixo, tipo in MAPA_PREFIXOS:
        if s.startswith(prefixo):
            return tipo
    return "OUTROS_TIPO"


df["component_tipo"] = df["Component"].fillna("").astype(str).str.strip().apply(tipo_componente)
df["loc_tipo"] = df["Location Desc"].fillna("").astype(str).str.strip().apply(tipo_componente)

# ------------------------------------------------------- [1] descritiva
secao("[1] ESTATISTICA DESCRITIVA")

cont = df["estagio_defeito"].value_counts().sort_index()
desc = pd.DataFrame({
    "estagio": cont.index,
    "defeitos": cont.values,
    "percentual": (cont.values / len(df) * 100).round(2),
})
print("\nDistribuicao de estagio_defeito:")
print(desc.to_string(index=False))

cont_rep = df["houve_reparo"].value_counts().sort_index()
print("\nDistribuicao de houve_reparo:")
for v, q in cont_rep.items():
    print(f"    {v}: {q} ({q/len(df)*100:.1f}%)")

n_def = df.groupby("evento_id").size().value_counts().sort_index()
print("\nDefeitos por evento:")
for v, q in n_def.items():
    print(f"    {v} defeito(s): {q} eventos ({q/df['evento_id'].nunique()*100:.1f}%)")

print("\nTop 10 tipos de componente:")
for v, q in df["component_tipo"].value_counts().head(10).items():
    print(f"    {v:15s}: {q:5d} ({q/len(df)*100:5.2f}%)")

print("\nTop 10 causas (Reason Desc):")
for v, q in df["Reason Desc"].value_counts().head(10).items():
    print(f"    {str(v)[:40]:40s}: {q:5d} ({q/len(df)*100:5.2f}%)")

desc.to_csv(DIR_TABELAS / "01_descritiva.csv", index=False)
print(f"\n    tabela salva: {DIR_TABELAS / '01_descritiva.csv'}")

# ------------------------------------------- [2] qui-quadrado e V de Cramer
secao("[2] QUI-QUADRADO E V DE CRAMER")


def v_cramer(tabela):
    chi2, p, gl, _ = chi2_contingency(tabela)
    n = tabela.values.sum()
    k = min(tabela.shape) - 1
    v = np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else 0.0
    return {"chi2": chi2, "p": p, "gl": gl, "v": v, "n": n}


def magnitude(v):
    if v < 0.10:
        return "desprezivel"
    if v < 0.30:
        return "fraco"
    if v < 0.50:
        return "moderado"
    return "forte"


VARIAVEIS = [
    "Model Name", "Test Line", "Test Group", "Test Station",
    "Reason Desc", "Duty Type", "component_tipo", "loc_tipo",
]

resultados = []
for var in VARIAVEIS:
    if var not in df.columns:
        continue
    tabela = pd.crosstab(df[var], df["estagio_defeito"])
    if tabela.shape[0] < 2 or tabela.shape[1] < 2:
        continue
    r = v_cramer(tabela)
    resultados.append({
        "variavel": var,
        "chi2": round(r["chi2"], 2),
        "p_valor": r["p"],
        "gl": r["gl"],
        "v_cramer": round(r["v"], 4),
        "n": r["n"],
        "significativo": "sim" if r["p"] < ALPHA else "nao",
        "magnitude": magnitude(r["v"]),
    })

df_assoc = pd.DataFrame(resultados).sort_values("v_cramer", ascending=False)
print()
for _, r in df_assoc.iterrows():
    p_fmt = f"{r['p_valor']:.2e}" if r["p_valor"] < 0.001 else f"{r['p_valor']:.4f}"
    print(f"    {r['variavel']:15s} chi2={r['chi2']:>9.2f}  p={p_fmt:>10s}  "
          f"V={r['v_cramer']:.4f}  {r['magnitude']:12s}  {r['significativo']}")

df_assoc.to_csv(DIR_TABELAS / "02_qui_quadrado.csv", index=False)
print(f"\n    tabela salva: {DIR_TABELAS / '02_qui_quadrado.csv'}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(df_assoc["variavel"][::-1], df_assoc["v_cramer"][::-1], color="#4C72B0")
ax.axvline(0.10, color="#C44E52", linestyle="--", label="limiar de efeito pratico (0,10)")
ax.set_title("V de Cramer por variavel")
ax.set_xlabel("V de Cramer")
ax.legend()
salvar_figura("02_v_cramer")

# ------------------------------------------------ [3] regras de associacao
secao("[3] REGRAS DE ASSOCIACAO")

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_OK = True
except ImportError:
    MLXTEND_OK = False
    print("\n    mlxtend nao instalado. Execute: pip install mlxtend")
    print("    Secao ignorada.")

if MLXTEND_OK:
    CATEGORICAS = ["Model Name", "Test Line", "Test Group",
                   "Test Station", "Reason Desc", "component_tipo"]

    df_regras = df[CATEGORICAS].copy()
    for col in CATEGORICAS:
        s = df_regras[col].fillna("SEM_VALOR").astype(str)
        vc = s.value_counts()
        raras = vc[vc < 10].index
        s = s.where(~s.isin(raras), "OUTROS")
        vc2 = s.value_counts()
        if len(vc2) > 15:
            manter = vc2.head(14).index
            s = s.where(s.isin(manter), "OUTROS")
        df_regras[col] = s

    onehot = pd.get_dummies(df_regras, dtype=bool)
    itemsets = apriori(onehot, min_support=MIN_SUPORTE, use_colnames=True, max_len=3)

    if len(itemsets) == 0:
        print("\n    Nenhum itemset encontrado com o suporte minimo.")
    else:
        regras = association_rules(itemsets, metric="confidence", min_threshold=MIN_CONFIANCA)
        regras = regras[regras["lift"] > MIN_LIFT].sort_values("lift", ascending=False)

        if len(regras) == 0:
            print("\n    Nenhuma regra encontrada com os limiares definidos.")
        else:
            regras["antecedentes"] = regras["antecedents"].apply(lambda x: ", ".join(sorted(x)))
            regras["consequentes"] = regras["consequents"].apply(lambda x: ", ".join(sorted(x)))
            saida = regras[["antecedentes", "consequentes", "support", "confidence", "lift"]].head(TOP_N_REGRAS)
            saida.columns = ["antecedentes", "consequentes", "suporte", "confianca", "lift"]

            print(f"\n    {len(regras)} regras encontradas. Top {TOP_N_REGRAS}:")
            for _, r in saida.iterrows():
                print(f"    {r['antecedentes'][:45]:45s} -> {r['consequentes'][:30]:30s} "
                      f"sup={r['suporte']:.3f} conf={r['confianca']:.3f} lift={r['lift']:.2f}")

            saida.to_csv(DIR_TABELAS / "03_regras_associacao.csv", index=False)
            print(f"\n    tabela salva: {DIR_TABELAS / '03_regras_associacao.csv'}")

            fig, ax = plt.subplots(figsize=(8, 5))
            disp = ax.scatter(regras["support"], regras["confidence"],
                              c=regras["lift"], cmap="viridis", alpha=0.7)
            plt.colorbar(disp, label="lift")
            ax.set_title("Regras de associacao: suporte x confianca")
            ax.set_xlabel("Suporte")
            ax.set_ylabel("Confianca")
            salvar_figura("03_regras")

# --------------------------------------------- [4] detector de desvios
secao("[4] DETECTOR DE DESVIOS DE PROCESSO")


def wilson_interval(sucessos, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = sucessos / n
    denom = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / denom
    margem = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centro - margem), min(1.0, centro + margem))


df["refalha"] = (df["estagio_defeito"] == 3).astype(int)
taxa_global = df["refalha"].mean()
print(f"\nTaxa global de refalha: {taxa_global*100:.2f}% "
      f"({df['refalha'].sum()} de {len(df)})")

AGRUPAMENTOS = {
    "Test Line": ["Test Line"],
    "Test Station": ["Test Station"],
    "Linha + Estacao": ["Test Line", "Test Station"],
    "Model Name": ["Model Name"],
    "component_tipo": ["component_tipo"],
}

todos = []
for nome, cols in AGRUPAMENTOS.items():
    grupo = df.groupby(cols).agg(
        n=("refalha", "size"),
        sucessos=("refalha", "sum"),
    ).reset_index()
    grupo = grupo[grupo["n"] >= 30].copy()
    if len(grupo) == 0:
        continue
    grupo["taxa"] = grupo["sucessos"] / grupo["n"]
    grupo["ic_inf"] = grupo.apply(lambda r: wilson_interval(r["sucessos"], r["n"])[0], axis=1)
    grupo["ic_sup"] = grupo.apply(lambda r: wilson_interval(r["sucessos"], r["n"])[1], axis=1)

    def classificar(r):
        if r["ic_inf"] > taxa_global:
            return "POSITIVO"
        if r["ic_sup"] < taxa_global:
            return "NEGATIVO"
        return "NORMAL"

    grupo["desvio"] = grupo.apply(classificar, axis=1)
    grupo["agrupamento"] = nome
    grupo["grupo"] = grupo[cols].astype(str).agg(" | ".join, axis=1)

    positivos = grupo[grupo["desvio"] == "POSITIVO"].sort_values("taxa", ascending=False)
    print(f"\n--- {nome} ({len(grupo)} grupos com n >= 30) ---")
    if len(positivos):
        for _, r in positivos.iterrows():
            print(f"    {r['grupo'][:40]:40s} n={r['n']:>4}  taxa={r['taxa']*100:>5.1f}%  "
                  f"IC=[{r['ic_inf']*100:.1f}%, {r['ic_sup']*100:.1f}%]")
    else:
        print("    Nenhum desvio positivo.")

    todos.append(grupo[["agrupamento", "grupo", "n", "taxa", "ic_inf", "ic_sup", "desvio"]])

df_desvios = pd.DataFrame()
if todos:
    df_desvios = pd.concat(todos, ignore_index=True).sort_values("taxa", ascending=False)
    df_desvios.to_csv(DIR_TABELAS / "04_desvios_processo.csv", index=False)
    print(f"\n    tabela salva: {DIR_TABELAS / '04_desvios_processo.csv'}")

    positivos = df_desvios[df_desvios["desvio"] == "POSITIVO"].head(15)
    if len(positivos):
        fig, ax = plt.subplots(figsize=(9, 6))
        rotulos = positivos["agrupamento"] + ": " + positivos["grupo"]
        ax.barh(rotulos[::-1], positivos["taxa"][::-1] * 100, color="#C44E52")
        ax.axvline(taxa_global * 100, color="#4C72B0", linestyle="--",
                   label=f"taxa global ({taxa_global*100:.1f}%)")
        ax.set_title("Desvios positivos: taxa de refalha por grupo")
        ax.set_xlabel("Taxa de refalha (%)")
        ax.legend()
        salvar_figura("04_desvios")

# ------------------------------------------------------------- [5] sintese
secao("[5] SINTESE")

n_sig = int((df_assoc["significativo"] == "sim").sum())
n_efeito = int((df_assoc["v_cramer"] > 0.10).sum())
n_desvios = int((df_desvios["desvio"] == "POSITIVO").sum()) if len(df_desvios) else 0

print(f"\n    Variaveis analisadas          : {len(df_assoc)}")
print(f"    Com associacao significativa  : {n_sig}")
print(f"    Com efeito pratico (V > 0,10) : {n_efeito}")
print(f"    Desvios positivos detectados  : {n_desvios}")

print("\n    Arquivos gerados:")
for pasta in (DIR_TABELAS, DIR_FIGURAS):
    for arq in sorted(pasta.glob("*")):
        print(f"      {arq}")

print("\nAnalise concluida.")
