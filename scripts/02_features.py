"""02 - Engenharia de atributos para ANALISE DE CAUSA-RAIZ."""
import pandas as pd
from _config import WORKING, CFG

ENTRADA = WORKING / "base_limpa.xlsx"
SAIDA_DEFEITO = WORKING / "base_defeito.xlsx"
SAIDA_EVENTO = WORKING / "base_evento.xlsx"

MIN_OCORRENCIAS = CFG["features"]["min_ocorrencias"]
MAX_CATEGORIAS = CFG["features"]["max_categorias"]

df = pd.read_excel(ENTRADA)
print(f"Entrada: {len(df)} linhas x {len(df.columns)} colunas")

# [1] STATE FLAG COMO FEATURE DESCRITIVA
print("\n[1] State Flag como feature descritiva de estagio")
sf = pd.to_numeric(df["State Flag"], errors="coerce")
n_invalidos = int(sf.isna().sum())
print(f"    Valores invalidos (NaN): {n_invalidos}")

if n_invalidos:
    print(f"    ATRIBUINDO State Flag = 1 aos {n_invalidos} registros invalidos")
    print("    Justificativa: defeitos completos, evento unico, padrao DEBUG/FT0/TS")
    sf = sf.fillna(1)

df["estagio_defeito"] = sf.astype(int)
print("    Distribuicao:")
for v, q in df["estagio_defeito"].value_counts().sort_index().items():
    print(f"      {v}: {q} ({q/len(df)*100:.1f}%)")
print("    estagio_defeito: 1=primeira falha, 3=refalha (2=reteste nao ocorre)")

# [2] MAPA DE PREFIXOS -> TIPO DE COMPONENTE
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

# [3] LOCATION DESC -> tipo de componente + n posicoes
print("\n[2] Location Desc -> tipo de componente")
loc = df["Location Desc"].fillna("").astype(str).str.strip()

def contar_posicoes(s):
    s = str(s).strip()
    if s in ("", "nan", "None"):
        return 0
    return len([p for p in s.replace(";", ",").split(",") if p.strip()])

df["loc_n_posicoes"] = loc.apply(contar_posicoes)
df["loc_tipo"] = loc.apply(tipo_componente)
print(f"    loc_n_posicoes: media {df['loc_n_posicoes'].mean():.2f}, max {df['loc_n_posicoes'].max()}")
print(f"    loc_tipo: {df['loc_tipo'].nunique()} tipos")
print("    Distribuicao de loc_tipo:")
for v, q in df["loc_tipo"].value_counts().items():
    print(f"      {v:15s}: {q:5d} ({q/len(df)*100:5.2f}%)")

# [4] COMPONENT -> tipo
print("\n[3] Component -> tipo de componente")
comp = df["Component"].fillna("").astype(str).str.strip()
df["component_tipo"] = comp.apply(tipo_componente)
print(f"    Component: {df['Component'].nunique()} valores -> {df['component_tipo'].nunique()} tipos")
print("    Distribuicao de component_tipo:")
for v, q in df["component_tipo"].value_counts().items():
    print(f"      {v:15s}: {q:5d} ({q/len(df)*100:5.2f}%)")

# [5] CONTROLE DE CARDINALIDADE
print(f"\n[4] Controle de cardinalidade (min={MIN_OCORRENCIAS}, max={MAX_CATEGORIAS})")

def controlar_cardinalidade(serie, nome):
    s = serie.fillna("SEM_VALOR").astype(str)
    vc = s.value_counts()
    raras = vc[vc < MIN_OCORRENCIAS].index
    n_raras = len(raras)
    s = s.where(~s.isin(raras), "OUTROS")
    vc2 = s.value_counts()
    if len(vc2) > MAX_CATEGORIAS:
        manter = vc2.head(MAX_CATEGORIAS - 1).index
        s = s.where(s.isin(manter), "OUTROS")
    print(f"    {nome}: {serie.nunique()} -> {s.nunique()} categorias ({n_raras} raras agrupadas)")
    return s

CATEGORICAS = ["Model Name", "Test Line", "Test Group", "Test Station",
               "Reason Desc", "Duty Type", "loc_tipo", "component_tipo"]
CATEGORICAS = [c for c in CATEGORICAS if c in df.columns]

for col in CATEGORICAS:
    df[col] = controlar_cardinalidade(df[col], col)

# [6] FEATURES TEMPORAIS
print("\n[5] Features temporais")
df["Test Time"] = pd.to_datetime(df["Test Time"], errors="coerce")
df["hora_teste"] = df["Test Time"].dt.hour.fillna(-1).astype(int)
df["dia_semana"] = df["Test Time"].dt.day_of_week.fillna(-1).astype(int)
df["semana_iso"] = df["Test Time"].dt.isocalendar().week.fillna(0).astype(int)
print("    hora_teste, dia_semana, semana_iso criadas")

# [7] BASE DE DEFEITO
print("\n[6] Base de defeito")
NUMERICAS = ["loc_n_posicoes", "houve_reparo", "estagio_defeito",
             "hora_teste", "dia_semana", "semana_iso"]

dummies = pd.get_dummies(df[CATEGORICAS].astype(str), prefix=CATEGORICAS, dtype=int)
print(f"    One-hot: {len(CATEGORICAS)} colunas -> {dummies.shape[1]} dummies")

cols_id = ["defeito_id", "evento_id", "Serial Number", "Test Time"]
base_defeito = pd.concat(
    [df[cols_id + NUMERICAS].reset_index(drop=True), dummies.reset_index(drop=True)],
    axis=1
)
print(f"    base_defeito: {base_defeito.shape[0]} linhas x {base_defeito.shape[1]} colunas")

# [8] BASE DE EVENTO
print("\n[7] Base de evento")
agg = {c: "first" for c in NUMERICAS + CATEGORICAS + ["Serial Number", "Test Time"]}
base_evento = df.groupby("evento_id", as_index=False).agg(agg)

n_def = df.groupby("evento_id").size().rename("n_defeitos")
n_comp = df.groupby("evento_id")["component_tipo"].nunique().rename("n_componentes")
n_est = df.groupby("evento_id")["estagio_defeito"].nunique().rename("n_estagios")

base_evento = base_evento.merge(n_def, left_on="evento_id", right_index=True)
base_evento = base_evento.merge(n_comp, left_on="evento_id", right_index=True)
base_evento = base_evento.merge(n_est, left_on="evento_id", right_index=True)

NUM_EVENTO = NUMERICAS + ["n_defeitos", "n_componentes", "n_estagios"]
dummies_ev = pd.get_dummies(base_evento[CATEGORICAS].astype(str), prefix=CATEGORICAS, dtype=int)
print(f"    One-hot: {len(CATEGORICAS)} colunas -> {dummies_ev.shape[1]} dummies")

base_evento_final = pd.concat(
    [base_evento[["evento_id", "Serial Number", "Test Time"] + NUM_EVENTO].reset_index(drop=True),
     dummies_ev.reset_index(drop=True)],
    axis=1
)
print(f"    base_evento: {base_evento_final.shape[0]} linhas x {base_evento_final.shape[1]} colunas")

# [9] VALIDACAO
print("\n[8] Validacao")
for nome, base in [("base_defeito", base_defeito), ("base_evento", base_evento_final)]:
    nulos = int(base.isna().sum().sum())
    razao = base.shape[1] / base.shape[0]
    print(f"    {nome}: {base.shape[0]} linhas x {base.shape[1]} colunas")
    print(f"      nulos: {nulos} | razao colunas/linhas: {razao:.3f}")

print("\n    Distribuicao de estagio_defeito (base_defeito):")
for v, q in base_defeito["estagio_defeito"].value_counts().sort_index().items():
    print(f"      {v}: {q} ({q/len(base_defeito)*100:.1f}%)")

print("\n    Eventos com multiplos defeitos:")
multi = base_evento_final[base_evento_final["n_defeitos"] > 1]
print(f"      {len(multi)} eventos ({len(multi)/len(base_evento_final)*100:.1f}%)")

# [10] SALVAR
base_defeito.to_excel(SAIDA_DEFEITO, index=False)
base_evento_final.to_excel(SAIDA_EVENTO, index=False)

print(f"\nSaida: {SAIDA_DEFEITO} - {base_defeito.shape[0]} linhas x {base_defeito.shape[1]} colunas")
print(f"Saida: {SAIDA_EVENTO} - {base_evento_final.shape[0]} linhas x {base_evento_final.shape[1]} colunas")
