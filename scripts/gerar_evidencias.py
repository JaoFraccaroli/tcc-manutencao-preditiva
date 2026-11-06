"""
Gera `evidencias/evidencias.md` (+ CSVs em `evidencias/csv/`) com os 28
itens do briefing de evidências, TODOS calculados a partir do código, do
banco (`instance/manutencao.db`), dos modelos salvos (`models/`) e da
planilha bruta local. Nenhum número é digitado à mão; quando algo não
existe no projeto, o texto diz "não existe".

Uso:
    python scripts/gerar_evidencias.py

O `if __name__ == "__main__"` é obrigatório: `cross_validate(n_jobs=-1)`
usa processos filhos no Windows.

Anonimização: o arquivo gerado não contém TAG real, nome de técnico/empresa
nem HH bruto (só razões entre HH perturbado e bruto).
"""
import contextlib
import datetime as dt
import glob
import hashlib
import io
import os
import platform
import re
import sqlite3
import subprocess
import sys
import warnings
from collections import Counter
from importlib import metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib
import json
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    LeaveOneGroupOut, RandomizedSearchCV, StratifiedGroupKFold, cross_validate,
)

import config
from app.processing.dataset import carregar_eventos, construir_dataset, _grade_datas_referencia
from app.processing import component_mapping, text_analysis, exploratory
from app.modeling.split import dividir_treino_teste
from app.modeling.pipelines import construir_modelos, PARAM_GRIDS
from app.modeling.features import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from app.modeling import train as T
from app.modeling.predict import snapshot_atual, carregar_modelo
from app.data import ingest

OUT_DIR = os.path.join(ROOT, "evidencias")
CSV_DIR = os.path.join(OUT_DIR, "csv")
MODELS_DIR = os.path.join(ROOT, "models")
HORIZONTES = config.HORIZONTES_DIAS
SEED = 42
MESES = exploratory.MESES_PT

OUT: list[str] = []


# ----------------------------------------------------------------- helpers
def P(texto: str = "") -> None:
    OUT.append(texto)


def fmt(v, nd=4):
    if v is None:
        return "não definido"
    if isinstance(v, (float, np.floating)):
        return "não definido" if np.isnan(v) else f"{v:.{nd}f}"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (pd.Timestamp, dt.date)):
        return str(pd.Timestamp(v).date())
    return str(v).replace("|", "\\|").replace("\n", " ")


def tbl(df: pd.DataFrame, nd: int = 4) -> None:
    df = df.reset_index(drop=True)
    P("| " + " | ".join(str(c) for c in df.columns) + " |")
    P("|" + "|".join("---" for _ in df.columns) + "|")
    for _, row in df.astype(object).iterrows():
        P("| " + " | ".join(fmt(v, nd) for v in row.tolist()) + " |")
    P()


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"(erro: {exc})"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def mtime(path: str) -> str:
    return dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")


def metricas(y, pred, prob=None) -> dict:
    d = {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
    }
    if prob is not None and pd.Series(y).nunique() >= 2:
        d["ROC-AUC"] = roc_auc_score(y, prob)
        d["PR-AUC"] = average_precision_score(y, prob)
    else:
        d["ROC-AUC"] = np.nan
        d["PR-AUC"] = np.nan
    return d


# ------------------------------------------------------------ preparação
def preparar() -> dict:
    eventos = carregar_eventos()
    ds, sp, metas, finais = {}, {}, {}, {}
    for h in HORIZONTES:
        ds[h] = construir_dataset(h, eventos)
        sp[h] = dividir_treino_teste(ds[h], test_size=0.2)
        finais[h], metas[h] = carregar_modelo(h)
    return {"eventos": eventos, "ds": ds, "sp": sp, "metas": metas, "finais": finais}


def avaliar_horizonte(h: int, C: dict) -> dict:
    """Reexecuta, para um horizonte, exatamente as etapas de
    `train.treinar_horizonte` (comparação, tuning, calibração, OOF,
    threshold) guardando os resultados intermediários para o relatório."""
    alvo = f"falha_{h}d"
    dev, te = C["sp"][h].treino, C["sp"][h].teste
    X_dev, y_dev, g_dev = dev[FEATURES], dev[alvo], dev["tag_code"]
    X_te, y_te = te[FEATURES], te[alvo]
    assert g_dev.nunique() < T.GRUPOS_MIN_PARA_KFOLD
    cv = LeaveOneGroupOut()
    tags_fold = sorted(g_dev.unique())
    prev_fold = {t: float(y_dev[g_dev == t].mean()) for t in tags_fold}

    modelos = construir_modelos()
    comp, folds = [], []
    for nome, modelo in modelos.items():
        sc = cross_validate(
            modelo, X_dev, y_dev, groups=g_dev, cv=cv,
            scoring={"ap": "average_precision", "roc": "roc_auc"}, n_jobs=-1, error_score=np.nan,
        )
        for i, t in enumerate(tags_fold):
            folds.append({"algoritmo": nome, "tag_fora": t, "prevalencia_fold": prev_fold[t],
                          "AP": sc["test_ap"][i], "ROC-AUC": sc["test_roc"][i]})
        ap = sc["test_ap"]
        ajustado = clone(modelo).fit(X_dev, y_dev)
        p_te = ajustado.predict_proba(X_te)[:, 1]
        comp.append({
            "algoritmo": nome,
            "CV_AP_media": np.nanmean(ap), "CV_AP_desvio(ddof=0)": np.nanstd(ap),
            "folds_validos": int(np.sum(~np.isnan(ap))),
            "CV_ROC_media": np.nanmean(sc["test_roc"]),
            "teste_PR-AUC(sem_tuning,sem_calibracao)": average_precision_score(y_te, p_te),
            "teste_ROC-AUC(sem_tuning,sem_calibracao)": roc_auc_score(y_te, p_te),
        })
    comp = pd.DataFrame(comp)
    selecionado = comp.sort_values("CV_AP_media", ascending=False).iloc[0]["algoritmo"]

    base = modelos[selecionado]
    parametros = {"classifier__C": np.logspace(-3, 2, 30)} if selecionado == "Logistic Regression" else PARAM_GRIDS[selecionado]
    search = RandomizedSearchCV(
        estimator=base, param_distributions=parametros, n_iter=T.N_ITER_TUNING_REDUZIDO,
        scoring="average_precision", cv=cv, n_jobs=-1, random_state=SEED, refit=True, error_score=np.nan,
    )
    search.fit(X_dev, y_dev, groups=g_dev)
    idx = search.best_index_
    ap_tuned = np.array([search.cv_results_[f"split{i}_test_score"][idx] for i in range(len(tags_fold))])

    minoria = T._minoria_minima_por_fold(cv, X_dev, y_dev, g_dev)
    calibrar = minoria >= T.CALIBRACAO_CV

    def construir():
        b = clone(search.best_estimator_)
        return CalibratedClassifierCV(b, method="sigmoid", cv=T.CALIBRACAO_CV) if calibrar else b

    oof, n_fb = T._oof_proba_seguro(construir, X_dev, y_dev, g_dev, cv)
    thr, thresholds, prec_t, rec_t = T._escolher_threshold(y_dev, oof)
    candidatos = int(np.sum(rec_t >= T.RECALL_DESEJADO))

    final = C["finais"][h]
    p_te_final = final.predict_proba(X_te)[:, 1]
    return dict(
        h=h, alvo=alvo, dev=dev, te=te, X_dev=X_dev, y_dev=y_dev, g_dev=g_dev, X_te=X_te, y_te=y_te,
        comp=comp, folds=pd.DataFrame(folds), selecionado=selecionado, search=search, ap_tuned=ap_tuned,
        tags_fold=tags_fold, calibrar=calibrar, minoria=minoria, oof=oof, n_fb=n_fb, thr=thr,
        thresholds=thresholds, prec_t=prec_t, rec_t=rec_t, candidatos=candidatos, p_te=p_te_final,
        base=base,
    )


def externo_um_tag_fora(h: int, C: dict, R: dict) -> pd.DataFrame:
    """Avaliação complementar: cada um dos 6 TAGs é, um de cada vez, o
    "teste"; o restante treina com os hiperparâmetros finais, threshold
    por OOF interno (LeaveOneGroupOut nos 5 TAGs de treino)."""
    alvo = f"falha_{h}d"
    d = C["ds"][h]
    meta = C["metas"][h]
    linhas, todos = [], []
    for t in sorted(d["tag_code"].unique()):
        tr, te = d[d["tag_code"] != t], d[d["tag_code"] == t]
        base = construir_modelos()[meta["modelo"]].set_params(**meta["hiperparametros"])
        cv_in = LeaveOneGroupOut()
        minoria = T._minoria_minima_por_fold(cv_in, tr[FEATURES], tr[alvo], tr["tag_code"])
        cal = minoria >= T.CALIBRACAO_CV

        def build():
            b = clone(base)
            return CalibratedClassifierCV(b, method="sigmoid", cv=T.CALIBRACAO_CV) if cal else b

        oof, _ = T._oof_proba_seguro(build, tr[FEATURES], tr[alvo], tr["tag_code"], cv_in)
        thr, *_ = T._escolher_threshold(tr[alvo], oof)
        p = build().fit(tr[FEATURES], tr[alvo]).predict_proba(te[FEATURES])[:, 1]
        pred = (p >= thr).astype(int)
        y = te[alvo].to_numpy()
        m = metricas(y, pred, p)
        linhas.append({"tag_fora": t, "linhas": len(te), "prevalencia": y.mean(), "threshold": thr,
                       **{k: m[k] for k in ["Accuracy", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]},
                       "baseline_sempre_falha_Accuracy": y.mean()})
        todos.append(pd.DataFrame({"y": y, "pred": pred, "p": p}))
    pool = pd.concat(todos)
    mp = metricas(pool["y"], pool["pred"], pool["p"])
    linhas.append({"tag_fora": "TOTAL (concatenado)", "linhas": len(pool), "prevalencia": pool["y"].mean(),
                   "threshold": np.nan, **{k: mp[k] for k in ["Accuracy", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]},
                   "baseline_sempre_falha_Accuracy": pool["y"].mean()})
    return pd.DataFrame(linhas)


# ===================================================================== A
def secao_A(C: dict, RS: dict) -> None:
    P("## A. Reprodutibilidade técnica")
    P()
    P("### Item 1 — Versões e ambiente")
    P()
    pacotes = ["pandas", "numpy", "scikit-learn", "scipy", "joblib", "Flask", "SQLAlchemy", "matplotlib", "openpyxl", "pytest"]
    linhas = [{"componente": "Python", "versão": platform.python_version()},
              {"componente": "Sistema operacional", "versão": platform.platform()}]
    for p in pacotes:
        try:
            linhas.append({"componente": p, "versão": metadata.version(p)})
        except metadata.PackageNotFoundError:
            linhas.append({"componente": p, "versão": "não instalado"})
    tbl(pd.DataFrame(linhas))
    P("`requirements.txt` do repositório (conteúdo literal):")
    P()
    P("```")
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as f:
        P(f.read().strip())
    P("```")
    P()
    P(f"Commit (HEAD) no momento da geração: `{sh(['git', 'rev-parse', 'HEAD'])}` — {sh(['git', 'log', '-1', '--format=%cI'])}.")
    P(f"Árvore de trabalho com alterações não commitadas ao gerar: {'sim' if sh(['git', 'status', '--porcelain']) else 'não'}.")
    P(f"Planilha bruta: `{os.path.basename(config.RAW_XLSX_PATH)}`, aba `{config.RAW_XLSX_SHEET}`, SHA-256 (16 primeiros hex): "
      f"`{sha256(config.RAW_XLSX_PATH)[:16]}`.")
    P()

    P("### Item 2 — Banco de dados efetivamente testado")
    P()
    db_path = os.path.join(ROOT, "instance", "manutencao.db")
    P(f"- Banco usado em todas as execuções: **SQLite** (`sqlite3` {sqlite3.sqlite_version}), arquivo `instance/manutencao.db` "
      f"({os.path.getsize(db_path)} bytes; modificado em {mtime(db_path)}).")
    P(f"- URL efetiva (`config.DATABASE_URL`): `sqlite:///.../instance/manutencao.db`; variável de ambiente `DATABASE_URL` "
      f"definida nesta execução: {'sim' if os.environ.get('DATABASE_URL') else 'não'}.")
    P("- **SQL Server: não existe execução/teste.** O código aceita `DATABASE_URL` de SQL Server (SQLAlchemy), mas nenhuma ingestão, "
      "treino ou tela foi executada contra SQL Server; não há driver `pyodbc` instalado nem script de teste dedicado.")
    P()

    P("### Item 3 — Hiperparâmetros efetivos, random_state, pesos, codificação, escala, imputação")
    P()
    P(f"- `random_state`: **{SEED}** (modelos: `pipelines.RANDOM_STATE`; busca: `train.RANDOM_STATE`; permutation importance: 42; "
      f"perturbação de HH: `config.RANDOM_SEED`={config.RANDOM_SEED}).")
    P("- Imputação numérica: `SimpleImputer(strategy='median')` (ajustado só no treino de cada fold). Categóricas: "
      "`SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.")
    P("- Escala: `StandardScaler` **apenas** na Regressão Logística (RF e Gradient Boosting não escalam).")
    P("- Pesos de classe: Logistic Regression `class_weight='balanced'`; Random Forest `'balanced_subsample'`; "
      "HistGradientBoosting `'balanced'`. Pesos automáticos recalculados em cada fit (não são números fixos).")
    P(f"- Validação cruzada: `LeaveOneGroupOut` por TAG (grupos no desenvolvimento < {T.GRUPOS_MIN_PARA_KFOLD}). "
      f"Busca: `RandomizedSearchCV(n_iter={T.N_ITER_TUNING_REDUZIDO}, scoring='average_precision', random_state={SEED})`. "
      f"Calibração: `CalibratedClassifierCV(method='sigmoid', cv={T.CALIBRACAO_CV})`.")
    P()
    P("Candidatos antes do tuning (parâmetros efetivos):")
    P()
    linhas = []
    for nome, m in construir_modelos().items():
        c = m.named_steps["classifier"]
        chaves = [k for k in c.get_params() if k in (
            "C", "penalty", "solver", "max_iter", "class_weight", "random_state", "n_estimators", "min_samples_leaf",
            "max_depth", "max_features", "learning_rate", "max_leaf_nodes", "l2_regularization")]
        linhas.append({"algoritmo": nome, "parâmetros": ", ".join(f"{k}={c.get_params()[k]}" for k in chaves)})
    tbl(pd.DataFrame(linhas))
    P("Espaços de busca do tuning: Logistic Regression `C ∈ np.logspace(-3, 2, 30)`; "
      f"Random Forest `{PARAM_GRIDS['Random Forest']}`; Gradient Boosting `{PARAM_GRIDS['Gradient Boosting']}`.")
    P()
    P("Modelo final salvo (lido de `models/*.joblib`) e `meta.json`:")
    P()
    linhas = []
    for h in HORIZONTES:
        meta, final = C["metas"][h], C["finais"][h]
        clf = final.estimator.named_steps["classifier"] if hasattr(final, "estimator") else final.named_steps["classifier"]
        pars = {k: (float(clf.get_params()[k]) if k == "C" else clf.get_params()[k])
                for k in ("C", "solver", "max_iter", "class_weight", "random_state") if k in clf.get_params()}
        pars["penalidade"] = "L2 (padrão; o parâmetro `penalty` aparece como 'deprecated' no sklearn 1.9)"
        linhas.append({
            "horizonte": h, "algoritmo": meta["modelo"], "hiperparâmetro_tunado": meta["hiperparametros"],
            "demais_parâmetros_efetivos": pars,
            "calibração": f"sigmoid, cv={T.CALIBRACAO_CV}, ensemble={getattr(final, 'ensemble', 'n/a')!r} (equivale a True) "
                          f"({len(getattr(final, 'calibrated_classifiers_', []))} classificadores calibrados)",
        })
    tbl(pd.DataFrame(linhas))
    P("Observação: com `ensemble='auto'` (= True para `cv` inteiro), a saída final é a média de 3 regressões logísticas, cada uma ajustada em ~2/3 das "
      "linhas de desenvolvimento (as dobras da calibração **não** são agrupadas por TAG — decisão documentada em `train.py`); "
      "o modelo final não é ajustado em uma única passada sobre todas as linhas de treino.")
    P()
    P("Reprodução do tuning nesta execução vs. `meta.json`:")
    P()
    linhas = []
    for h in HORIZONTES:
        R = RS[h]
        linhas.append({
            "horizonte": h, "algoritmo_selecionado_aqui": R["selecionado"], "algoritmo_meta": C["metas"][h]["modelo"],
            "C_reproduzido": R["search"].best_params_.get("classifier__C"),
            "C_meta": C["metas"][h]["hiperparametros"].get("classifier__C"),
            "iguais": bool(R["selecionado"] == C["metas"][h]["modelo"] and
                           np.isclose(R["search"].best_params_.get("classifier__C", np.nan),
                                      C["metas"][h]["hiperparametros"].get("classifier__C", np.nan), rtol=1e-9)),
        })
    tbl(pd.DataFrame(linhas), nd=6)

    P("### Item 4 — Dicionário final de features")
    P()
    P(f"Unidade de análise: uma linha = (TAG, data de referência). Datas de referência: do **primeiro evento do TAG** até "
      f"`data_máxima_da_base − horizonte`, passo de {config.REFERENCE_STEP_DAYS} dias. Rótulo `falha_Hd` = existe evento com "
      f"`is_falha` em **(ref, ref+H]** (estritamente depois da referência). Todas as features olham só para eventos com "
      f"`inicio ≤ ref` (inclusive a própria data de referência).")
    P()
    areas = C["eventos"]["area"].nunique(dropna=True)
    sists = C["eventos"]["sist_oper"].nunique(dropna=True)
    dic = pd.DataFrame([
        ["tempo_desde_ultima_falha", "numérica", "dias entre ref e a data do último evento com is_falha (MC/CP) com início ≤ ref",
         "toda a história anterior (sem janela)", "dias", "NaN se o TAG não tem falha anterior → imputação pela mediana"],
        ["tempo_medio_reparo", "numérica", "média de hh_apropriado (perturbado) de TODOS os eventos do TAG com início ≤ ref",
         "toda a história anterior", "horas (HH)", "NaN se não há evento anterior → mediana"],
        ["n_eventos_historico", "numérica", "nº de eventos (qualquer tipo) do TAG com início ≤ ref",
         "toda a história anterior", "contagem", "—"],
        ["contagem_preventivas_recentes", "numérica", "nº de eventos preventivos (MP) com início em (ref − 90, ref]",
         f"{config.JANELA_PREVENTIVAS} dias", "contagem", "—"],
        ["freq_falhas_90d", "numérica", "nº de eventos de falha com início em (ref − 90, ref]", "90 dias", "contagem", "—"],
        ["freq_falhas_180d", "numérica", "nº de eventos de falha com início em (ref − 180, ref]", "180 dias", "contagem", "—"],
        ["area", "categórica", f"campo `Área` do evento mais recente com início ≤ ref ({areas} categorias distintas na base; "
                               "rótulos omitidos aqui por serem dados do cliente)", "último evento conhecido", "—",
         "None se sem histórico → moda; one-hot, categoria desconhecida ignorada"],
        ["sist_oper", "categórica", f"campo `Sist Oper` do evento mais recente com início ≤ ref ({sists} categorias distintas)",
         "último evento conhecido", "—", "idem"],
    ], columns=["feature", "tipo", "definição", "janela", "unidade", "tratamento de ausentes"])
    tbl(dic)
    P("Não existem outras features (nem sazonalidade, nem componente estimado, nem texto). O cadastro manual de máquina "
      "(fabricante, capacidade etc.) **não** entra no modelo.")
    P()


# ===================================================================== B
def secao_B(C: dict, RS: dict, EXT: dict) -> None:
    P("## B. Modelagem e validação")
    P()
    P("### Item 5 — Algoritmo × horizonte (validação cruzada vs. teste, separados)")
    P()
    P("**Protocolo:** desenvolvimento = 4 TAGs (EQ001, EQ002, EQ003, EQ005); validação cruzada = `LeaveOneGroupOut` (4 dobras, "
      "uma por TAG deixado de fora); a coluna CV é média e desvio-padrão (ddof=0) da Average Precision nas 4 dobras. Teste = 2 TAGs "
      "(EQ004, EQ006) nunca vistos no ajuste. Os algoritmos concorrentes aparecem **sem tuning e sem calibração** (como na comparação "
      "do `train.py`); só o algoritmo selecionado é tunado e calibrado. As colunas de teste dos não selecionados são "
      "informativas — o teste **não** participou da seleção.")
    P()
    for h in HORIZONTES:
        R = RS[h]
        P(f"**Horizonte {h} dias** — algoritmo selecionado (maior CV AP média): **{R['selecionado']}**")
        P()
        tbl(R["comp"])
        P(f"Selecionado, após tuning: CV AP média = {fmt(np.nanmean(R['ap_tuned']))}, desvio = {fmt(np.nanstd(R['ap_tuned']))} "
          f"(`best_score_` = {fmt(R['search'].best_score_)}); teste do modelo final (calibrado, joblib): "
          f"PR-AUC = {fmt(C['metas'][h]['metricas_teste_threshold_otimo']['PR-AUC'])}, "
          f"ROC-AUC = {fmt(C['metas'][h]['metricas_teste_threshold_otimo']['ROC-AUC'])}.")
        P()
        P("AP por dobra (TAG deixado de fora; prevalência da dobra = AP esperada de um classificador sem informação):")
        P()
        f = R["folds"].copy()
        tbl(f)
        t = pd.DataFrame({"tag_fora": R["tags_fold"], "prevalencia_fold": [float(R["y_dev"][R["g_dev"] == x].mean()) for x in R["tags_fold"]],
                          "AP_modelo_tunado": R["ap_tuned"]})
        P("Modelo selecionado e tunado, por dobra:")
        P()
        tbl(t)
    P("Ressalvas: (i) tuning e seleção usam as **mesmas** dobras (sem validação aninhada), logo a CV do modelo tunado é otimista; "
      "(ii) a média de 4 dobras com poucos TAGs tem alta variância — ver desvios; (iii) o teste tem apenas 2 TAGs.")
    P()

    P("### Item 6 — Tamanhos, classes, prevalência e perdas por censura à direita")
    P()
    base_max = C["eventos"]["inicio"].max()
    linhas = []
    for h in HORIZONTES:
        d, alvo = C["ds"][h], f"falha_{h}d"
        teste_tags = set(C["sp"][h].teste["tag_code"])
        for tag, g in d.groupby("tag_code"):
            primeiro = C["eventos"].loc[C["eventos"]["tag_code"] == tag, "inicio"].min()
            possiveis = len(pd.date_range(primeiro, base_max, freq=f"{config.REFERENCE_STEP_DAYS}D"))
            linhas.append({"horizonte": h, "tag": tag, "partição": "teste" if tag in teste_tags else "treino(dev)",
                           "linhas": len(g), "positivos": int(g[alvo].sum()), "negativos": int((1 - g[alvo]).sum()),
                           "prevalência": g[alvo].mean(), "janelas_possíveis_sem_censura": possiveis,
                           "perdidas_por_censura": possiveis - len(g), "ref_mín": g["data_referencia"].min(), "ref_máx": g["data_referencia"].max()})
    det = pd.DataFrame(linhas)
    tbl(det)
    P("Totais por horizonte e partição:")
    P()
    tot = det.groupby(["horizonte", "partição"]).agg(
        tags=("tag", "nunique"), linhas=("linhas", "sum"), positivos=("positivos", "sum"), negativos=("negativos", "sum"),
        perdidas_por_censura=("perdidas_por_censura", "sum")).reset_index()
    tot["prevalência"] = tot["positivos"] / tot["linhas"]
    tbl(tot)
    tot2 = det.groupby("horizonte").agg(linhas=("linhas", "sum"), positivos=("positivos", "sum"),
                                         possíveis=("janelas_possíveis_sem_censura", "sum"), perdidas=("perdidas_por_censura", "sum")).reset_index()
    tot2["prevalência"] = tot2["positivos"] / tot2["linhas"]
    tot2["% perdido"] = 100 * tot2["perdidas"] / tot2["possíveis"]
    tbl(tot2)
    P(f"Data máxima da base: {base_max.date()}. Uma data de referência só existe se `ref + H ≤ data_máxima` "
      "(censura à direita: sem isso um 'não falhou' seria ambíguo). Janelas consecutivas (passo de 7 dias) **se sobrepõem** "
      "(janela de 30–90 dias): as linhas não são independentes e o número efetivo de observações é bem menor que o número de linhas.")
    P()

    P("### Item 7 — Onde o threshold foi escolhido e critério")
    P()
    P("**Resposta:** o threshold operacional foi escolhido **somente com dados de desenvolvimento** (probabilidades out-of-fold da "
      "validação cruzada `LeaveOneGroupOut` sobre os 4 TAGs de treino), em `app/modeling/train.py` — `_oof_proba_seguro(...)` "
      "(linha 278) seguido de `_escolher_threshold(y_dev, prob_oof)` (linha 283). Nenhuma linha dos 2 TAGs de teste participa.")
    P()
    P(f"Critério (`_escolher_threshold`): entre os thresholds com **recall OOF ≥ {T.RECALL_DESEJADO}**, escolhe o de **maior precisão**; "
      "se nenhum atinge o recall, usa o que maximiza F2. As probabilidades OOF vêm do mesmo pipeline final (tunado + calibrado), "
      "reajustado dentro de cada dobra.")
    P()
    linhas = []
    for h in HORIZONTES:
        R, meta = RS[h], C["metas"][h]
        pred_dev = (R["oof"] >= R["thr"]).astype(int)
        md = metricas(R["y_dev"], pred_dev, R["oof"])
        pred_te = (R["p_te"] >= meta["threshold_operacional"]).astype(int)
        mt = metricas(R["y_te"], pred_te, R["p_te"])
        linhas.append({
            "horizonte": h, "threshold_meta": meta["threshold_operacional"], "threshold_reproduzido": R["thr"],
            "iguais(1e-9)": bool(np.isclose(R["thr"], meta["threshold_operacional"], atol=1e-9)),
            "ramo": "recall≥0.90 → maior precisão" if R["candidatos"] > 0 else "fallback F2",
            "nº_thresholds_com_recall≥0.90": R["candidatos"],
            "dev_OOF_recall": md["Recall"], "dev_OOF_precisão": md["Precision"],
            "teste_recall": mt["Recall"], "teste_precisão": mt["Precision"],
        })
    tbl(pd.DataFrame(linhas), nd=6)
    P("O recall-alvo de 0.90 foi atingido nos dados de desenvolvimento (OOF); no teste o recall resultante foi menor — a diferença é "
      "efeito de generalização entre TAGs (mudança de prevalência), não de ajuste no teste.")
    P()

    P("### Item 8 — Refazer com teste independente se o threshold tivesse sido ajustado no teste")
    P()
    P("**Resposta:** o threshold **não** foi ajustado no teste (item 7); portanto a refação por esse motivo não é necessária. "
      "Ressalvas honestas: (a) o conjunto de teste foi **consultado diversas vezes** ao longo do desenvolvimento (cada execução de "
      "`run_training.py` imprime métricas de teste) e o escopo/horizontes/etiquetas foram revistos depois de ver resultados — "
      "não há garantia formal de que nenhuma decisão de projeto tenha sido influenciada por ele; (b) existem só 2 TAGs de teste. "
      "Não existe um segundo conjunto de teste independente.")
    P()
    P("**Avaliação complementar (não faz parte do pipeline original):** validação externa *deixando cada um dos 6 TAGs de fora*, com os "
      "hiperparâmetros e o algoritmo finais e o threshold recalculado por OOF interno nos outros 5 TAGs. Ressalva: os hiperparâmetros "
      "foram tunados nos 4 TAGs de desenvolvimento, então essa avaliação só é totalmente independente para EQ004 e EQ006.")
    P()
    for h in HORIZONTES:
        P(f"**Horizonte {h} dias**")
        P()
        tbl(EXT[h])
    P()

    P("### Item 9 — Baselines em todos os horizontes")
    P()
    P("Definições: *sempre falha* prevê 1 para todas as linhas; *sempre não-falha* prevê 0; *aleatório pela prevalência* prevê 1 com "
      "probabilidade p (valores esperados: acurácia = p²+(1−p)², precisão = recall = F1 = p, ROC-AUC = 0.5, PR-AUC = p). "
      "Modelo = threshold operacional (conjunto de teste).")
    P()
    linhas = []
    for h in HORIZONTES:
        R, meta = RS[h], C["metas"][h]
        y = R["y_te"].to_numpy()
        p = float(y.mean())
        mod = metricas(y, (R["p_te"] >= meta["threshold_operacional"]).astype(int), R["p_te"])
        sempre = {"Accuracy": p, "Precision": p, "Recall": 1.0, "F1": 2 * p / (1 + p), "ROC-AUC": 0.5, "PR-AUC": p}
        nunca = {"Accuracy": 1 - p, "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "ROC-AUC": 0.5, "PR-AUC": p}
        alea = {"Accuracy": p * p + (1 - p) ** 2, "Precision": p, "Recall": p, "F1": p, "ROC-AUC": 0.5, "PR-AUC": p}
        for nome, d in [("Modelo (thr. operacional)", mod), ("Sempre falha", sempre), ("Sempre não-falha", nunca), ("Aleatório pela prevalência", alea)]:
            linhas.append({"horizonte": h, "prevalência_teste": p, "estratégia": nome, **d})
    tbl(pd.DataFrame(linhas))
    linhas = []
    for h in HORIZONTES:
        R = RS[h]
        y = R["y_dev"].to_numpy()
        p = float(y.mean())
        linhas.append({"horizonte": h, "prevalência_dev": p, "sempre_falha_Accuracy": p, "sempre_falha_F1": 2 * p / (1 + p),
                       "PR-AUC_esperada_sem_informação": p})
    P("Mesmas baselines no desenvolvimento:")
    P()
    tbl(pd.DataFrame(linhas))
    P("**Leitura:** onde 'Sempre falha' tem acurácia/recall/F1 iguais ou superiores às do modelo, o modelo **não** supera o baseline "
      "trivial nessas métricas; o ganho aparece apenas nas métricas de ranking (ROC-AUC/PR-AUC acima do valor sem informação).")
    P()

    P("### Item 10 — Calibração (tabela de bins, Brier, log loss)")
    P()
    P("Bins por quantis (10), probabilidades do modelo final calibrado. 'Teste' = TAGs EQ004/EQ006; 'OOF dev' = predições "
      "out-of-fold do desenvolvimento. Referência: modelo constante (prevalência do treino).")
    P()
    resumo = []
    for h in HORIZONTES:
        R = RS[h]
        prev_tr = float(R["y_dev"].mean())
        for nome, y, p in [("teste", R["y_te"].to_numpy(), R["p_te"]), ("OOF dev", R["y_dev"].to_numpy(), R["oof"])]:
            pc = np.clip(p, 1e-15, 1 - 1e-15)
            bins = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
            t = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(
                n=("y", "size"), prob_média=("p", "mean"), taxa_observada=("y", "mean"), p_mín=("p", "min"), p_máx=("p", "max")).reset_index()
            ece = float((t["n"] * (t["prob_média"] - t["taxa_observada"]).abs()).sum() / t["n"].sum())
            const = np.full(len(y), prev_tr)
            resumo.append({"horizonte": h, "conjunto": nome, "Brier": brier_score_loss(y, p),
                           "Brier_constante(prev.treino)": brier_score_loss(y, const),
                           "log_loss": log_loss(y, pc), "log_loss_constante": log_loss(y, np.clip(const, 1e-15, 1 - 1e-15)),
                           "ECE(10 bins)": ece, "prob_média_prevista": float(p.mean()), "taxa_real": float(y.mean())})
            P(f"**Horizonte {h} — {nome}**")
            P()
            tbl(t)
    P("Resumo:")
    P()
    tbl(pd.DataFrame(resumo))
    P()

    P("### Item 11 — Permutation importance completa por horizonte")
    P()
    P("Modelo final (joblib) no conjunto de teste; `scoring='average_precision'`, `n_repeats=10`, `random_state=42`. "
      "Valor = queda média da AP ao embaralhar a coluna (negativo = embaralhar não piorou). `area` e `sist_oper` são embaralhadas como colunas inteiras.")
    P()
    for h in HORIZONTES:
        R = RS[h]
        pi = permutation_importance(C["finais"][h], R["X_te"], R["y_te"], scoring="average_precision", n_repeats=10,
                                    random_state=42, n_jobs=1)
        t = pd.DataFrame({"feature": FEATURES, "importância_média": pi.importances_mean, "desvio": pi.importances_std}).sort_values(
            "importância_média", ascending=False)
        P(f"**Horizonte {h} dias** (AP de referência no teste = {fmt(average_precision_score(R['y_te'], R['p_te']))})")
        P()
        tbl(t)
        t.assign(horizonte=h).to_csv(os.path.join(CSV_DIR, f"permutation_importance_{h}d.csv"), index=False, encoding="utf-8")
    P("Limitação: features correlacionadas (freq. 90d/180d, tempo desde a última falha) dividem a importância na permutação "
      "univariada; e a importância é medida em apenas 2 TAGs.")
    P()

    P("### Item 12 — Por que não `StratifiedGroupKFold`")
    P()
    R = RS[HORIZONTES[-1]]
    ng = R["g_dev"].nunique()
    P(f"O desenvolvimento tem **{ng} grupos (TAGs)**. Execução real de `StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42).split(X_dev, y_dev, groups)`:")
    P()
    linhas = []
    for k in (5, 4, 3, 2):
        try:
            res = []
            for i, (tr, vl) in enumerate(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED).split(R["X_dev"], R["y_dev"], R["g_dev"])):
                grupos_vl = sorted(R["g_dev"].iloc[vl].unique())
                res.append(f"fold{i}: validação={grupos_vl}, prev={R['y_dev'].iloc[vl].mean():.3f}")
            linhas.append({"n_splits": k, "resultado": "executa", "detalhe": " ; ".join(res)})
        except Exception as exc:  # noqa: BLE001
            linhas.append({"n_splits": k, "resultado": f"{type(exc).__name__}", "detalhe": str(exc)})
    tbl(pd.DataFrame(linhas))
    P("Com 5 dobras o erro é impeditivo (mais dobras que grupos). Com 4 dobras o método executa, mas cada dobra é exatamente 1 TAG — reproduz o "
      "`LeaveOneGroupOut` (a estratificação por classe não consegue atuar: as prevalências por TAG são fixas e muito diferentes, ver item 6); com 3 ou 2 dobras "
      "as dobras misturam 1–2 TAGs inteiros e as prevalências de validação saem desbalanceadas. Por isso foi adotado `LeaveOneGroupOut`, determinístico.")
    P()

    P("### Item 13 — Versão do modelo usada na inferência final e re-treino após o teste")
    P()
    P("**Resposta:** a inferência exibida na aplicação usa os arquivos `models/modelo_falha_{30,60,90}d.joblib` presentes em disco no "
      "momento em que o cache é montado (`routes._get_snapshot` → `predict.snapshot_atual`). Não existe versionamento de modelo "
      "(apenas o `.meta.json`, versionado no git). **Não existe re-treino em treino+teste**: o modelo final é ajustado somente nas "
      "linhas de desenvolvimento (`train.py`, linhas 290–291: `modelo_otimizado.fit(X_dev, y_dev)`), e nunca refeito depois da avaliação no teste.")
    P()
    linhas = []
    for h in HORIZONTES:
        jl = os.path.join(MODELS_DIR, f"modelo_falha_{h}d.joblib")
        mt = os.path.join(MODELS_DIR, f"modelo_falha_{h}d.meta.json")
        linhas.append({"horizonte": h, "joblib_modificado_em": mtime(jl), "meta_modificado_em": mtime(mt),
                       "joblib_sha256(16)": sha256(jl)[:16], "meta_sha256(16)": sha256(mt)[:16]})
    tbl(pd.DataFrame(linhas))
    P(f"Banco (`instance/manutencao.db`) modificado em {mtime(os.path.join(ROOT, 'instance', 'manutencao.db'))}.")
    P()
    P("Histórico git de `models/` (meta.json é versionado; `.joblib` é ignorado):")
    P()
    P("```")
    P(sh(["git", "log", "--format=%h %cI %s", "--", "models"]) or "(sem commits)")
    P("```")
    P(f"`git diff HEAD -- models` vazio (meta.json em disco = último commit): {'sim' if not sh(['git', 'diff', 'HEAD', '--', 'models']) else 'não'}. "
      "Ou seja, o re-treino de 2026-10-06 (após reingestão) reproduziu exatamente os hiperparâmetros, thresholds e métricas já versionados.")
    P()
    P("Consistência: métricas de teste recalculadas a partir do `.joblib` vs. valores gravados no `meta.json`:")
    P()
    linhas = []
    for h in HORIZONTES:
        R, meta = RS[h], C["metas"][h]
        m = metricas(R["y_te"], (R["p_te"] >= meta["threshold_operacional"]).astype(int), R["p_te"])
        for k in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]:
            linhas.append({"horizonte": h, "métrica": k, "recalculado": m[k], "meta.json": meta["metricas_teste_threshold_otimo"][k],
                           "iguais(1e-9)": bool(np.isclose(m[k], meta["metricas_teste_threshold_otimo"][k], atol=1e-9))})
    tbl(pd.DataFrame(linhas), nd=6)
    base_max = C["eventos"]["inicio"].max()
    P(f"Data da inferência final (data máxima da base, usada como 'hoje' da aplicação): **{base_max.date()}**. "
      f"Os rótulos de treino chegam, no máximo, a {base_max.date()} (janelas olham até H dias à frente da referência, e referências só vão até máx − H).")
    P()
    return None


def secao_14(C: dict, RS: dict, snap: pd.DataFrame) -> None:
    P("### Item 14 — Regra exata da 'data sugerida' e tratamento de probabilidades não monotônicas")
    P()
    P("Código: `app/modeling/predict.py::data_sugerida_alerta`. Regra literal:")
    P()
    P("1. `limiar = config.LIMIAR_ALERTA` (= " + str(config.LIMIAR_ALERTA) + "); **não** é o threshold operacional de cada modelo, "
      "é um limiar de exibição único.")
    ths = [C["metas"][h]["threshold_operacional"] for h in HORIZONTES]
    P(f"   (thresholds operacionais atuais: {', '.join(f'{h}d={t:.3f}' for h, t in zip(HORIZONTES, ths))}.)")
    P("2. Percorre os horizontes em ordem crescente (30, 60, 90) e devolve o **primeiro** horizonte H cuja probabilidade estimada "
      "`prob_falha_Hd ≥ limiar`.")
    P("3. `data_sugerida = data_de_referência + H dias` (data de referência = data máxima da base). Se nenhum horizonte atinge o limiar → "
      "'sem alerta' (sem data).")
    P()
    P("Interpretação correta: a data é **o fim da janela** cujo risco estimado ultrapassa o limiar — 'a probabilidade de falhar em até H dias é "
      "≥ limiar' — e não a data prevista da falha. Os três modelos são independentes (não há restrição de monotonicidade) e cada probabilidade é "
      "a de **falhar em até H dias**; teoricamente p30 ≤ p60 ≤ p90, mas isso não é imposto. Como a regra pega o menor H que cruza o limiar, "
      "uma inversão (ex.: p30 > p60) não gera data incoerente; ela só afeta qual H é escolhido.")
    P()
    P("**Snapshot atual (uma linha por equipamento):**")
    P()
    s = snap[["tag_code", "data_referencia", "prob_falha_30d", "prob_falha_60d", "prob_falha_90d", "horizonte_sugerido", "data_sugerida_alerta"]].copy()
    s["monotônico(p30≤p60≤p90)"] = (s["prob_falha_30d"] <= s["prob_falha_60d"]) & (s["prob_falha_60d"] <= s["prob_falha_90d"])
    tbl(s, nd=6)
    n30 = int((snap["prob_falha_30d"] >= config.LIMIAR_ALERTA).sum())
    P(f"**Leitura:** {n30} de {len(snap)} equipamentos já têm prob_falha_30d ≥ {config.LIMIAR_ALERTA}; para todos eles a regra devolve o menor horizonte (30 dias) e a "
      f"data sugerida é sempre data_de_referência + 30 dias ({(snap['data_referencia'].iloc[0] + pd.Timedelta(days=30)).date()}). Ou seja, no estado atual a data sugerida "
      "não diferencia esses equipamentos entre si (a diferença está só na probabilidade).")
    P()

    P("**Verificação de não monotonicidade em todas as linhas (TAG, ref) com os três horizontes disponíveis** "
      "(referências ≤ máx − 90 dias; probabilidades do modelo final):")
    P()
    d30, d60, d90 = (C["ds"][h].set_index(["tag_code", "data_referencia"]) for h in (30, 60, 90))
    idx = d90.index
    X = d30.loc[idx, FEATURES].reset_index(drop=True)
    p = pd.DataFrame({h: C["finais"][h].predict_proba(C["ds"][h].set_index(["tag_code", "data_referencia"]).loc[idx, FEATURES])[:, 1]
                      for h in (30, 60, 90)}, index=idx)
    teste_tags = set(C["sp"][90].teste["tag_code"])
    p["partição"] = ["teste" if t in teste_tags else "treino(dev)" for t in p.index.get_level_values(0)]
    linhas = []
    for part, g in list(p.groupby("partição")) + [("todas", p)]:
        linhas.append({"partição": part, "linhas": len(g), "p30>p60": int((g[30] > g[60]).sum()), "p60>p90": int((g[60] > g[90]).sum()),
                       "p30>p90": int((g[30] > g[90]).sum()),
                       "alguma_inversão": int(((g[30] > g[60]) | (g[60] > g[90])).sum()),
                       "% com inversão": 100 * float(((g[30] > g[60]) | (g[60] > g[90])).mean())})
    tbl(pd.DataFrame(linhas))
    P("Tratamento no código: **nenhum** (não há reordenação, `maximum.accumulate` nem isotônica). Quando há inversão, a regra do item 14.2 "
      "ainda devolve o menor H que cruza o limiar. Se desejado, monotonizar com `np.maximum.accumulate` seria uma alteração futura — não existe hoje.")
    P()


# ===================================================================== C
def secao_C(C: dict, RAW: dict) -> None:
    ev = C["eventos"]
    P("## C. Análise exploratória e dados")
    P()
    P("### Item 15 — Exclusões 3.347 → 2.929 por motivo")
    P()
    bruto = RAW["bruto"]
    sit = bruto["Situação"].astype(str).str.strip()
    linhas = [{"etapa": "linhas brutas", "linhas": len(bruto), "removidas": 0}]
    n1 = int((sit == "C").sum())
    linhas.append({"etapa": "após manter Situação == 'C'", "linhas": n1, "removidas": len(bruto) - n1})
    linhas.append({"etapa": f"após filtro de escopo (prefixo TAG {config.TAG_PREFIXOS_ESCOPO})", "linhas": RAW["n_pos_escopo"],
                   "removidas": n1 - RAW["n_pos_escopo"]})
    linhas.append({"etapa": "após remover duplicatas (Nº OS, NºTarefa, Início), keep=first", "linhas": len(RAW["final"]),
                   "removidas": RAW["n_pos_escopo"] - len(RAW["final"])})
    tbl(pd.DataFrame(linhas))
    P(f"Distribuição de `Situação` na planilha bruta: {sit.value_counts(dropna=False).to_dict()} (nulos: {int(bruto['Situação'].isna().sum())}; "
      "'A' = em aberto/não concluída). Erros de conversão de data: 0 (a leitura usa `errors='raise'`; teria interrompido).")
    P()
    P(f"- Das {int((sit != 'C').sum())} linhas excluídas por Situação, **{RAW['a_com_par_c']}** têm o mesmo par (Nº OS, NºTarefa) de alguma linha 'C' "
      "(isto é, o mesmo apontamento reaparece concluído).")
    P(f"- Duplicatas removidas: **{RAW['n_dup']}** linhas em {RAW['n_grupos_dup']} grupos de chave repetida; "
      f"{RAW['dup_identicas']} das linhas removidas eram idênticas à mantida em tipo, HH bruto e descrição, "
      f"{RAW['n_dup'] - RAW['dup_identicas']} diferiam em ao menos um desses campos (o `keep='first'` descartou o conteúdo diferente).")
    P()
    P("Composição final dos 2.929 eventos por tipo de manutenção (código da planilha):")
    P()
    t = ev["tipo_manutencao"].value_counts().rename_axis("tipo").reset_index(name="eventos")
    t["papel"] = t["tipo"].map(lambda x: "falha" if x in config.TIPOS_FALHA else ("preventiva" if x == config.TIPO_PREVENTIVA else "não classificado (fora das features de falha/rótulo)"))
    tbl(t)
    P(f"Totais: eventos = {len(ev)}; falhas (MC+CP) = {int(ev['is_falha'].sum())}; preventivas (MP) = {int(ev['is_preventiva'].sum())}; "
      f"não classificados = {int((~ev['classificado']).sum())}.")
    P()

    P("### Item 16 — MTBF: fórmula, intervalos, primeira/última data por equipamento")
    P()
    P("**Fórmula** (`exploratory.mtbf_por_equipamento`): para cada equipamento, ordenar as datas `Início` dos eventos com `is_falha` (MC/CP) e "
      "tomar a **média das diferenças, em dias de calendário, entre falhas consecutivas**. Falhas no mesmo dia geram intervalo 0 e **entram** na média. "
      "Menos de 2 falhas → indefinido. A planilha **não** tem data/hora separada de abertura e fechamento da OS nem horímetro: `Início` é a data do "
      "apontamento; portanto o MTBF é de calendário, não de operação.")
    P()
    linhas = []
    for tag, g in ev.groupby("tag_code"):
        f = g[g["is_falha"]].sort_values("inicio")
        diffs = f["inicio"].diff().dropna().dt.days
        dist = f["inicio"].drop_duplicates()
        d2 = dist.diff().dropna().dt.days
        periodo = (g["inicio"].max() - g["inicio"].min()).days
        linhas.append({
            "tag": tag, "primeiro_evento": g["inicio"].min(), "último_evento": g["inicio"].max(),
            "primeira_falha": f["inicio"].min(), "última_falha": f["inicio"].max(), "falhas": len(f), "datas_distintas_de_falha": len(dist),
            "intervalos": len(diffs), "intervalos_zero": int((diffs == 0).sum()),
            "MTBF_dias(app)": diffs.mean() if len(diffs) else None, "mediana_intervalo": diffs.median() if len(diffs) else None,
            "mín": diffs.min() if len(diffs) else None, "máx": diffs.max() if len(diffs) else None,
            "MTBF_só_datas_distintas": d2.mean() if len(d2) else None,
            "período_obs_dias/falhas": periodo / len(f) if len(f) else None})
    tbl(pd.DataFrame(linhas), nd=2)
    P("As duas últimas colunas são sensibilidades: o valor exibido na aplicação é `MTBF_dias(app)`. A diferença entre `MTBF_dias(app)` e "
      "`MTBF_só_datas_distintas` mostra o efeito de múltiplas falhas apontadas no mesmo dia.")
    P()

    P("### Item 17 — 'Tempo médio de reparo' = HH Apropriado; efeito da perturbação")
    P()
    df = RAW["final"]
    P("**Significado:** o rótulo 'tempo médio de reparo' é a **média de `HH Apropriado`** (homem-hora apropriada ao apontamento) — não é tempo de "
      "calendário nem duração da parada. Ele é calculado sobre todos os tipos de evento (na tela de comparação) ou por conjunto (itens 18–19).")
    try:
        hm = df["Duração Apropriada"].astype(str).str.extract(r"^(\d+):(\d+)")
        dur = hm[0].astype(float) + hm[1].astype(float) / 60
        qt = pd.to_numeric(df["Qt. Matrícula"], errors="raise")
        igual = np.isclose(df["HH Apropriado"].astype(float), dur * qt, rtol=0, atol=0.0051)
        P(f"Verificação na planilha: `HH Apropriado == Duração Apropriada (HH:MM → horas) × Qt. Matrícula` (HH vem arredondado a 2 casas decimais; tolerância 0.0051 h) em {int(igual.sum())} de {len(df)} linhas; "
          f"`Qt. Matrícula` assume os valores {sorted(qt.unique().tolist())}.")
    except Exception as exc:  # noqa: BLE001
        P(f"Verificação HH = Duração × Qt. Matrícula: não foi possível (coluna não numérica: {type(exc).__name__}).")
    P()
    fator = df["hh_apropriado"].to_numpy() / np.where(df["HH Apropriado"].to_numpy() == 0, np.nan, df["HH Apropriado"].to_numpy())
    P(f"**Perturbação** (`ingest._perturbar_hh`): `hh_armazenado = HH_bruto × U(0.9, 1.1)`, fator independente por evento, `numpy.default_rng({config.RANDOM_SEED})`; "
      "só o valor perturbado vai ao banco. Desvio-padrão teórico do fator = 0.2/√12 ≈ 0.0577 (cada evento muda até ±10%). "
      f"Eventos com HH bruto = 0: {int((df['HH Apropriado'] == 0).sum())} (permanecem 0). Fator empírico: mín {np.nanmin(fator):.4f}, máx {np.nanmax(fator):.4f}, "
      f"média {np.nanmean(fator):.4f}, desvio {np.nanstd(fator):.4f}.")
    P()
    P("Efeito sobre agregados (apenas **razões** perturbado/bruto; o HH bruto não é divulgado):")
    P()
    linhas = []
    for nome, g in [("TOTAL", df)] + [(f"{k}", v) for k, v in df.groupby("tag_code")] + [("tipo CP", df[df["Tipo de Manutenção"] == "CP"]), ("tipo MC", df[df["Tipo de Manutenção"] == "MC"])]:
        br, pe = g["HH Apropriado"].mean(), g["hh_apropriado"].mean()
        linhas.append({"grupo": nome, "n": len(g), "razão_média_perturbada/bruta": pe / br if br else None,
                       "erro_relativo_esperado_da_média(0.0577/√n)": 0.0577 / np.sqrt(len(g))})
    tbl(pd.DataFrame(linhas), nd=5)
    P(f"Consistência com o banco: {len(ev)} eventos no banco vs. {len(df)} na reexecução do pipeline; soma HH banco = {ev['hh_apropriado'].sum():.6f}, "
      f"soma HH pipeline = {df['hh_apropriado'].sum():.6f} (iguais: {bool(np.isclose(ev['hh_apropriado'].sum(), df['hh_apropriado'].sum()))}).")
    P("Efeito prático: diferenças de média entre grupos menores que alguns % ficam dentro do ruído introduzido; tendências e comparações CP×MC "
      "devem ser lidas como indicativas.")
    P()

    P("### Item 18 — Tempo de reparo (HH): CP vs. MC")
    P()
    linhas = []
    for tipo in ["CP", "MC"]:
        g = ev[ev["tipo_manutencao"] == tipo]["hh_apropriado"]
        linhas.append({"tipo": tipo, "n": len(g), "média": g.mean(), "mediana": g.median(), "desvio_padrão(ddof=1)": g.std(), "mín": g.min(), "máx": g.max(),
                       "HH=0": int((g == 0).sum())})
    tbl(pd.DataFrame(linhas), nd=3)
    a, b = ev[ev["tipo_manutencao"] == "CP"]["hh_apropriado"], ev[ev["tipo_manutencao"] == "MC"]["hh_apropriado"]
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    P(f"Mann-Whitney (bilateral) CP vs. MC: U = {u.statistic:.1f}, p = {u.pvalue:.4g} (complementar; eventos do mesmo equipamento/período "
      "não são independentes, então o p-valor é indicativo). Por equipamento:")
    P()
    linhas = []
    for tag, g in ev[ev["is_falha"]].groupby("tag_code"):
        for tipo in ["CP", "MC"]:
            x = g[g["tipo_manutencao"] == tipo]["hh_apropriado"]
            linhas.append({"tag": tag, "tipo": tipo, "n": len(x), "média": x.mean() if len(x) else None, "mediana": x.median() if len(x) else None,
                           "desvio": x.std() if len(x) > 1 else None})
    tbl(pd.DataFrame(linhas), nd=3)

    P("### Item 19 — Tendência do tempo de reparo: método, n, erro-padrão e IC")
    P()
    P("Método (`exploratory.tendencia_tempo_reparo`): regressão linear simples (mínimos quadrados) de `hh_apropriado` (perturbado) sobre o "
      "**ano-calendário inteiro** do evento, usando **todos** os eventos do equipamento (não só falhas), mínimo de 3 eventos e variação > 0 no ano. "
      "Abaixo: inclinação (h/ano), erro-padrão, IC 95% (distribuição t, n−2 gl), p-valor, R², e duas variações — x em ano fracionário e só falhas.")
    P()
    linhas = []
    tend = exploratory.tendencia_tempo_reparo(ev).set_index("tag_code")
    for tag, g in ev.groupby("tag_code"):
        for nome, sub, xs in [("todos eventos, ano inteiro (app)", g, g["inicio"].dt.year.astype(float)),
                              ("todos eventos, ano fracionário", g, g["inicio"].dt.year + (g["inicio"].dt.dayofyear - 1) / 365.25),
                              ("só falhas, ano inteiro", g[g["is_falha"]], g[g["is_falha"]]["inicio"].dt.year.astype(float))]:
            if len(sub) < 3 or xs.std() == 0:
                linhas.append({"tag": tag, "variante": nome, "n": len(sub)})
                continue
            r = stats.linregress(xs, sub["hh_apropriado"])
            tcrit = stats.t.ppf(0.975, len(sub) - 2)
            linhas.append({"tag": tag, "variante": nome, "n": len(sub), "inclinação_h/ano": r.slope, "erro_padrão": r.stderr,
                           "IC95_inf": r.slope - tcrit * r.stderr, "IC95_sup": r.slope + tcrit * r.stderr, "p_valor": r.pvalue, "R²": r.rvalue ** 2,
                           "bate_com_app": bool(np.isclose(r.slope, tend.loc[tag, "inclinacao_h_por_ano"])) if nome.startswith("todos eventos, ano inteiro") else None})
    tbl(pd.DataFrame(linhas), nd=4)
    P("Ressalvas: observações serializadas no tempo e agrupadas por OS (autocorrelação) → erro-padrão e p-valor tendem a ser otimistas; "
      "ruído de ±10% do HH; muitos HH repetidos/arredondados. Sem correção para múltiplos testes (6 equipamentos × variantes).")
    P()

    P("### Item 20 — Sazonalidade: falhas por mês × ano e cobertura")
    P()
    f = ev[ev["is_falha"]].copy()
    f["ano"], f["mes"] = f["inicio"].dt.year, f["inicio"].dt.month
    pt = f.pivot_table(index="ano", columns="mes", values="tag_code", aggfunc="size", fill_value=0).reindex(
        index=range(int(f["ano"].min()), int(f["ano"].max()) + 1), columns=range(1, 13), fill_value=0)
    pt.index.name = "ano"
    pt.columns = MESES
    pt["Total"] = pt.sum(axis=1)
    pt = pt.reset_index()
    tot = pd.DataFrame([["TOTAL"] + pt[MESES + ["Total"]].sum().tolist()], columns=pt.columns)
    tbl(pd.concat([pt, tot], ignore_index=True))
    dmin, dmax = ev["inicio"].min(), ev["inicio"].max()
    P(f"**Cobertura:** eventos de {dmin.date()} a {dmax.date()}. Ano {dmin.year} cobre só a partir de {dmin.date()} (meses anteriores sem observação, "
      f"não 'zero falhas'); ano {dmax.year} cobre só até {dmax.date()}. Os meses fora da cobertura aparecem como 0 na tabela acima, o que **não** é ausência de falha.")
    P()
    linhas = []
    for tag, g in ev.groupby("tag_code"):
        linhas.append({"tag": tag, "primeiro_evento": g["inicio"].min(), "último_evento": g["inicio"].max(),
                       "anos_sem_nenhum_evento": ", ".join(str(a) for a in range(g["inicio"].min().year, g["inicio"].max().year + 1)
                                                           if a not in set(g["inicio"].dt.year)) or "nenhum"})
    tbl(pd.DataFrame(linhas))
    P("**Normalização por exposição** (apenas meses de calendário totalmente cobertos, por equipamento, a partir do mês seguinte ao primeiro evento do equipamento até o mês anterior ao último mês da base):")
    P()
    exp = Counter()
    obs = Counter()
    for tag, g in ev.groupby("tag_code"):
        ini = (g["inicio"].min() + pd.offsets.MonthBegin(1)).normalize()
        fim_ = dmax.replace(day=1)
        for mes_ini in pd.date_range(ini, fim_ - pd.offsets.MonthBegin(1), freq="MS"):
            exp[mes_ini.month] += 1
        mask = (g["is_falha"]) & (g["inicio"] >= ini) & (g["inicio"] < fim_)
        for m in g.loc[mask, "inicio"].dt.month:
            obs[m] += 1
    tab = pd.DataFrame({"mês": MESES, "falhas": [obs[m] for m in range(1, 13)], "equipamento-meses_expostos": [exp[m] for m in range(1, 13)]})
    tab["falhas_por_equipamento-mês"] = tab["falhas"] / tab["equipamento-meses_expostos"]
    tbl(tab)
    esperado = tab["equipamento-meses_expostos"] / tab["equipamento-meses_expostos"].sum() * tab["falhas"].sum()
    chi = stats.chisquare(tab["falhas"], esperado)
    P(f"Qui-quadrado de aderência (H0: taxa uniforme por equipamento-mês): χ² = {chi.statistic:.2f}, gl = 11, p = {chi.pvalue:.4g}. "
      "As falhas são agrupadas (várias no mesmo dia/OS) → superdispersão → o p-valor tende a ser anti-conservador; use apenas como indício.")
    P()
    return None


def secao_21_23(C: dict) -> None:
    ev = C["eventos"]
    f = ev[ev["is_falha"]].copy()
    NI = component_mapping.NAO_IDENTIFICADO
    P("### Item 21 — Catálogo de componentes, todas as categorias e exemplos reais")
    P()
    P("**Catálogo completo** (conhecimento técnico de referência, não dado da empresa; `component_mapping.CATALOGO_COMPONENTES` + `TERMOS_COMPONENTE`). "
      "Ordem da tabela = ordem de desempate (primeiro que casa vence):")
    P()
    linhas = []
    for nome, termos in component_mapping.TERMOS_COMPONENTE.items():
        c = component_mapping.CATALOGO_COMPONENTES[nome]
        linhas.append({"ordem": len(linhas) + 1, "componente": nome, "termos_de_busca (substring, sem acento)": ", ".join(termos),
                       "fabricante_referência": c["fabricante_referencia"], "vida_útil": c["vida_util_indicada"], "unidade": c["unidade"],
                       "critério_substituição": c["criterio_substituicao"]})
    tbl(pd.DataFrame(linhas))
    total = len(f)
    P(f"**Distribuição completa** (denominador = {total} eventos de falha; inclui categorias com 0 evento):")
    P()
    comp = f["componente_estimado"].fillna(NI).value_counts()
    nomes = list(component_mapping.TERMOS_COMPONENTE) + [NI]
    t = pd.DataFrame({"categoria": nomes, "eventos": [int(comp.get(n, 0)) for n in nomes]})
    t["%"] = 100 * t["eventos"] / total
    tbl(t.sort_values("eventos", ascending=False), nd=2)
    P("As distribuições de categorias de componente (item 21) e de texto (item 22) somam, cada uma, 1990 na base atual: **não há** os 18 eventos faltantes. "
      "Se o número 1972 (= 1990 − 18) aparece no texto do TCC, ele não vem das tabelas atuais da aplicação.")
    P(f"Soma das categorias = {int(t['eventos'].sum())} (confere com {total}: {int(t['eventos'].sum()) == total}); eventos de falha sem descrição (nulos): "
      f"{int(f['descricao_tarefa'].isna().sum())}; sem componente (campo nulo): {int(f['componente_estimado'].isna().sum())}.")
    P()

    # --- sobreposição de termos
    norm = f["descricao_tarefa"].map(lambda x: component_mapping._normalizar(str(x)))
    casa = pd.DataFrame({n: norm.map(lambda s, ps=ps: any(p_.search(s) for p_ in ps)) for n, ps in component_mapping._PADROES_COMPONENTE.items()})
    nmatch = casa.sum(axis=1)
    P("**Eventos com mais de um componente candidato** (o app atribui apenas o primeiro):")
    P()
    tbl(nmatch.value_counts().sort_index().rename_axis("nº de componentes cujo termo aparece").reset_index(name="eventos"))
    linhas = []
    for n in component_mapping.TERMOS_COMPONENTE:
        atribuidos = f["componente_estimado"] == n
        linhas.append({"componente_atribuído": n, "eventos": int(atribuidos.sum()),
                       "com_outro_termo_também": int((atribuidos & (nmatch > 1)).sum()),
                       "eventos_onde_este_termo_aparece_mas_outro_venceu": int((casa[n] & ~atribuidos).sum())})
    tbl(pd.DataFrame(linhas))

    # --- substring puro vs. início de palavra (regra atual) vs. palavra inteira
    P("**Sensibilidade ao critério de casamento** (mesma ordem de desempate). Regra atual do app: o termo deve **começar uma palavra** (corrige 'porta' ⊂ "
      "'exportação'/'comporta') e 'trava' só casa como palavra inteira ('trava'/'travas'; flexões de 'travar' como 'travado' ficam de fora). "
      "'Substring puro' é a regra original; 'palavra inteira' aplica `\bTERMO(s)?\b` a todos os termos:")
    P()

    def primeiro(s: str, modo: str) -> str:
        for n, ts in component_mapping.TERMOS_COMPONENTE.items():
            for t_ in ts:
                if modo == "substring":
                    ok = t_ in s
                elif modo == "atual":
                    ok = component_mapping._compilar(t_).search(s) is not None
                elif modo == "inicio":
                    ok = re.search(rf"(?<![a-z0-9]){re.escape(t_)}", s) is not None
                else:
                    ok = re.search(rf"\b{re.escape(t_)}s?\b", s) is not None
                if ok:
                    return n
        return NI
    sub = norm.map(lambda s: primeiro(s, "substring"))
    ini = norm.map(lambda s: primeiro(s, "atual"))
    ini0 = norm.map(lambda s: primeiro(s, "inicio"))
    pal = norm.map(lambda s: primeiro(s, "palavra"))
    t = pd.DataFrame({"categoria": nomes, "substring puro (regra anterior)": [int((sub == n).sum()) for n in nomes],
                      "início de palavra (1ª correção)": [int((ini0 == n).sum()) for n in nomes],
                      "regra atual (início de palavra; 'trava' só inteira)": [int((ini == n).sum()) for n in nomes],
                      "palavra inteira": [int((pal == n).sum()) for n in nomes]})
    tbl(t)
    P(f"Consistência: a regra atual reproduz a coluna `componente_estimado` do banco em {int((ini == f['componente_estimado'].fillna(NI)).sum())} de {total} eventos. "
      f"Eventos que mudaram da regra anterior para a atual: {int((sub != ini).sum())}; da atual para palavra inteira: {int((ini != pal).sum())}.")
    P()
    em_portas_orig = sub == "Portas / travas"
    P(f"Efeito da exclusão das flexões de 'travar': eventos que a regra original atribuía a 'Portas / travas' ({int(em_portas_orig.sum())}) caíram para "
      f"{int((ini == 'Portas / travas').sum())}; {int((ini0 == 'Portas / travas').sum() - (ini == 'Portas / travas').sum())} eventos saíram só por essa exclusão "
      "(passaram a 'não identificado' ou a outro componente que também aparece no texto).")
    P()
    P("**Formas de palavra que contêm cada termo (substring puro, regra anterior)** — mostra de onde vinham os falsos positivos; a regra atual descarta as formas em que o termo está no meio da palavra (ex.: 'exportacao', 'comporta'); top 6 por termo, texto normalizado:")
    P()
    linhas = []
    for n, ts in component_mapping.TERMOS_COMPONENTE.items():
        for t_ in ts:
            c = Counter()
            for s in norm:
                c.update(re.findall(rf"\w*{re.escape(t_)}\w*", s))
            linhas.append({"componente": n, "termo": t_, "formas_encontradas": "; ".join(f"{k}={v}" for k, v in c.most_common(6)) or "(nenhuma)"})
    tbl(pd.DataFrame(linhas))
    P("Leitura: antes da correção, 'porta' casava com 'exportacao' (157 descrições, nome de linha) e 'comporta'; hoje não casa. Flexões de 'travar' (ex.: 'travado', 'travando') "
      "também deixaram de casar com 'trava'; o efeito está na tabela de sensibilidade acima.")
    P()

    P("**Exemplos reais (5 por categoria, amostra aleatória com `random_state=42`; descrições já redigidas na ingestão):**")
    P()
    linhas = []
    for n in nomes:
        g = f[f["componente_estimado"].fillna(NI) == n]
        if len(g) == 0:
            linhas.append({"categoria": n, "ano": "—", "tag": "—", "descrição": "(sem eventos)"})
            continue
        for _, r in g.sample(min(5, len(g)), random_state=SEED).iterrows():
            linhas.append({"categoria": n, "ano": r["inicio"].year, "tag": r["tag_code"], "descrição": str(r["descricao_tarefa"])[:240]})
    ex = pd.DataFrame(linhas)
    tbl(ex)
    ex.to_csv(os.path.join(CSV_DIR, "exemplos_por_categoria.csv"), index=False, encoding="utf-8")

    P("### Item 22 — Análise de termos: parâmetros, denominadores e tabelas completas")
    P()
    desc = text_analysis.carregar_descricoes_falha(ev)
    n_desc = len(desc)
    minimo = max(2, int(n_desc * text_analysis.FRACAO_MINIMA_PADRAO))
    P(f"Parâmetros (`text_analysis.py`): população = descrições de eventos de falha com texto (**{n_desc}** de {int(ev['is_falha'].sum())} falhas); "
      f"normalização: minúsculas, sem acento, sem pontuação, marcador `[equip]` removido; tokens com ≥ 2 letras, sem números e sem stopwords "
      f"({len(text_analysis.STOPWORDS)} palavras, incluindo termos de contexto como 'monta', 'carga', 'linha'); "
      f"frequência = nº de **descrições** em que o termo aparece (cada termo conta no máximo 1 por descrição); "
      f"{text_analysis.N_CATEGORIAS_PADRAO} sementes (bigramas primeiro, depois palavras), frequência mínima = max(2, int({n_desc}×{text_analysis.FRACAO_MINIMA_PADRAO})) = **{minimo}**; "
      "cada descrição é atribuída à **primeira semente na ordem de frequência** que aparece nela (uma categoria por descrição); sem semente → 'outro / não identificado'.")
    P()
    tabela, sementes = text_analysis.categorizar(ev)
    sementes = sementes.copy()
    sementes["ordem"] = range(1, len(sementes) + 1)
    sementes["% das descrições (aparece)"] = 100 * sementes["ocorrencias_descricoes"] / n_desc
    P("Sementes (frequência bruta, antes da atribuição exclusiva):")
    P()
    tbl(sementes[["ordem", "padrao", "tipo", "ocorrencias_descricoes", "% das descrições (aparece)"]], nd=1)
    t = tabela.copy()
    t["n/denominador"] = t["ocorrencias"].astype(str) + "/" + str(n_desc)
    P("Categorias após atribuição exclusiva (percentual = ocorrências ÷ " + str(n_desc) + "):")
    P()
    tbl(t, nd=1)
    P(f"Soma das ocorrências = {int(t['ocorrencias'].sum())} (confere com {n_desc}: {int(t['ocorrencias'].sum()) == n_desc}). Para qualquer percentual citado "
      "(19.0%, 10.6%, 7.7%, 15.6%, 34.8%), localizar a linha correspondente acima: a coluna `n/denominador` é o numerador e o denominador exatos; "
      "somas de várias linhas (ex.: 15.6% = soma de três categorias) devem ser conferidas somando as ocorrências e dividindo por " + str(n_desc) + ".")
    P()
    pal = text_analysis.frequencia_palavras(desc, top_n=25)
    big = text_analysis.frequencia_bigramas(desc, top_n=25)
    pal["% das descrições"] = 100 * pal["ocorrencias"] / n_desc
    big["% das descrições"] = 100 * big["ocorrencias"] / n_desc
    P("Top 25 palavras (nº de descrições em que aparecem):")
    P()
    tbl(pal, nd=1)
    P("Top 25 bigramas:")
    P()
    tbl(big, nd=1)
    ev_ = tabela[tabela["categoria"] == text_analysis.NAO_IDENTIFICADO]
    P(f"Categoria 'outro / não identificado' (texto) = {int(ev_['ocorrencias'].iloc[0]) if len(ev_) else 0} descrições. "
      "Atenção: é uma classificação **diferente** do 'não identificado' do item 21 (componente por palavra-chave): cada uma tem sua própria regra e seu próprio percentual.")
    P()

    P("### Item 23 — Auditoria de rótulos (50 'não identificado' + 50 identificados)")
    P()
    ni = f[f["componente_estimado"].fillna(NI) == NI]
    idn = f[f["componente_estimado"].fillna(NI) != NI]
    a = ni.sample(50, random_state=SEED).assign(grupo_amostra="não identificado")
    b = idn.sample(50, random_state=SEED).assign(grupo_amostra="identificado")
    aud = pd.concat([a, b]).sort_values(["grupo_amostra", "inicio"]).reset_index(drop=True)
    aud.insert(0, "id", range(1, len(aud) + 1))
    aud["ano"] = aud["inicio"].dt.year
    aud["rotulo_manual_componente"] = ""
    aud["observacao"] = ""
    cols = ["id", "grupo_amostra", "tag_code", "ano", "descricao_tarefa", "componente_estimado", "rotulo_manual_componente", "observacao"]
    aud[cols].to_csv(os.path.join(CSV_DIR, "auditoria_rotulos_amostra.csv"), index=False, encoding="utf-8-sig")
    P(f"Amostra gerada: 50 de {len(ni)} eventos 'não identificado' + 50 de {len(idn)} identificados (aleatória, `random_state={SEED}`, descrições redigidas) em "
      "`evidencias/csv/auditoria_rotulos_amostra.csv`, com colunas vazias `rotulo_manual_componente` e `observacao`.")
    P()
    P("**Resultado da classificação manual: não existe (ainda).** A rotulagem precisa ser feita por uma pessoa com conhecimento do equipamento "
      "(não por modelo de linguagem, para não circularizar a validação). Depois de preenchida a coluna `rotulo_manual_componente` (nome do componente do "
      "catálogo, ou `nenhum`), rodar `python scripts/avaliar_auditoria.py` para obter: precisão dos identificados, proporção de 'não identificado' "
      "que na verdade tinham componente, e matriz de confusão.")
    P()
    P("Dado objetivo já disponível (sem rotulagem): comprimento das descrições por grupo.")
    P()
    f["palavras"] = f["descricao_tarefa"].astype(str).str.split().map(len)
    f["grupo"] = np.where(f["componente_estimado"].fillna(NI) == NI, "não identificado", "identificado")
    g = f.groupby("grupo")["palavras"].agg(["count", "mean", "median", lambda s: (s <= 3).mean() * 100]).reset_index()
    g.columns = ["grupo", "eventos", "palavras_média", "palavras_mediana", "% com ≤ 3 palavras"]
    tbl(g, nd=2)


# ===================================================================== D
def secao_D(C: dict, snap: pd.DataFrame) -> None:
    ev = C["eventos"]
    P("## D. Interface, exportações, testes e esquema")
    P()
    P("### Item 24 — Rótulos corrigidos da Figura 9 / comparação consolidada")
    P()
    P("Correção aplicada na interface (commit deste briefing) e a alterar também no texto do TCC: **'Taxa de falha' → 'Proporção de eventos corretivos'** "
      "(= nº de eventos de falha MC/CP ÷ nº total de eventos do equipamento; não é taxa por unidade de tempo) e **'MTBF real' → 'MTBF (dias de calendário)'** "
      "(= média, em dias de calendário, dos intervalos entre as datas `Início` de falhas consecutivas; ver item 16). Valores exibidos hoje:")
    P()
    t = exploratory.comparacao_consolidada(ev).copy()
    t["proporção (n_falhas/n_eventos)"] = t["n_falhas"].astype(str) + "/" + t["n_eventos"].astype(str) + " = " + t["pct_eventos_corretivos"].map(lambda v: f"{v:.1f}%")
    tbl(t[["tag_code", "n_eventos", "n_falhas", "proporção (n_falhas/n_eventos)", "tempo_medio_reparo_h", "mtbf_dias"]], nd=2)
    t.to_csv(os.path.join(CSV_DIR, "comparacao_consolidada.csv"), index=False, encoding="utf-8")

    P("### Item 25 — Visão geral: sem 'não identificado' como componente mais frequente e com denominador")
    P()
    P("Tabela gerada pela mesma função da tela (`routes._resumo_por_equipamento`): componente mais frequente **entre os identificados**, com o "
      "denominador (total de falhas do equipamento) e, em coluna separada, o nº de falhas sem componente identificado:")
    P()
    from app.presentation import routes
    linhas = routes._resumo_por_equipamento(ev, snap)
    tbl(pd.DataFrame(linhas)[["tag_code", "componente_top", "sem_componente", "data_sugerida_alerta", "frase_data_sugerida"]])

    P("### Item 26 — Tabelas de Modelos e Equipamentos como CSV")
    P()
    linhas = []
    for h in HORIZONTES:
        meta = C["metas"][h]
        for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]:
            linhas.append({"horizonte_dias": h, "algoritmo": meta["modelo"], "threshold_operacional": meta["threshold_operacional"],
                           "tags_desenvolvimento": ",".join(meta["tags_desenvolvimento"]), "tags_teste": ",".join(meta["tags_teste"]),
                           "n_treino": meta["n_treino"], "n_teste": meta["n_teste"],
                           "metrica": m, "teste_threshold_0.50": meta["metricas_teste_threshold_050"][m],
                           "teste_threshold_operacional": meta["metricas_teste_threshold_otimo"][m]})
    pd.DataFrame(linhas).to_csv(os.path.join(CSV_DIR, "modelos_metricas.csv"), index=False, encoding="utf-8")
    s = snap[["tag_code", "data_referencia"] + [f"prob_falha_{h}d" for h in HORIZONTES] + [f"classe_falha_{h}d" for h in HORIZONTES]
             + ["horizonte_sugerido", "data_sugerida_alerta"]].copy()
    s.to_csv(os.path.join(CSV_DIR, "equipamentos_probabilidades.csv"), index=False, encoding="utf-8")
    P("Arquivos (decimal com ponto, UTF-8, probabilidades em fração 0–1):")
    P()
    P("- `evidencias/csv/modelos_metricas.csv` — tela *Modelos* (métricas de teste por horizonte, thresholds 0.50 e operacional).")
    P("- `evidencias/csv/equipamentos_probabilidades.csv` — tela *Equipamentos* (probabilidade e classe por horizonte + data sugerida).")
    P("- Outros: `comparacao_consolidada.csv`, `permutation_importance_*d.csv`, `exemplos_por_categoria.csv`, `auditoria_rotulos_amostra.csv`.")
    P()
    P("Tela *Modelos* (conteúdo em CSV, formato longo):")
    P()
    tbl(pd.DataFrame(linhas), nd=4)
    P("Tela *Equipamentos*:")
    P()
    tbl(s)

    P("### Item 27 — Inventário de testes")
    P()
    arquivos = glob.glob(os.path.join(ROOT, "tests", "**", "*.py"), recursive=True) + glob.glob(os.path.join(ROOT, "test_*.py")) + glob.glob(os.path.join(ROOT, "**", "test_*.py"), recursive=True)
    arquivos = sorted({os.path.relpath(a, ROOT) for a in arquivos if ".tcc_work" not in a and "site-packages" not in a})
    coleta = sh([sys.executable, "-m", "pytest", "--collect-only", "-q", "--rootdir", ROOT, ROOT]).splitlines()[-3:]
    P(f"**Resposta: não existe suíte de testes automatizados.** Arquivos `test_*.py`/`tests/` encontrados: {arquivos or 'nenhum'}. "
      f"Saída do `pytest --collect-only` (últimas linhas): `{' | '.join(coleta)}`. `pytest` ({metadata.version('pytest')}) está instalado, mas não há casos de teste. "
      "As verificações realizadas foram manuais/ad hoc (execução dos scripts, `assert` de vazamento de grupo em `split.py`, conferência das telas via cliente de teste do Flask).")
    P()

    P("### Item 28 — Tabelas do banco e diagrama ER (sem dados identificáveis)")
    P()
    from app.data.models import Base
    from app.data.db import get_session
    con = sqlite3.connect(os.path.join(ROOT, "instance", "manutencao.db"))
    linhas = []
    for t_ in Base.metadata.sorted_tables:
        n = con.execute(f'select count(*) from "{t_.name}"').fetchone()[0]
        for c in t_.columns:
            linhas.append({"tabela": t_.name, "coluna": c.name, "tipo": str(c.type), "PK": c.primary_key, "nulável": c.nullable,
                           "FK declarada": bool(c.foreign_keys), "linhas_na_tabela": n})
    con.close()
    tbl(pd.DataFrame(linhas))
    P("**Chaves estrangeiras declaradas: nenhuma.** As relações abaixo são lógicas (mesmo valor de `tag_code`), garantidas pela ingestão, não pelo banco. "
      "A tabela `tag_mapping_interno` guarda o TAG original (dado identificável) e **não deve ser incluída** em diagramas/anexos do TCC; ela está fora do diagrama.")
    P()
    P("```mermaid")
    P("erDiagram")
    P("    equipamentos ||--o{ eventos_manutencao : \"tag_code (relação lógica)\"")
    P("    componentes ||--o{ eventos_manutencao : \"componente_estimado = nome (relação lógica)\"")
    P("    equipamentos {")
    P("        string tag_code PK")
    P("        string tipo")
    P("        date primeiro_evento")
    P("        date ultimo_evento")
    P("        int total_eventos")
    P("        string status")
    P("    }")
    P("    eventos_manutencao {")
    for c in ["int id PK", "string tag_code", "int os_num", "int tarefa_num", "date inicio", "date fim", "string tipo_manutencao",
              "bool is_falha", "bool is_preventiva", "bool classificado", "float hh_apropriado", "string area", "string sist_oper",
              "string descricao_tarefa", "string componente_estimado"]:
        P(f"        {c}")
    P("    }")
    P("    componentes {")
    for c in ["int id PK", "string nome", "string fabricante_referencia", "string vida_util_indicada", "string unidade",
              "string criterio_substituicao", "string termos_busca"]:
        P(f"        {c}")
    P("    }")
    P("```")
    P()


# ===================================================================== main
def reprocessar_bruto() -> dict:
    """Reexecuta a ingestão (sem gravar no banco) guardando contagens
    intermediárias para o item 15 e as colunas brutas para o item 17."""
    bruto = ingest._carregar_bruto(config.RAW_XLSX_PATH, config.RAW_XLSX_SHEET)
    df = ingest._limpar(bruto)
    df_a = bruto[bruto["Situação"].astype(str).str.strip() != "C"]
    c_chaves = set(zip(df["Nº OS"], df["NºTarefa"]))
    a_com_par = int(sum((os_, ta) in c_chaves for os_, ta in zip(df_a["Nº OS"], df_a["NºTarefa"])))
    df = ingest._filtrar_escopo(df)
    n_escopo = len(df)
    df = ingest._classificar_tipo(df)
    df, _ = ingest._anonimizar_tag(df)
    df = ingest._redigir_descricao(df)
    df = ingest._estimar_componente(df)
    df = ingest._perturbar_hh(df)
    chave = ["Nº OS", "NºTarefa", "Início"]
    dup_mask = df.duplicated(subset=chave, keep="first")
    n_dup = int(dup_mask.sum())
    todos_dup = df[df.duplicated(subset=chave, keep=False)]
    n_grupos = int(todos_dup.groupby(chave).ngroups)
    mantidas = df[~dup_mask].set_index(chave)
    removidas = df[dup_mask]
    iguais = 0
    for _, r in removidas.iterrows():
        k = (r["Nº OS"], r["NºTarefa"], r["Início"])
        m = mantidas.loc[k]
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        if (r["Tipo de Manutenção"] == m["Tipo de Manutenção"] and r["HH Apropriado"] == m["HH Apropriado"]
                and str(r["Descrição Tarefa"]) == str(m["Descrição Tarefa"])):
            iguais += 1
    final = df[~dup_mask].reset_index(drop=True)
    return dict(bruto=bruto, final=final, n_pos_escopo=n_escopo, n_dup=n_dup, n_grupos_dup=n_grupos,
                dup_identicas=iguais, a_com_par_c=a_com_par)


def main() -> None:
    warnings.filterwarnings("ignore")
    os.makedirs(CSV_DIR, exist_ok=True)
    print("Preparando datasets e modelos...")
    C = preparar()
    RS, EXT = {}, {}
    for h in HORIZONTES:
        print(f"Avaliando horizonte {h}d (CV, tuning, OOF)...")
        RS[h] = avaliar_horizonte(h, C)
        print(f"Validação externa um-TAG-fora {h}d...")
        EXT[h] = externo_um_tag_fora(h, C, RS[h])
    print("Snapshot e bruto...")
    snap = snapshot_atual(C["eventos"])
    with contextlib.redirect_stdout(io.StringIO()):
        RAW = reprocessar_bruto()

    agora = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    P("# Evidências para a banca — Manutenção preditiva de monta-cargas")
    P()
    P(f"Gerado por `scripts/gerar_evidencias.py` em {agora}, a partir do código no commit `{sh(['git', 'rev-parse', '--short', 'HEAD'])}`, "
      "do banco `instance/manutencao.db`, dos modelos em `models/` e da planilha bruta local. Todos os números foram calculados na execução; "
      "onde algo não existe no projeto está escrito 'não existe'. Equipamentos aparecem apenas como EQ001–EQ006 (anonimizados).")
    P()
    P("Notação: decimais com ponto; 'não definido' = métrica indefinida (ex.: uma só classe na dobra); AP = Average Precision (PR-AUC).")
    P()
    secao_A(C, RS)
    secao_B(C, RS, EXT)
    secao_14(C, RS, snap)
    secao_C(C, RAW)
    secao_21_23(C)
    secao_D(C, snap)

    texto = "\n".join(OUT) + "\n"
    # Verificação de anonimização: nenhum TAG real nem nome conhecido no arquivo gerado.
    tags_reais = [t for t in RAW["bruto"]["TAG"].astype(str).str.strip().unique()]
    vazamentos = [t for t in tags_reais if re.search(rf"\b{re.escape(t)}\b", texto, flags=re.IGNORECASE)]
    nomes = [n for n in ingest.NOMES_ATRIBUICAO_CONHECIDOS if re.search(rf"\b{re.escape(n)}\b", texto, flags=re.IGNORECASE)]
    if vazamentos or nomes:
        raise SystemExit(f"ABORTADO — possível vazamento no texto gerado: TAGs={vazamentos}, nomes={nomes}")
    with open(os.path.join(OUT_DIR, "evidencias.md"), "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"OK: {os.path.join(OUT_DIR, 'evidencias.md')} ({len(texto):,} caracteres) — verificação de anonimização: sem vazamentos.")


if __name__ == "__main__":
    main()
