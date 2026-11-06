"""
Módulo de modelagem (iii) — treino e seleção de modelo por horizonte,
reaproveitando a estrutura do Ex1.py (comparação de modelos via
cross-validation, tuning, predições out-of-fold, otimização de
threshold priorizando recall) — briefing seções 8 e 9.

Executa tudo separadamente para falha_30d e falha_60d (são dois modelos
independentes).
"""
import json
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedGroupKFold,
    LeaveOneGroupOut,
    cross_validate,
    RandomizedSearchCV,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve

import config
from app.processing.dataset import carregar_eventos, construir_dataset
from app.modeling.split import dividir_treino_teste
from app.modeling.pipelines import construir_modelos, PARAM_GRIDS
from app.modeling.features import FEATURES
from app.modeling.evaluate import avaliar, gerar_graficos

RANDOM_STATE = 42
RECALL_DESEJADO = 0.90

# Nº de dobras para calibração de probabilidade (interna, não agrupada por
# TAG — ver nota em `treinar_horizonte`).
CALIBRACAO_CV = 3

# Abaixo desse número de equipamentos (grupos) no conjunto de
# desenvolvimento, StratifiedGroupKFold(n_splits=5) deixa de fazer sentido
# (exige pelo menos 5 grupos, idealmente vários por fold para estratificar
# de verdade). Nesse regime a validação passa a ser LeaveOneGroupOut — um
# fold por equipamento, o máximo de validação cruzada agrupada possível
# com o volume disponível (ex.: escopo monta-cargas, 4-5 TAGs no treino).
GRUPOS_MIN_PARA_KFOLD = 8

# Com poucos TAGs, o número de combinações de hiperparâmetros que dá pra
# avaliar com confiança cai bastante — reduz o orçamento de busca.
N_ITER_TUNING_PADRAO = 12
N_ITER_TUNING_REDUZIDO = 6

METRICAS_CV = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}

MODELS_DIR = os.path.join(config.BASE_DIR, "models")
FIGURES_DIR = os.path.join(config.BASE_DIR, "reports", "figures")


def _escolher_threshold(y_dev, prob_oof) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Reaproveita a lógica do Ex1.py: entre os thresholds que atingem pelo
    menos RECALL_DESEJADO de recall, escolhe o de maior precision. Se
    nenhum atingir, cai para o threshold que maximiza F2 (ainda dando mais
    peso a recall do que a precision).
    """
    precision_values, recall_values, thresholds = precision_recall_curve(y_dev, prob_oof)
    precision_t = precision_values[:-1]
    recall_t = recall_values[:-1]

    candidatos = np.where(recall_t >= RECALL_DESEJADO)[0]
    if len(candidatos) > 0:
        melhor_idx = candidatos[np.argmax(precision_t[candidatos])]
    else:
        beta = 2
        f2 = (
            (1 + beta**2) * precision_t * recall_t
            / (beta**2 * precision_t + recall_t + 1e-12)
        )
        melhor_idx = int(np.argmax(f2))

    return float(thresholds[melhor_idx]), thresholds, precision_t, recall_t


def _minoria_minima_por_fold(cv, X_dev, y_dev, groups_dev) -> int:
    """Menor contagem da classe minoritária entre os conjuntos de TREINO de
    cada fold de `cv`. Usado para decidir se a calibração cruzada
    (CALIBRACAO_CV dobras, estratificada) é numericamente viável — ela
    exige pelo menos CALIBRACAO_CV exemplos de cada classe. Com poucos
    TAGs no escopo, um fold pode ter o treino quase (ou inteiramente) de
    uma única classe.
    """
    minimo = None
    for train_idx, _ in cv.split(X_dev, y_dev, groups_dev):
        contagem = y_dev.iloc[train_idx].value_counts()
        pior = int(contagem.min()) if len(contagem) == 2 else 0
        minimo = pior if minimo is None else min(minimo, pior)
    return minimo if minimo is not None else 0


def _oof_proba_seguro(construir_estimador, X_dev, y_dev, groups_dev, cv) -> tuple[np.ndarray, int]:
    """Substitui `cross_val_predict` por uma versão tolerante a falhas.

    Com poucos TAGs no escopo, ALGUM fold pode ter um conjunto de treino
    com uma única classe (ex.: só "falha"), o que quebra o `.fit()` de
    alguns modelos (Logistic Regression exige as duas classes) e não tem
    como ser contornado com `error_score` (`cross_val_predict` não aceita
    esse parâmetro). Quando isso acontece, o fold problemático recebe como
    previsão a taxa de falha do próprio treino daquele fold (equivalente a
    um modelo "sem informação"), e o motivo fica registrado no log — em
    vez de a execução inteira travar.

    `construir_estimador` é uma função sem argumentos que devolve um
    estimador NÃO treinado a cada chamada (um `clone` novo por fold).
    """
    prob_oof = np.full(len(y_dev), np.nan)
    n_folds_com_fallback = 0

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_dev, y_dev, groups_dev)):
        X_tr, y_tr = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        X_te = X_dev.iloc[test_idx]
        try:
            estimador = construir_estimador()
            estimador.fit(X_tr, y_tr)
            prob_oof[test_idx] = estimador.predict_proba(X_te)[:, 1]
        except Exception as exc:
            taxa_treino = y_tr.mean()
            prob_oof[test_idx] = taxa_treino
            n_folds_com_fallback += 1
            print(f"[aviso] Fold {fold_idx}: não deu para ajustar o modelo "
                  f"({type(exc).__name__}: {exc}). Usando a taxa de falha do "
                  f"treino desse fold ({taxa_treino:.3f}) como previsão.")

    return prob_oof, n_folds_com_fallback


def treinar_horizonte(horizonte: int, eventos: pd.DataFrame = None) -> dict:
    print("\n" + "=" * 70)
    print(f"HORIZONTE: falha_{horizonte}d")
    print("=" * 70)

    dataset = construir_dataset(horizonte, eventos)
    alvo = f"falha_{horizonte}d"

    divisao = dividir_treino_teste(dataset, test_size=0.2)
    print(f"Divisão por equipamento: desenvolvimento {sorted(divisao.treino['tag_code'].unique())} | "
          f"teste {sorted(divisao.teste['tag_code'].unique())}")
    print(f"Treino: {len(divisao.treino)} linhas / {divisao.treino['tag_code'].nunique()} TAGs "
          f"({divisao.treino['data_referencia'].min().date()} a {divisao.treino['data_referencia'].max().date()})")
    print(f"Teste:  {len(divisao.teste)} linhas / {divisao.teste['tag_code'].nunique()} TAGs "
          f"({divisao.teste['data_referencia'].min().date()} a {divisao.teste['data_referencia'].max().date()})")
    print(f"Falhas no treino: {divisao.treino[alvo].mean():.3f} | Falhas no teste: {divisao.teste[alvo].mean():.3f}")

    X_dev, y_dev, groups_dev = divisao.treino[FEATURES], divisao.treino[alvo], divisao.treino["tag_code"]
    X_test, y_test = divisao.teste[FEATURES], divisao.teste[alvo]

    # -- Diagnóstico por TAG (importante com poucos equipamentos no escopo) ------
    taxa_por_tag_dev = divisao.treino.groupby("tag_code")[alvo].agg(["size", "mean"])
    degenerados = taxa_por_tag_dev[(taxa_por_tag_dev["mean"] == 0) | (taxa_por_tag_dev["mean"] == 1)]
    if len(degenerados) > 0:
        print(f"\n[aviso] {len(degenerados)} TAG(s) no treino com taxa de falha 0% ou 100% "
              f"(sempre/nunca falha nessa janela) — comum com poucos equipamentos no escopo:")
        print(degenerados.rename(columns={"size": "linhas", "mean": "taxa_falha"}).to_string())

    n_grupos_dev = groups_dev.nunique()
    if n_grupos_dev < GRUPOS_MIN_PARA_KFOLD:
        cv = LeaveOneGroupOut()
        n_iter_tuning = N_ITER_TUNING_REDUZIDO
        print(f"\n[aviso] Apenas {n_grupos_dev} equipamentos no treino — usando "
              f"LeaveOneGroupOut ({n_grupos_dev} folds, um por TAG) em vez de "
              f"StratifiedGroupKFold(5). ROC-AUC/Average Precision ficam NaN nos "
              f"folds cujo TAG deixado de fora tem taxa de falha 0%/100% (não há a "
              f"outra classe para comparar); a agregação abaixo ignora esses NaN "
              f"(np.nanmean) e reporta quantos folds válidos entraram em cada média.")
    else:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        n_iter_tuning = N_ITER_TUNING_PADRAO

    # -- Comparação de modelos ------------------------------------------------
    modelos = construir_modelos()
    resultados = []
    for nome, modelo in modelos.items():
        scores = cross_validate(
            modelo, X_dev, y_dev, groups=groups_dev, cv=cv, scoring=METRICAS_CV, n_jobs=-1, error_score=np.nan
        )
        linha = {"modelo": nome}
        for m in METRICAS_CV:
            valores = scores[f"test_{m}"]
            linha[f"{m}_mean"] = np.nanmean(valores)
            linha[f"{m}_std"] = np.nanstd(valores)
        linha["folds_validos_ap"] = int(np.sum(~np.isnan(scores["test_average_precision"])))
        resultados.append(linha)

    resultados_df = pd.DataFrame(resultados).sort_values("average_precision_mean", ascending=False)
    pd.set_option("display.max_columns", None)
    nome_cv = "LeaveOneGroupOut" if n_grupos_dev < GRUPOS_MIN_PARA_KFOLD else "StratifiedGroupKFold(5)"
    print(f"\nComparação de modelos (cross-validation, {nome_cv} por TAG; "
          f"'folds_validos_ap' = quantos dos {cv.get_n_splits(X_dev, y_dev, groups_dev)} folds "
          f"tinham as duas classes p/ calcular average_precision):")
    print(resultados_df.round(4).to_string(index=False))

    melhor_nome = resultados_df.iloc[0]["modelo"]
    melhor_modelo = modelos[melhor_nome]
    print(f"\nMelhor modelo (Average Precision): {melhor_nome}")

    # -- Tuning de hiperparâmetros ---------------------------------------------
    if melhor_nome == "Logistic Regression":
        parametros = {"classifier__C": np.logspace(-3, 2, 30)}
    else:
        parametros = PARAM_GRIDS[melhor_nome]

    search = RandomizedSearchCV(
        estimator=melhor_modelo,
        param_distributions=parametros,
        n_iter=n_iter_tuning,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        error_score=np.nan,
    )
    search.fit(X_dev, y_dev, groups=groups_dev)
    print(f"Melhor Average Precision (tuning): {search.best_score_:.4f}")
    print(f"Melhores hiperparâmetros: {search.best_params_}")

    # -- Calibração de probabilidade ---------------------------------------------
    # `class_weight="balanced"` (usado em todos os candidatos para lidar com o
    # desbalanceamento) melhora a separação das classes, mas deixa o
    # `predict_proba` bruto sistematicamente inflado: a probabilidade média
    # prevista no próprio conjunto de treino fica bem acima da taxa real de
    # falha (a média prevista ficava bem acima da taxa real). Sem corrigir
    # isso, o número exibido na interface como "probabilidade de falha" não
    # seria uma probabilidade de verdade — só um score de risco relativo.
    # `CalibratedClassifierCV` reajusta a saída para refletir a frequência
    # observada (briefing seção 9: gráfico de calibração já fazia parte da
    # avaliação; aqui ele passa a ser corrigido, não só diagnosticado).
    #
    # A dobra usada para calibrar (CALIBRACAO_CV) não é agrupada por TAG —
    # diferente do split treino/teste e do OOF abaixo, que continuam 100%
    # agrupados por TAG. É uma simplificação aceitável porque a calibração
    # só ajusta a ESCALA das probabilidades, não decide generalização nem
    # métricas de avaliação — essas continuam vindo de divisões sem
    # vazamento de grupo.
    #
    # Com poucos TAGs no escopo, o conjunto de TREINO de algum fold de
    # validação pode sobrar com poucos (ou nenhum) exemplo da classe
    # minoritária — nesse caso não dá pra calibrar com segurança (a
    # calibração em si exige CALIBRACAO_CV exemplos de cada classe).
    # Verifica isso ANTES de tentar, em vez de deixar estourar exceção.
    minoria_min = _minoria_minima_por_fold(cv, X_dev, y_dev, groups_dev)
    calibrar = minoria_min >= CALIBRACAO_CV
    if calibrar:
        print(f"Calibração de probabilidade: ativada (pior fold de validação tem "
              f"{minoria_min} exemplos da classe minoritária no treino, "
              f">= {CALIBRACAO_CV} exigidos).")
    else:
        print(f"[aviso] Calibração de probabilidade DESATIVADA para falha_{horizonte}d: "
              f"pelo menos um fold de validação teria só {minoria_min} exemplo(s) da classe "
              f"minoritária no treino do fold (mínimo necessário: {CALIBRACAO_CV}). As "
              f"probabilidades deste horizonte são a saída bruta do modelo (não recalibrada) "
              f"— tendem a ficar deslocadas da taxa real. Documentar como limitação.")

    def _construir_modelo_final():
        base = clone(search.best_estimator_)
        return CalibratedClassifierCV(base, method="sigmoid", cv=CALIBRACAO_CV) if calibrar else base

    # -- Predições out-of-fold + escolha de threshold ------------------------------
    # `cross_val_predict` não tem `error_score` — com poucos TAGs, algum
    # fold pode ter o treino de uma única classe e quebrar o `.fit()` de
    # alguns modelos (ex.: Logistic Regression). `_oof_proba_seguro` cobre
    # esse caso sem derrubar a execução inteira (ver docstring da função).
    prob_oof, n_folds_fallback = _oof_proba_seguro(_construir_modelo_final, X_dev, y_dev, groups_dev, cv)
    if n_folds_fallback > 0:
        print(f"[aviso] {n_folds_fallback} de {cv.get_n_splits(X_dev, y_dev, groups_dev)} folds "
              f"usaram a taxa de falha do treino como previsão (não deu pra ajustar o modelo "
              f"nesses folds — ver avisos acima).")
    threshold_otimo, thresholds, precision_t, recall_t = _escolher_threshold(y_dev, prob_oof)
    print(f"Threshold padrão: 0.500 | Threshold operacional: {threshold_otimo:.3f} "
          f"(alvo: recall >= {RECALL_DESEJADO})")
    print(f"Probabilidade média prevista (OOF): {prob_oof.mean():.3f} "
          f"| taxa real de falha no treino: {y_dev.mean():.3f}")

    # -- Treino final + avaliação no teste ------------------------------------------
    modelo_otimizado = _construir_modelo_final()
    modelo_otimizado.fit(X_dev, y_dev)
    prob_test = modelo_otimizado.predict_proba(X_test)[:, 1]
    pred_050 = (prob_test >= 0.50).astype(int)
    pred_otimo = (prob_test >= threshold_otimo).astype(int)

    resultado_050 = avaliar(y_test, pred_050, prob_test)
    resultado_otimo = avaliar(y_test, pred_otimo, prob_test)
    comparacao = pd.DataFrame({"Threshold 0.50": resultado_050, f"Threshold {threshold_otimo:.3f}": resultado_otimo})
    print("\nResultado final — TESTE:")
    print(comparacao.round(4).to_string())

    # -- Persistência -----------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    caminho_modelo = os.path.join(MODELS_DIR, f"modelo_falha_{horizonte}d.joblib")
    joblib.dump(modelo_otimizado, caminho_modelo)

    caminho_meta = os.path.join(MODELS_DIR, f"modelo_falha_{horizonte}d.meta.json")
    with open(caminho_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "horizonte": horizonte,
                "modelo": melhor_nome,
                "hiperparametros": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in search.best_params_.items()},
                "threshold_operacional": threshold_otimo,
                "recall_desejado": RECALL_DESEJADO,
                "features": FEATURES,
                "metricas_teste_threshold_050": resultado_050,
                "metricas_teste_threshold_otimo": resultado_otimo,
                "tags_desenvolvimento": sorted(divisao.treino["tag_code"].unique().tolist()),
                "tags_teste": sorted(divisao.teste["tag_code"].unique().tolist()),
                "n_treino": len(divisao.treino),
                "n_teste": len(divisao.teste),
                "n_tags_treino": int(n_grupos_dev),
                "n_tags_teste": int(divisao.teste["tag_code"].nunique()),
                "validacao_cruzada": nome_cv,
                "tags_escopo": sorted(dataset["tag_code"].unique().tolist()),
                "calibracao_probabilidade": calibrar,
                "folds_com_fallback_oof": n_folds_fallback,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Modelo salvo em {caminho_modelo}")

    # -- Gráficos -----------------------------------------------------------------
    gerar_graficos(
        horizonte=horizonte,
        nome_modelo=melhor_nome,
        modelo_final=modelo_otimizado,
        X_test=X_test,
        y_test=y_test,
        prob_test=prob_test,
        pred_050=pred_050,
        pred_otimo=pred_otimo,
        threshold_otimo=threshold_otimo,
        thresholds=thresholds,
        precision_threshold=precision_t,
        recall_threshold=recall_t,
        out_dir=FIGURES_DIR,
    )
    print(f"Gráficos salvos em {FIGURES_DIR}")

    return {
        "horizonte": horizonte,
        "modelo": melhor_nome,
        "threshold": threshold_otimo,
        "metricas_teste": resultado_otimo,
    }


def executar_tudo() -> list[dict]:
    eventos = carregar_eventos()
    resumo = []
    for h in config.HORIZONTES_DIAS:
        resumo.append(treinar_horizonte(h, eventos))
    return resumo


if __name__ == "__main__":
    resultados = executar_tudo()
    print("\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)
    for r in resultados:
        print(r)
