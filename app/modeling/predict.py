"""
Aplicação dos modelos já treinados sobre o estado ATUAL de cada
equipamento (uma "foto" da base na data mais recente disponível), para
uso pela camada de apresentação.

Importante (briefing seção 1 e 10): este módulo só calcula e devolve a
probabilidade estimada por equipamento. Não ordena, não rotula nada como
"prioritário" e não decide o que o usuário deve fazer com a informação —
isso é responsabilidade exclusiva da interface/usuário.
"""
import json
import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

import config
from app.processing.dataset import _features_para_grupo  # reaproveita o mesmo cálculo do treino
from app.modeling.features import FEATURES

MODELS_DIR = os.path.join(config.BASE_DIR, "models")


def carregar_modelo(horizonte: int) -> tuple:
    caminho_modelo = os.path.join(MODELS_DIR, f"modelo_falha_{horizonte}d.joblib")
    caminho_meta = os.path.join(MODELS_DIR, f"modelo_falha_{horizonte}d.meta.json")

    modelo = joblib.load(caminho_modelo)
    with open(caminho_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return modelo, meta


def features_atuais(eventos: pd.DataFrame, data_referencia: pd.Timestamp = None) -> pd.DataFrame:
    """Uma linha por TAG com as features calculadas na `data_referencia`
    (por padrão, a data mais recente presente na base — o "hoje" da
    aplicação, já que não há dados depois disso).
    """
    if data_referencia is None:
        data_referencia = eventos["inicio"].max()

    refs = pd.DatetimeIndex([data_referencia])

    blocos = []
    for tag_code, grupo in eventos.groupby("tag_code", sort=False):
        grupo = grupo.sort_values("inicio")
        if grupo["inicio"].min() > data_referencia:
            continue  # equipamento só passou a existir na base depois da data de referência
        feats = _features_para_grupo(grupo, refs)
        feats.insert(0, "tag_code", tag_code)
        blocos.append(feats)

    df = pd.concat(blocos, ignore_index=True)
    return df


def prever(df_features: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    """Adiciona ao DataFrame as colunas `prob_falha_{h}d` e
    `classe_falha_{h}d` (0/1 no threshold operacional do modelo).
    """
    modelo, meta = carregar_modelo(horizonte)
    threshold = meta["threshold_operacional"]

    prob = modelo.predict_proba(df_features[FEATURES])[:, 1]

    df = df_features.copy()
    df[f"prob_falha_{horizonte}d"] = prob
    df[f"classe_falha_{horizonte}d"] = (prob >= threshold).astype(int)
    return df


def data_sugerida_alerta(
    linha: pd.Series, data_referencia: pd.Timestamp, limiar: float = None
) -> dict | None:
    """Deriva uma "data sugerida" de atenção a partir das probabilidades
    JÁ CALCULADAS por horizonte — não é uma nova estimativa de modelo, é
    só encontrar o primeiro horizonte (30, 60 ou 90 dias) em que a
    probabilidade estimada cruza `limiar` (briefing complementar 2,
    seção 4).

    Devolve `None` quando nenhum horizonte cruza o limiar ("sem alerta
    nos horizontes avaliados"), em vez de forçar uma data. A data
    resultante representa quando o RISCO ESTIMADO cruza o limiar
    escolhido — não uma previsão determinística de quando a falha vai
    ocorrer.
    """
    limiar = config.LIMIAR_ALERTA if limiar is None else limiar
    for h in sorted(config.HORIZONTES_DIAS):
        prob = linha.get(f"prob_falha_{h}d")
        if prob is not None and not pd.isna(prob) and prob >= limiar:
            return {
                "horizonte": h,
                "probabilidade": float(prob),
                "data": (data_referencia + timedelta(days=h)).date(),
            }
    return None


def snapshot_atual(eventos: pd.DataFrame) -> pd.DataFrame:
    """DataFrame com uma linha por TAG, features atuais, a probabilidade
    estimada de falha em cada horizonte de config.HORIZONTES_DIAS, e a
    data sugerida de alerta derivada dessas probabilidades (briefing
    complementar 2, seção 4).
    """
    base = features_atuais(eventos)
    for h in config.HORIZONTES_DIAS:
        pred = prever(base, h)
        base[f"prob_falha_{h}d"] = pred[f"prob_falha_{h}d"]
        base[f"classe_falha_{h}d"] = pred[f"classe_falha_{h}d"]

    data_referencia = base["data_referencia"].iloc[0] if "data_referencia" in base.columns else eventos["inicio"].max()
    alertas = base.apply(lambda linha: data_sugerida_alerta(linha, data_referencia), axis=1)
    base["horizonte_sugerido"] = alertas.map(lambda a: a["horizonte"] if a else None)
    base["data_sugerida_alerta"] = alertas.map(lambda a: a["data"] if a else None)

    return base
