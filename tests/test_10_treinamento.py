"""
Treinamento de ponta a ponta com dados fictícios.

Os modelos e gráficos são gravados numa pasta TEMPORÁRIA: os modelos
reais em models/ nunca são tocados.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

import config
from app.modeling import predict, train
from tests.fabrica import evento, eventos

HORIZONTE = 30

# Treinar leva alguns segundos: o teste de mutação pula este arquivo.
pytestmark = pytest.mark.lento

PASTA_MODELOS_REAIS = os.path.join(config.BASE_DIR, "models")


def _estado_dos_modelos_reais():
    if not os.path.isdir(PASTA_MODELOS_REAIS):
        return {}
    return {n: os.path.getmtime(os.path.join(PASTA_MODELOS_REAIS, n)) for n in os.listdir(PASTA_MODELOS_REAIS)}


ESTADO_INICIAL = _estado_dos_modelos_reais()


def _historico_ficticio():
    """6 equipamentos ao longo de 3 anos, cada um com uma frequência de
    falha diferente (uns falham muito, outros pouco)."""
    rng = np.random.default_rng(123)
    linhas = []
    inicio = pd.Timestamp("2020-01-01")
    intervalos = {"EQ001": 6, "EQ002": 10, "EQ003": 15, "EQ004": 25, "EQ005": 40, "EQ006": 60}
    for i, (tag, media) in enumerate(intervalos.items()):
        dia = inicio + pd.Timedelta(days=int(rng.integers(0, 5)))
        while dia < inicio + pd.Timedelta(days=3 * 365):
            tipo = "MP" if rng.random() < 0.25 else "MC"
            linhas.append(evento(tag, dia, tipo, hh=float(rng.uniform(0.5, 4)),
                                 area=f"Area {i % 2}", sist="Sist 1"))
            dia += pd.Timedelta(days=int(max(1, rng.exponential(media))))
    return eventos(*linhas)


@pytest.fixture(scope="module")
def treino(tmp_path_factory):
    pasta = tmp_path_factory.mktemp("modelos")
    originais = (train.MODELS_DIR, train.FIGURES_DIR, predict.MODELS_DIR)
    train.MODELS_DIR = str(pasta / "models")
    train.FIGURES_DIR = str(pasta / "figures")
    predict.MODELS_DIR = train.MODELS_DIR
    try:
        resultado = train.treinar_horizonte(HORIZONTE, _historico_ficticio())
        yield resultado, pasta
    finally:
        train.MODELS_DIR, train.FIGURES_DIR, predict.MODELS_DIR = originais


def test_treino_grava_modelo_e_metadados_na_pasta_temporaria(treino):
    _, pasta = treino
    assert (pasta / "models" / f"modelo_falha_{HORIZONTE}d.joblib").exists()
    assert (pasta / "models" / f"modelo_falha_{HORIZONTE}d.meta.json").exists()


def test_treino_gera_graficos(treino):
    _, pasta = treino
    assert len(list((pasta / "figures").glob("*.png"))) >= 1


def test_threshold_operacional_valido(treino):
    _, pasta = treino
    meta = json.loads((pasta / "models" / f"modelo_falha_{HORIZONTE}d.meta.json").read_text(encoding="utf-8"))
    assert 0 <= meta["threshold_operacional"] <= 1


def test_metadados_trazem_as_metricas(treino):
    _, pasta = treino
    texto = (pasta / "models" / f"modelo_falha_{HORIZONTE}d.meta.json").read_text(encoding="utf-8")
    for metrica in ["Precision", "Recall", "F1"]:
        assert metrica in texto


def test_modelo_treinado_gera_probabilidades_validas(treino):
    eventos_ficticios = _historico_ficticio()
    foto = predict.prever(predict.features_atuais(eventos_ficticios), HORIZONTE)
    prob = foto[f"prob_falha_{HORIZONTE}d"]
    assert prob.between(0, 1).all()
    assert foto[f"classe_falha_{HORIZONTE}d"].isin([0, 1]).all()
    assert len(foto) == 6


def test_modelos_reais_nao_foram_tocados(treino):
    assert train.MODELS_DIR != PASTA_MODELOS_REAIS
    assert _estado_dos_modelos_reais() == ESTADO_INICIAL
