"""
Data sugerida de alerta: primeiro horizonte em que a probabilidade
cruza o limiar. Não é previsão de quando a falha acontece.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

import config
from app.modeling.predict import data_sugerida_alerta, features_atuais
from app.presentation.routes import _frase_data_sugerida
from tests.fabrica import evento, eventos

REF = pd.Timestamp("2025-01-01")


def _linha(p30=None, p60=None, p90=None):
    return pd.Series({"prob_falha_30d": p30, "prob_falha_60d": p60, "prob_falha_90d": p90})


@pytest.mark.parametrize("probabilidade, tem_alerta", [
    (0.49, False),
    (0.4999, False),
    (0.50, True),   # exatamente no limiar já é alerta (>=)
    (0.51, True),
    (1.0, True),
    (0.0, False),
])
def test_fronteira_do_limiar(probabilidade, tem_alerta):
    assert (data_sugerida_alerta(_linha(p30=probabilidade), REF) is not None) is tem_alerta


@pytest.mark.parametrize("probs, horizonte", [
    ((0.9, 0.9, 0.9), 30),
    ((0.1, 0.6, 0.9), 60),
    ((0.1, 0.2, 0.5), 90),
    ((0.6, 0.1, 0.1), 30),   # não precisa ser crescente: vale o primeiro que cruza
])
def test_vale_o_primeiro_horizonte_que_cruza(probs, horizonte):
    alerta = data_sugerida_alerta(_linha(*probs), REF)
    assert alerta["horizonte"] == horizonte
    assert alerta["data"] == (REF + pd.Timedelta(days=horizonte)).date()


def test_sem_nenhum_horizonte_acima_do_limiar_nao_inventa_data():
    assert data_sugerida_alerta(_linha(0.1, 0.2, 0.3), REF) is None


@pytest.mark.parametrize("vazio", [None, np.nan, float("nan")])
def test_probabilidade_vazia_e_ignorada(vazio):
    assert data_sugerida_alerta(_linha(vazio, 0.7, None), REF)["horizonte"] == 60


def test_limiar_personalizado():
    assert data_sugerida_alerta(_linha(0.3, 0.3, 0.3), REF, limiar=0.25)["horizonte"] == 30
    assert data_sugerida_alerta(_linha(0.3, 0.3, 0.3), REF, limiar=0.31) is None


def test_limiar_padrao_vem_da_configuracao(monkeypatch):
    monkeypatch.setattr(config, "LIMIAR_ALERTA", 0.8)
    assert data_sugerida_alerta(_linha(0.7, 0.7, 0.7), REF) is None


def test_devolve_a_probabilidade_e_a_data():
    alerta = data_sugerida_alerta(_linha(0.81, 0.9, 0.95), REF)
    assert alerta == {"horizonte": 30, "probabilidade": 0.81, "data": date(2025, 1, 31)}


# ---------- Frase explicativa ----------

def test_frase_com_alerta():
    frase = _frase_data_sugerida({"data_sugerida_alerta": date(2025, 1, 31), "horizonte_sugerido": 30,
                                  "prob_falha_30d": 0.81})
    assert frase == "A probabilidade estimada de falha ultrapassa o limiar de 50% no horizonte de 30 dias (81%)."


def test_frase_sem_alerta():
    assert _frase_data_sugerida({"data_sugerida_alerta": None}) == \
        "Nenhum horizonte avaliado ultrapassa o limiar de alerta configurado."


# ---------- Foto atual (features na data mais recente) ----------

def test_foto_atual_tem_uma_linha_por_equipamento():
    base = eventos(evento("EQ001", "2024-01-01"), evento("EQ001", "2024-06-01"), evento("EQ002", "2024-03-01"))
    foto = features_atuais(base)
    assert foto["tag_code"].tolist() == ["EQ001", "EQ002"]
    assert (foto["data_referencia"] == pd.Timestamp("2024-06-01")).all()


def test_equipamento_que_so_aparece_depois_da_referencia_fica_fora():
    base = eventos(evento("EQ001", "2024-01-01"), evento("EQ002", "2024-09-01"))
    foto = features_atuais(base, data_referencia=pd.Timestamp("2024-06-01"))
    assert foto["tag_code"].tolist() == ["EQ001"]


def test_snapshot_com_modelo_falso(modelo_falso):
    from app.modeling.predict import snapshot_atual

    base = eventos(*[evento("EQ001", REF - pd.Timedelta(days=d)) for d in (1, 5, 10, 20)],
                   evento("EQ002", REF - pd.Timedelta(days=400), "MP"), evento("EQ002", REF, "MP"))
    foto = snapshot_atual(base).set_index("tag_code")
    # EQ001 teve 4 falhas recentes -> probabilidade alta -> alerta em 30 dias
    assert foto.loc["EQ001", "horizonte_sugerido"] == 30
    # EQ002 nunca falhou -> probabilidade baixa -> sem alerta
    assert pd.isna(foto.loc["EQ002", "horizonte_sugerido"])
    for h in config.HORIZONTES_DIAS:
        assert foto[f"classe_falha_{h}d"].isin([0, 1]).all()
