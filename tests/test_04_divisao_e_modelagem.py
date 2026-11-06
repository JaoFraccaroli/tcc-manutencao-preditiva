"""
Divisão treino/teste por equipamento, escolha do threshold e métricas.
"""
import numpy as np
import pandas as pd
import pytest

from app.modeling import evaluate, train
from app.modeling.split import dividir_treino_teste


def _dataset(linhas_por_tag: dict, mediana_por_tag: dict):
    partes = []
    for tag, n in linhas_por_tag.items():
        centro = pd.Timestamp(mediana_por_tag[tag])
        datas = pd.date_range(centro - pd.Timedelta(days=7 * (n // 2)), periods=n, freq="7D")
        partes.append(pd.DataFrame({"tag_code": tag, "data_referencia": datas, "falha_30d": 0}))
    return pd.concat(partes, ignore_index=True)


# ---------- Divisão treino/teste ----------

@pytest.fixture
def seis_equipamentos():
    return _dataset(
        {f"EQ00{i}": 100 for i in range(1, 7)},
        {"EQ001": "2012-01-01", "EQ002": "2013-01-01", "EQ003": "2014-01-01",
         "EQ004": "2016-01-01", "EQ005": "2011-01-01", "EQ006": "2018-01-01"},
    )


def test_nenhum_equipamento_aparece_nos_dois_lados(seis_equipamentos):
    divisao = dividir_treino_teste(seis_equipamentos)
    assert not set(divisao.treino["tag_code"]) & set(divisao.teste["tag_code"])


def test_nenhuma_linha_e_perdida(seis_equipamentos):
    divisao = dividir_treino_teste(seis_equipamentos)
    assert len(divisao.treino) + len(divisao.teste) == len(seis_equipamentos)


def test_os_mais_recentes_vao_para_o_teste(seis_equipamentos):
    divisao = dividir_treino_teste(seis_equipamentos)
    # 20% de 600 linhas = 120: entram EQ006 (2018) e EQ004 (2016)
    assert set(divisao.teste["tag_code"]) == {"EQ006", "EQ004"}
    assert divisao.data_mediana_corte == divisao.teste.groupby("tag_code")["data_referencia"].median().min()


@pytest.mark.parametrize("test_size, esperados", [
    (0.0, 0),
    (0.01, 1),
    (0.2, 2),
    (0.5, 3),
    (1.0, 6),
])
def test_tamanho_do_teste(seis_equipamentos, test_size, esperados):
    assert dividir_treino_teste(seis_equipamentos, test_size).teste["tag_code"].nunique() == esperados


def test_equipamento_inteiro_vai_para_um_lado_so_mesmo_passando_do_alvo():
    ds = _dataset({"EQ001": 10, "EQ002": 90}, {"EQ001": "2010-01-01", "EQ002": "2020-01-01"})
    divisao = dividir_treino_teste(ds, test_size=0.2)
    # O mais recente (EQ002) tem 90% das linhas: vai inteiro para o teste
    assert set(divisao.teste["tag_code"]) == {"EQ002"}
    assert len(divisao.teste) == 90


# ---------- Escolha do threshold ----------

def test_escolhe_a_maior_precisao_com_recall_minimo():
    # 10 falhas e 4 não-falhas. Conferência manual:
    #   corte 0.5 -> recall 9/10 = 0.90 e precisão 9/10  = 0.900
    #   corte 0.4 -> recall 10/10 = 1.0 e precisão 10/11 = 0.909  <- maior precisão
    #   corte 0.6 -> recall 8/10 = 0.80 (abaixo do mínimo)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    p = np.array([.1, .2, .3, .85, .4, .5, .6, .7, .8, .9, .91, .92, .93, .94])
    threshold, _, _, _ = train._escolher_threshold(y, p)
    assert threshold == pytest.approx(0.4)
    previsto = p >= threshold
    assert previsto[y == 1].mean() >= train.RECALL_DESEJADO


def test_threshold_nunca_deixa_o_recall_abaixo_do_minimo():
    rng = np.random.default_rng(42)
    for _ in range(20):
        y = rng.integers(0, 2, 100)
        y[:2] = [0, 1]
        p = np.clip(y * 0.4 + rng.random(100) * 0.6, 0, 1)
        threshold, _, _, _ = train._escolher_threshold(y, p)
        assert (p >= threshold)[y == 1].mean() >= train.RECALL_DESEJADO


def test_alternativa_f2_nunca_e_usada():
    """Achado da análise de cobertura: o menor threshold da curva sempre
    marca todos como falha, então sempre existe um corte com recall 1.0.
    Por isso o caminho alternativo (F2) de `_escolher_threshold` nunca
    executa. Não muda o resultado do TCC; só é código que não roda."""
    from sklearn.metrics import precision_recall_curve

    rng = np.random.default_rng(7)
    for _ in range(20):
        y = rng.integers(0, 2, 50)
        y[:2] = [0, 1]
        _, recall, _ = precision_recall_curve(y, rng.random(50))
        assert recall[0] == 1.0


def test_threshold_e_um_dos_valores_possiveis():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    threshold, thresholds, _, _ = train._escolher_threshold(y, p)
    assert threshold in thresholds
    assert 0 <= threshold <= 1


def test_recall_desejado_e_noventa_por_cento():
    assert train.RECALL_DESEJADO == 0.90


# ---------- Métricas ----------

def test_metricas_de_um_caso_conhecido():
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 1, 0]
    prob = [0.9, 0.8, 0.3, 0.7, 0.1]
    m = evaluate.avaliar(y_true, y_pred, prob)
    assert m["Accuracy"] == pytest.approx(3 / 5)
    assert m["Precision"] == pytest.approx(2 / 3)
    assert m["Recall"] == pytest.approx(2 / 3)
    assert m["F1"] == pytest.approx(2 / 3)
    assert 0 <= m["ROC-AUC"] <= 1
    assert 0 <= m["PR-AUC"] <= 1


@pytest.mark.parametrize("classe_unica", [0, 1])
def test_teste_com_uma_classe_so_nao_quebra(classe_unica):
    m = evaluate.avaliar([classe_unica] * 4, [classe_unica] * 4, [0.5] * 4)
    assert m["ROC-AUC"] is None
    assert m["PR-AUC"] is None
    assert m["Accuracy"] == 1.0


def test_nenhuma_previsao_positiva_nao_divide_por_zero():
    m = evaluate.avaliar([1, 0, 1], [0, 0, 0], [0.1, 0.2, 0.3])
    assert m["Precision"] == 0
    assert m["Recall"] == 0
