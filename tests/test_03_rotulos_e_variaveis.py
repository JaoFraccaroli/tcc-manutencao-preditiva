"""
Construção do dataset: datas de referência, rótulo "houve falha nos
próximos N dias" e variáveis que olham só para o passado.

Um erro aqui contamina todos os modelos, então os testes ficam nas
fronteiras exatas das janelas.
"""
import numpy as np
import pandas as pd
import pytest

import config
from app.processing import dataset
from tests.fabrica import evento, eventos

REF = pd.Timestamp("2024-01-01")


def _datas(*dias):
    return np.array([np.datetime64(REF + pd.Timedelta(days=d)) for d in dias])


# ---------- Rótulo (fronteiras para os três horizontes) ----------

@pytest.mark.parametrize("horizonte", config.HORIZONTES_DIAS)
@pytest.mark.parametrize("dias_ate_a_falha, rotulo", [
    (-30, False),  # falha antes da data de referência: é passado
    (-1, False),
    (0, False),    # no próprio dia da referência: já é conhecida, não é futuro
    (1, True),     # primeiro dia da janela
    ("h", True),   # último dia da janela
    ("h+1", False),  # um dia depois da janela
])
def test_rotulo_nas_fronteiras_da_janela(horizonte, dias_ate_a_falha, rotulo):
    dia = {"h": horizonte, "h+1": horizonte + 1}.get(dias_ate_a_falha, dias_ate_a_falha)
    assert dataset._rotular(_datas(dia), pd.DatetimeIndex([REF]), horizonte).tolist() == [rotulo]


def test_sem_falhas_o_rotulo_e_sempre_zero():
    refs = pd.date_range(REF, periods=5, freq="7D")
    assert not dataset._rotular(np.array([], dtype="datetime64[ns]"), refs, 30).any()


def test_rotulo_para_varias_datas_de_referencia():
    refs = pd.DatetimeIndex([REF, REF + pd.Timedelta(days=7), REF + pd.Timedelta(days=40)])
    # Falha no dia 35: dentro da janela de 30d das refs dia 7 (8..37), fora da ref dia 0 (1..30)
    assert dataset._rotular(_datas(35), refs, 30).tolist() == [False, True, False]


def test_varias_falhas_na_janela_continuam_sendo_rotulo_um():
    assert dataset._rotular(_datas(2, 3, 4), pd.DatetimeIndex([REF]), 30).tolist() == [True]


# ---------- Grade de datas de referência (censura à direita) ----------

def test_grade_vai_do_primeiro_evento_ate_o_fim_menos_o_horizonte():
    grade = dataset._grade_datas_referencia(_datas(0), REF + pd.Timedelta(days=60), 30)
    assert grade[0] == REF
    assert grade[-1] <= REF + pd.Timedelta(days=30)
    assert (grade[1:] - grade[:-1]).unique().tolist() == [pd.Timedelta(days=config.REFERENCE_STEP_DAYS)]
    assert len(grade) == 5  # dias 0, 7, 14, 21, 28


@pytest.mark.parametrize("dias_de_historico, quantidade", [
    (29, 0),   # não cabe nenhuma janela completa de 30 dias
    (30, 1),   # cabe exatamente uma
    (37, 2),
])
def test_grade_so_tem_janelas_completas(dias_de_historico, quantidade):
    grade = dataset._grade_datas_referencia(_datas(0), REF + pd.Timedelta(days=dias_de_historico), 30)
    assert len(grade) == quantidade


def test_horizonte_maior_gera_menos_datas_de_referencia():
    fim = REF + pd.Timedelta(days=365)
    tamanhos = [len(dataset._grade_datas_referencia(_datas(0), fim, h)) for h in config.HORIZONTES_DIAS]
    assert tamanhos == sorted(tamanhos, reverse=True)
    assert tamanhos[0] > tamanhos[-1]


# ---------- Variáveis (features) ----------

def _features(evts, *dias_ref):
    refs = pd.DatetimeIndex([REF + pd.Timedelta(days=d) for d in dias_ref])
    return dataset._features_para_grupo(evts, refs)


@pytest.fixture
def historico():
    return eventos(
        evento("EQ001", REF - pd.Timedelta(days=200), "MC", hh=1.0, area="Area A"),
        evento("EQ001", REF - pd.Timedelta(days=90), "MC", hh=2.0, area="Area A"),
        evento("EQ001", REF - pd.Timedelta(days=89), "MP", hh=3.0, area="Area A"),
        evento("EQ001", REF - pd.Timedelta(days=10), "CP", hh=6.0, area="Area B"),
    )


def test_tempo_desde_a_ultima_falha(historico):
    assert _features(historico, 0)["tempo_desde_ultima_falha"].iloc[0] == 10


def test_falha_no_dia_da_referencia_conta_como_passado(historico):
    assert _features(historico, -10)["tempo_desde_ultima_falha"].iloc[0] == 0


def test_antes_da_primeira_falha_o_tempo_fica_vazio(historico):
    assert np.isnan(_features(historico, -201)["tempo_desde_ultima_falha"].iloc[0])


def test_equipamento_sem_nenhuma_falha():
    so_preventivas = eventos(evento("EQ001", REF - pd.Timedelta(days=5), "MP"))
    feats = _features(so_preventivas, 0)
    assert np.isnan(feats["tempo_desde_ultima_falha"].iloc[0])
    assert feats["freq_falhas_90d"].iloc[0] == 0


def test_frequencia_de_falhas_nas_janelas(historico):
    feats = _features(historico, 0).iloc[0]
    # Janela de 90 dias = (ref-90, ref]: a falha de exatamente 90 dias atrás fica FORA
    assert feats["freq_falhas_90d"] == 1
    # Janela de 180 dias: entram as de 90 e 10 dias atrás; a de 200 fica fora
    assert feats["freq_falhas_180d"] == 2


def test_fronteira_da_janela_de_90_dias():
    evts = eventos(evento("EQ001", REF - pd.Timedelta(days=89), "MC"),
                   evento("EQ001", REF - pd.Timedelta(days=90), "MC"))
    assert _features(evts, 0)["freq_falhas_90d"].iloc[0] == 1


def test_preventivas_recentes(historico):
    assert _features(historico, 0)["contagem_preventivas_recentes"].iloc[0] == 1
    assert _features(historico, 1)["contagem_preventivas_recentes"].iloc[0] == 0  # saiu da janela


def test_tempo_medio_de_reparo_e_historico(historico):
    feats = _features(historico, 0).iloc[0]
    assert feats["tempo_medio_reparo"] == pytest.approx((1 + 2 + 3 + 6) / 4)
    assert feats["n_eventos_historico"] == 4


def test_area_e_a_do_evento_mais_recente(historico):
    assert _features(historico, 0)["area"].iloc[0] == "Area B"
    assert _features(historico, -50)["area"].iloc[0] == "Area A"


def test_antes_do_primeiro_evento_nao_ha_historico(historico):
    feats = _features(historico, -300).iloc[0]
    assert feats["n_eventos_historico"] == 0
    assert np.isnan(feats["tempo_medio_reparo"])
    assert feats["area"] is None


def test_eventos_do_futuro_nao_mudam_as_variaveis(historico):
    """Sem vazamento: acrescentar eventos DEPOIS da referência não pode
    mudar nenhuma variável calculada para ela."""
    antes = _features(historico, 0)
    futuro = pd.concat([historico, eventos(
        evento("EQ001", REF + pd.Timedelta(days=1), "MC", hh=50, area="Area Z"),
        evento("EQ001", REF + pd.Timedelta(days=20), "MP", hh=50, area="Area Z"),
    )]).sort_values("inicio").reset_index(drop=True)
    depois = _features(futuro, 0)
    pd.testing.assert_frame_equal(antes, depois)


# ---------- Dataset completo ----------

def _base_dois_equipamentos():
    linhas = []
    for i in range(0, 400, 20):
        linhas.append(evento("EQ001", REF + pd.Timedelta(days=i), "MC" if i % 40 == 0 else "MP"))
    for i in range(50, 400, 35):
        linhas.append(evento("EQ002", REF + pd.Timedelta(days=i), "MC"))
    return eventos(*linhas)


@pytest.mark.parametrize("horizonte", config.HORIZONTES_DIAS)
def test_dataset_tem_rotulo_e_variaveis(horizonte):
    ds = dataset.construir_dataset(horizonte, _base_dois_equipamentos())
    rotulo = f"falha_{horizonte}d"
    assert set(ds[rotulo].unique()) <= {0, 1}
    assert set(ds["tag_code"]) == {"EQ001", "EQ002"}
    for coluna in ["tempo_desde_ultima_falha", "freq_falhas_90d", "freq_falhas_180d",
                   "tempo_medio_reparo", "contagem_preventivas_recentes", "area", "sist_oper"]:
        assert coluna in ds.columns


@pytest.mark.parametrize("horizonte", config.HORIZONTES_DIAS)
def test_nenhuma_janela_passa_do_fim_da_base(horizonte):
    base = _base_dois_equipamentos()
    ds = dataset.construir_dataset(horizonte, base)
    assert (ds["data_referencia"] + pd.Timedelta(days=horizonte) <= base["inicio"].max()).all()


@pytest.mark.parametrize("horizonte", [0, 7, 45, 120, -30])
def test_horizonte_fora_da_configuracao_e_recusado(horizonte):
    with pytest.raises(ValueError, match="HORIZONTES_DIAS"):
        dataset.construir_dataset(horizonte, _base_dois_equipamentos())


def test_equipamento_com_historico_curto_fica_fora():
    base = pd.concat([_base_dois_equipamentos(), eventos(
        evento("EQ003", REF + pd.Timedelta(days=390), "MC"))])
    ds = dataset.construir_dataset(30, base)
    assert "EQ003" not in set(ds["tag_code"])


def test_rotulo_confere_com_contagem_manual():
    base = _base_dois_equipamentos()
    ds = dataset.construir_dataset(30, base)
    falhas = base[base["is_falha"]]
    for linha in ds.sample(15, random_state=1).itertuples():
        f = falhas[falhas["tag_code"] == linha.tag_code]["inicio"]
        esperado = ((f > linha.data_referencia) & (f <= linha.data_referencia + pd.Timedelta(days=30))).any()
        assert linha.falha_30d == int(esperado)
