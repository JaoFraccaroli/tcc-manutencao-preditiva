"""
MTBF, tendência do tempo de reparo, sazonalidade e comparação.
"""
import pandas as pd
import pytest

from app.processing import exploratory as ex
from tests.fabrica import evento, eventos


@pytest.mark.parametrize("dias_das_falhas, mtbf", [
    ([], None),             # sem falhas: equipamento nem aparece
    ([0], None),            # 1 falha: intervalo indefinido (não é zero)
    ([0, 10], 10.0),
    ([0, 10, 30], 15.0),
    ([0, 0, 10], 5.0),      # falhas no mesmo dia entram com intervalo 0
])
def test_mtbf(dias_das_falhas, mtbf):
    base = pd.Timestamp("2024-01-01")
    linhas = [evento("EQ001", base + pd.Timedelta(days=d), "MC") for d in dias_das_falhas]
    linhas.append(evento("EQ001", base, "MP"))  # preventiva não entra no MTBF
    tabela = ex.mtbf_por_equipamento(eventos(*linhas)) if dias_das_falhas else None
    if not dias_das_falhas:
        with pytest.raises(KeyError):
            ex.mtbf_por_equipamento(eventos(*linhas))
        return
    valor = tabela.iloc[0]["mtbf_dias"]
    assert (pd.isna(valor) if mtbf is None else valor == pytest.approx(mtbf))
    assert tabela.iloc[0]["n_falhas"] == len(dias_das_falhas)


def test_mtbf_por_equipamento_em_ordem():
    base = pd.Timestamp("2024-01-01")
    tabela = ex.mtbf_por_equipamento(eventos(
        evento("EQ002", base, "MC"), evento("EQ002", base + pd.Timedelta(days=4), "CP"),
        evento("EQ001", base, "MC"), evento("EQ001", base + pd.Timedelta(days=8), "MC")))
    assert tabela["tag_code"].tolist() == ["EQ001", "EQ002"]
    assert tabela["mtbf_dias"].tolist() == [8.0, 4.0]


def test_tendencia_crescente_do_tempo_de_reparo():
    linhas = [evento("EQ001", f"{ano}-06-01", "MC", hh=float(ano - 2018)) for ano in range(2018, 2024)]
    tabela = ex.tendencia_tempo_reparo(eventos(*linhas))
    assert tabela.iloc[0]["inclinacao_h_por_ano"] == pytest.approx(1.0)


def test_tendencia_estavel():
    linhas = [evento("EQ001", f"{ano}-06-01", "MC", hh=2.0) for ano in range(2018, 2024)]
    assert ex.tendencia_tempo_reparo(eventos(*linhas)).iloc[0]["inclinacao_h_por_ano"] == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("datas", [
    ["2024-01-01", "2024-02-01"],                 # menos de 3 eventos
    ["2024-01-01", "2024-05-01", "2024-09-01"],   # tudo no mesmo ano
])
def test_tendencia_sem_dados_suficientes_fica_vazia(datas):
    tabela = ex.tendencia_tempo_reparo(eventos(*[evento("EQ001", d, "MC") for d in datas]))
    assert tabela.iloc[0]["inclinacao_h_por_ano"] is None


def test_sazonalidade_sempre_tem_os_doze_meses():
    tabela = ex.sazonalidade_mensal(eventos(evento("EQ001", "2024-03-10", "MC"),
                                            evento("EQ002", "2023-03-02", "CP"),
                                            evento("EQ001", "2024-07-01", "MP")))
    assert tabela["mes"].tolist() == list(range(1, 13))
    assert tabela["mes_nome"].iloc[0].lower().startswith("jan")
    assert tabela.set_index("mes").loc[3, "falhas"] == 2
    assert tabela.set_index("mes").loc[7, "falhas"] == 0  # preventiva não conta
    assert tabela["falhas"].sum() == 2


def test_comparacao_consolidada():
    base = pd.Timestamp("2024-01-01")
    tabela = ex.comparacao_consolidada(eventos(
        evento("EQ001", base, "MC", hh=2.0),
        evento("EQ001", base + pd.Timedelta(days=10), "MC", hh=4.0),
        evento("EQ001", base + pd.Timedelta(days=20), "MP", hh=6.0),
        evento("EQ002", base, "MP", hh=1.0),
    )).set_index("tag_code")
    assert tabela.loc["EQ001", "n_eventos"] == 3
    assert tabela.loc["EQ001", "n_falhas"] == 2
    assert tabela.loc["EQ001", "pct_eventos_corretivos"] == pytest.approx(66.7)
    assert tabela.loc["EQ001", "tempo_medio_reparo_h"] == pytest.approx(4.0)
    assert tabela.loc["EQ001", "mtbf_dias"] == pytest.approx(10.0)
    assert tabela.loc["EQ002", "pct_eventos_corretivos"] == 0
    assert pd.isna(tabela.loc["EQ002", "mtbf_dias"])
