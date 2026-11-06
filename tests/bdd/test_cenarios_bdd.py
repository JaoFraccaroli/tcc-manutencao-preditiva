"""
Passos dos cenários BDD (pytest-bdd, Gherkin em português).
"""
import numpy as np
import pandas as pd
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from app.data import ingest
from app.modeling.predict import data_sugerida_alerta
from app.presentation.routes import _frase_data_sugerida
from app.processing import dataset
from tests import fabrica
from tests.fabrica import evento, eventos

scenarios("anonimizacao.feature", "rotulo_de_falha.feature", "data_de_alerta.feature")


@pytest.fixture
def contexto():
    return {}


def _data_br(texto):
    return pd.Timestamp(pd.to_datetime(texto, format="%d/%m/%Y"))


# ---------- Anonimização ----------

@given(parsers.parse('a descrição "{texto}" do equipamento "{tag}"'))
def descricao(contexto, texto, tag):
    contexto["df"] = pd.DataFrame({"TAG": [tag], "Descrição Tarefa": [texto]})


@when("a descrição é anonimizada")
def anonimizar(contexto):
    contexto["resultado"] = ingest._redigir_descricao(contexto["df"])["descricao_tarefa_redigida"].iloc[0]


@then(parsers.parse('a descrição gravada deve ser "{esperado}"'))
def descricao_gravada(contexto, esperado):
    assert contexto["resultado"] == esperado


@given("uma planilha fictícia com TAGs reais e nomes de técnicos")
def planilha(contexto, planilha_ficticia):
    contexto["planilha"] = planilha_ficticia


@when("a planilha é processada")
def processar(contexto):
    contexto["df"], contexto["mapa"] = ingest.processar_planilha(contexto["planilha"], "Planilha1")
    contexto["texto"] = " ".join(contexto["df"]["descricao_tarefa_redigida"].dropna()).lower()


@then("nenhum TAG real aparece no resultado")
def sem_tag(contexto):
    for tag in fabrica.TAGS_REAIS_FICTICIOS:
        assert tag.lower() not in contexto["texto"]


@then("nenhum nome de técnico aparece no resultado")
def sem_nome(contexto):
    assert fabrica.TECNICO_FICTICIO.lower() not in contexto["texto"]


@then(parsers.parse('os equipamentos ficam com os códigos "{codigos}"'))
def codigos(contexto, codigos):
    assert sorted(contexto["df"]["tag_code"].unique()) == [c.strip() for c in codigos.split(",")]


@then("as horas apropriadas ficam entre 90% e 110% das originais")
def horas(contexto):
    razao = contexto["df"]["hh_apropriado"] / contexto["df"]["HH Apropriado"]
    assert razao.between(0.9, 1.1).all()


# ---------- Rótulo ----------

@given(parsers.parse('uma data de referência em "{data}"'))
def referencia(contexto, data):
    contexto["ref"] = _data_br(data)


@given(parsers.parse("uma falha {dias:d} dias depois da referência"))
def falha_depois(contexto, dias):
    contexto["falhas"] = np.array([np.datetime64(contexto["ref"] + pd.Timedelta(days=dias))])


@when(parsers.parse("o rótulo de {horizonte:d} dias é calculado"))
def calcular_rotulo(contexto, horizonte):
    contexto["rotulo"] = bool(dataset._rotular(contexto["falhas"], pd.DatetimeIndex([contexto["ref"]]), horizonte)[0])


@then(parsers.parse('o rótulo deve ser "{esperado}"'))
def rotulo_esperado(contexto, esperado):
    assert contexto["rotulo"] is (esperado == "com falha")


@given("um equipamento com histórico de falhas até a referência")
def historico(contexto):
    contexto["ref"] = pd.Timestamp("2024-01-01")
    contexto["eventos"] = eventos(*[evento("EQ001", contexto["ref"] - pd.Timedelta(days=d), "MC") for d in (5, 40, 120)])
    contexto["antes"] = dataset._features_para_grupo(contexto["eventos"], pd.DatetimeIndex([contexto["ref"]]))


@when("acontecem novas falhas depois da referência")
def futuro(contexto):
    novos = eventos(*[evento("EQ001", contexto["ref"] + pd.Timedelta(days=d), "MC", hh=99) for d in (1, 2, 3)])
    contexto["eventos"] = pd.concat([contexto["eventos"], novos]).sort_values("inicio").reset_index(drop=True)


@then("as variáveis calculadas na referência continuam iguais")
def iguais(contexto):
    depois = dataset._features_para_grupo(contexto["eventos"], pd.DatetimeIndex([contexto["ref"]]))
    pd.testing.assert_frame_equal(contexto["antes"], depois)


# ---------- Data de alerta ----------

@given(parsers.parse("as probabilidades de falha {p30:f} em 30 dias, {p60:f} em 60 dias e {p90:f} em 90 dias"))
def probabilidades(contexto, p30, p60, p90):
    contexto["linha"] = pd.Series({"prob_falha_30d": p30, "prob_falha_60d": p60, "prob_falha_90d": p90})


@when(parsers.parse('a data sugerida é calculada a partir de "{data}"'))
def calcular_alerta(contexto, data):
    contexto["alerta"] = data_sugerida_alerta(contexto["linha"], _data_br(data))


@then(parsers.parse('o alerta deve ser "{esperado}"'))
def alerta_esperado(contexto, esperado):
    alerta = contexto["alerta"]
    obtido = "sem alerta" if alerta is None else "em " + alerta["data"].strftime("%d/%m/%Y")
    assert obtido == esperado


@then(parsers.parse('a frase deve ser "{frase}"'))
def frase(contexto, frase):
    a = contexto["alerta"]
    linha = {"data_sugerida_alerta": a["data"], "horizonte_sugerido": a["horizonte"],
             **contexto["linha"].to_dict()}
    assert _frase_data_sugerida(linha) == frase
