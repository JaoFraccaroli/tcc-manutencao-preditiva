"""
Estimativa de componente por palavra-chave e categorização do texto
livre das falhas.
"""
import csv
from pathlib import Path

import pandas as pd
import pytest

from app.processing import component_mapping as cm
from app.processing import text_analysis as ta
from tests.fabrica import evento, eventos

DADOS = Path(__file__).parent / "dados"


def _casos_componentes():
    with open(DADOS / "casos_componentes.csv", encoding="utf-8") as f:
        return [pytest.param(l["descricao"], l["componente_esperado"], id=l["por_que"] + " | " + l["descricao"])
                for l in csv.DictReader(f, delimiter=";")]


@pytest.mark.parametrize("descricao, esperado", _casos_componentes())
def test_componente_estimado(descricao, esperado):
    assert cm.classificar_componente(descricao) == esperado


@pytest.mark.parametrize("vazio", [None, "", float("nan")])
def test_descricao_vazia_nao_e_classificada(vazio):
    assert cm.classificar_componente(vazio) == cm.NAO_IDENTIFICADO


def test_todo_componente_tem_termos_e_ficha_no_catalogo():
    assert set(cm.TERMOS_COMPONENTE) == set(cm.CATALOGO_COMPONENTES)
    for nome, termos in cm.TERMOS_COMPONENTE.items():
        assert termos, nome
        ficha = cm.CATALOGO_COMPONENTES[nome]
        for campo in ["fabricante_referencia", "vida_util_indicada", "unidade", "criterio_substituicao"]:
            assert ficha[campo], (nome, campo)


@pytest.mark.parametrize("componente", list(cm.TERMOS_COMPONENTE))
def test_cada_termo_reconhece_o_proprio_componente_ou_um_anterior(componente):
    ordem = list(cm.TERMOS_COMPONENTE)
    for termo in cm.TERMOS_COMPONENTE[componente]:
        resultado = cm.classificar_componente(f"verificar {termo} hoje")
        assert resultado in ordem[: ordem.index(componente) + 1], termo


def test_normalizacao_tira_acento_maiuscula_e_marcador():
    assert cm._normalizar("[equip] Painél ÇÃO") == "  painel cao"


def test_classificar_serie():
    serie = pd.Series(["motor", "porta", None])
    assert cm.classificar_serie(serie).tolist() == ["Motor elétrico", "Portas / travas", cm.NAO_IDENTIFICADO]


def _eventos_com_componentes():
    linhas = [
        evento("EQ001", "2024-01-01", "MC", descricao="porta"),
        evento("EQ001", "2024-01-05", "MC", descricao="porta"),
        evento("EQ001", "2024-01-09", "MC", descricao="motor"),
        evento("EQ001", "2024-01-12", "MC", descricao="parou"),
        evento("EQ001", "2024-01-15", "MP", descricao="porta"),
    ]
    df = eventos(*linhas)
    df.loc[df["is_falha"], "componente_estimado"] = df.loc[df["is_falha"], "descricao_tarefa"].map(
        cm.classificar_componente)
    return df


def test_tabela_resumo_usa_so_falhas():
    tabela = cm.tabela_resumo(_eventos_com_componentes())
    assert tabela["ocorrencias"].sum() == 4
    assert tabela.iloc[0].to_dict() == {"componente": "Portas / travas", "ocorrencias": 2, "percentual": 50.0}
    assert tabela["percentual"].sum() == pytest.approx(100.0)


def test_tabela_resumo_sem_falhas():
    sem_falha = eventos(evento("EQ001", "2024-01-01", "MP"))
    assert cm.tabela_resumo(sem_falha).empty


def test_exemplos_do_mais_recente_para_o_mais_antigo():
    exemplos = cm.exemplos_por_componente(_eventos_com_componentes(), "Portas / travas")
    assert exemplos["inicio"].tolist() == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-01")]
    assert list(exemplos.columns) == ["tag_code", "inicio", "descricao_tarefa"]


@pytest.mark.parametrize("limite", [0, 1, 2, 10])
def test_limite_de_exemplos(limite):
    assert len(cm.exemplos_por_componente(_eventos_com_componentes(), "Portas / travas", limite)) == min(limite, 2)


# ---------- Categorização do texto livre ----------

def test_normalizacao_do_texto():
    assert ta._normalizar("Porta NÃO abre [equip]!") .split() == ["porta", "nao", "abre"]


def test_stopwords_e_palavras_do_equipamento_saem():
    tokens = ta._tokenizar(ta._normalizar("o monta carga da linha parou de funcionar"))
    assert "monta" not in tokens and "carga" not in tokens and "o" not in tokens
    assert "parou" in tokens


def test_bigramas():
    assert ta._bigramas(["porta", "nao", "abre"]) == ["porta nao", "nao abre"]
    assert ta._bigramas(["porta"]) == []


def test_so_descricoes_de_falha_entram_na_analise():
    df = eventos(evento("EQ001", "2024-01-01", "MC", descricao="porta"),
                 evento("EQ001", "2024-01-02", "MP", descricao="lubrificar"),
                 evento("EQ001", "2024-01-03", "MC", descricao=None))
    assert ta.carregar_descricoes_falha(df).tolist() == ["porta"]


def test_frequencia_conta_cada_descricao_uma_vez():
    tabela = ta.frequencia_palavras(pd.Series(["porta porta porta", "porta travada"]))
    assert tabela.set_index("palavra").loc["porta", "ocorrencias"] == 2


def test_categorias_somam_cem_por_cento():
    textos = ["porta nao abre"] * 6 + ["parou de funcionar"] * 3 + ["barulho estranho"]
    df = eventos(*[evento("EQ001", f"2024-01-{i + 1:02d}", "MC", descricao=t) for i, t in enumerate(textos)])
    tabela, sementes = ta.categorizar(df, n_categorias=3, fracao_minima=0.2)
    assert tabela["ocorrencias"].sum() == 10
    assert tabela["percentual"].sum() == pytest.approx(100.0, abs=0.2)
    # "de" é stopword: o bigrama fica "parou funcionar"
    assert "parou funcionar" in sementes["padrao"].tolist()
    # Só "barulho estranho" (1 vez, abaixo do mínimo de 20%) fica sem categoria
    assert tabela.set_index("categoria").loc[ta.NAO_IDENTIFICADO, "ocorrencias"] == 1


def test_bigrama_tem_prioridade_sobre_palavra_isolada():
    textos = ["parou de funcionar"] * 5 + ["parou"] * 5
    df = eventos(*[evento("EQ001", f"2024-01-{i + 1:02d}", "MC", descricao=t) for i, t in enumerate(textos)])
    _, sementes = ta.categorizar(df, n_categorias=2, fracao_minima=0.1)
    assert sementes["padrao"].tolist()[0] == "parou funcionar"


def test_sem_descricoes_de_falha_devolve_tabelas_vazias():
    df = eventos(evento("EQ001", "2024-01-01", "MP", descricao="lubrificar"))
    tabela, sementes = ta.categorizar(df)
    assert tabela.empty and sementes.empty
