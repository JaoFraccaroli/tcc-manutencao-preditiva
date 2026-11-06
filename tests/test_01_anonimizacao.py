"""
Anonimização (LGPD) — a parte mais sensível do TCC.

Garante que nenhum dado identificável chega ao banco: TAG real, nome de
técnico, nome de empresa, matrícula, centro de custo e o HH original.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from app.data import ingest
from tests import fabrica

DADOS = Path(__file__).parent / "dados"


def _casos(arquivo):
    with open(DADOS / arquivo, encoding="utf-8") as f:
        return [pytest.param(*linha.values(), id=linha["regra"] if "regra" in linha else None)
                for linha in csv.DictReader(f, delimiter=";")]


# ---------- Nome de técnico / empresa no fim da descrição (dirigido por dados) ----------

@pytest.mark.parametrize("original, esperado, regra", _casos("casos_atribuicao.csv"))
def test_remove_atribuicao_no_fim_da_descricao(original, esperado, regra):
    assert ingest._remover_atribuicao_final(original) == esperado


@pytest.mark.parametrize("nome", sorted(ingest.NOMES_ATRIBUICAO_CONHECIDOS))
def test_remove_cada_nome_conhecido_mesmo_sem_separador(nome):
    texto = f"monta carga nao sobe {nome}"
    resultado = ingest._remover_atribuicao_final(texto)
    assert nome.lower() not in resultado.lower()
    assert resultado == "monta carga nao sobe"


@pytest.mark.parametrize("nome", sorted(ingest.NOMES_ATRIBUICAO_CONHECIDOS))
def test_nome_conhecido_em_maiusculas_tambem_sai(nome):
    assert nome.upper() not in ingest._remover_atribuicao_final(f"porta travada {nome.upper()}").upper()


def test_nome_conhecido_dentro_de_outra_palavra_nao_e_cortado():
    # O nome só sai como palavra inteira: não pode "comer" pedaço de outra palavra.
    nome = sorted(ingest.NOMES_ATRIBUICAO_CONHECIDOS)[0]
    palavra = "x" + nome + "x"
    assert palavra in ingest._remover_atribuicao_final(f"peça {palavra} trocada")


def test_dois_nomes_separados_por_barra_saem_os_dois():
    a, b = sorted(ingest.NOMES_ATRIBUICAO_CONHECIDOS)[:2]
    resultado = ingest._remover_atribuicao_final(f"porta travada {a}/{b}")
    assert resultado == "porta travada"


# ---------- TAG real escrito no meio do texto ----------

@pytest.mark.parametrize("texto", [
    "Lâmpada do MC01 queimada",
    "Lâmpada do mc01 queimada",
    "Lâmpada do MC 01 queimada",
    "Lâmpada do MC  01 queimada",
    "MC01: porta travada",
    "porta do MC01, MC02 e MC03",
])
def test_tag_real_no_texto_vira_marcador(texto):
    df = pd.DataFrame({"TAG": ["MC01", "MC02", "MC03"], "Descrição Tarefa": [texto] * 3})
    redigido = ingest._redigir_descricao(df)["descricao_tarefa_redigida"].iloc[0]
    assert "[equip]" in redigido
    for tag in ["MC01", "MC02", "MC03", "MC 01"]:
        assert tag.lower() not in redigido.lower()


def test_tag_de_outro_equipamento_tambem_e_redigido():
    df = pd.DataFrame({"TAG": ["MC01", "MC02"], "Descrição Tarefa": ["igual ao defeito do MC02", "ok"]})
    assert ingest._redigir_descricao(df)["descricao_tarefa_redigida"].iloc[0] == "igual ao defeito do [equip]"


def test_numero_parecido_com_tag_nao_e_redigido():
    # "MC010" não é o MC01: não pode ser trocado pelo marcador.
    df = pd.DataFrame({"TAG": ["MC01"], "Descrição Tarefa": ["peça MC010 trocada"]})
    assert ingest._redigir_descricao(df)["descricao_tarefa_redigida"].iloc[0] == "peça MC010 trocada"


def test_descricao_vazia_continua_vazia():
    df = pd.DataFrame({"TAG": ["MC01"], "Descrição Tarefa": [None]})
    assert ingest._redigir_descricao(df)["descricao_tarefa_redigida"].iloc[0] is None


def test_planilha_sem_coluna_de_descricao():
    df = pd.DataFrame({"TAG": ["MC01"]})
    assert ingest._redigir_descricao(df)["descricao_tarefa_redigida"].iloc[0] is None


@pytest.mark.parametrize("tag, padrao_casa, padrao_nao_casa", [
    ("MC02", ["MC02", "MC 02", "MC-02", "MC - 02"], ["MC2", "MC03"]),
    ("ABC", ["ABC"], ["AB C"]),
])
def test_padrao_flexivel_do_tag(tag, padrao_casa, padrao_nao_casa):
    import re
    padrao = re.compile(r"^" + ingest._padrao_tag_flexivel(tag) + r"$")
    for texto in padrao_casa:
        assert padrao.match(texto), texto
    for texto in padrao_nao_casa:
        assert not padrao.match(texto), texto


@pytest.mark.defeito
@pytest.mark.xfail(strict=True, reason=(
    "DEFEITO: a remoção do nome do técnico corta tudo depois do último hífen ANTES de "
    "redigir o TAG. Com o TAG escrito com hífen ('MC-02'), a descrição perde o resto da frase: "
    "'Lâmpada do MC-02 queimada' vira 'Lâmpada do MC'."))
def test_tag_com_hifen_nao_apaga_o_resto_da_descricao():
    df = pd.DataFrame({"TAG": ["MC02"], "Descrição Tarefa": ["Lâmpada do MC-02 queimada"]})
    assert ingest._redigir_descricao(df)["descricao_tarefa_redigida"].iloc[0] == "Lâmpada do [equip] queimada"


def test_tag_com_hifen_pelo_menos_nao_vaza_o_numero():
    # Mesmo com o defeito acima, o TAG completo não aparece no resultado.
    df = pd.DataFrame({"TAG": ["MC02"], "Descrição Tarefa": ["Lâmpada do MC-02 queimada"]})
    redigido = ingest._redigir_descricao(df)["descricao_tarefa_redigida"].iloc[0]
    assert "02" not in redigido


@pytest.mark.xfail(strict=True, reason=(
    "LIMITAÇÃO: tudo depois do último hífen é tratado como nome de técnico. "
    "Uma descrição legítima com hífen perde a parte final."))
def test_hifen_no_meio_da_descricao_nao_perde_informacao():
    assert ingest._remover_atribuicao_final("porta não abre - sensor com defeito") == \
        "porta não abre - sensor com defeito"


# ---------- TAG -> EQ00N ----------

def test_tag_vira_codigo_sequencial_em_ordem_alfabetica():
    df = pd.DataFrame({"TAG": ["MC03", "MC01", "MC02", "MC01"]})
    resultado, mapa = ingest._anonimizar_tag(df)
    assert mapa == {"MC01": "EQ001", "MC02": "EQ002", "MC03": "EQ003"}
    assert resultado["tag_code"].tolist() == ["EQ003", "EQ001", "EQ002", "EQ001"]


def test_codigo_e_sempre_o_mesmo_em_execucoes_diferentes():
    df = pd.DataFrame({"TAG": ["MC02", "MC01"]})
    assert ingest._anonimizar_tag(df)[1] == ingest._anonimizar_tag(df.iloc[::-1])[1]


def test_codigo_tem_tres_digitos_mesmo_com_muitos_equipamentos():
    df = pd.DataFrame({"TAG": [f"MC{i:03d}" for i in range(1, 121)]})
    mapa = ingest._anonimizar_tag(df)[1]
    assert mapa["MC001"] == "EQ001"
    assert mapa["MC120"] == "EQ120"


@pytest.mark.xfail(strict=True, reason=(
    "RISCO: o TAG não é padronizado em maiúsculas antes da anonimização. "
    "Se a planilha trouxer 'mc01' e 'MC01', o mesmo equipamento vira dois códigos."))
def test_mesmo_tag_com_letras_diferentes_vira_um_codigo_so():
    df = pd.DataFrame({"TAG": ["MC01", "mc01"]})
    assert ingest._anonimizar_tag(df)[0]["tag_code"].nunique() == 1


# ---------- HH perturbado ----------

def test_hh_fica_entre_90_e_110_por_cento_do_original():
    original = np.linspace(0.5, 10, 500)
    df = pd.DataFrame({"HH Apropriado": original})
    perturbado = ingest._perturbar_hh(df)["hh_apropriado"].to_numpy()
    razao = perturbado / original
    assert razao.min() >= config.PERTURBACAO_MIN
    assert razao.max() <= config.PERTURBACAO_MAX


def test_hh_perturbado_e_diferente_do_original():
    original = np.linspace(1, 10, 100)
    perturbado = ingest._perturbar_hh(pd.DataFrame({"HH Apropriado": original}))["hh_apropriado"].to_numpy()
    assert (perturbado != original).sum() >= 99


def test_perturbacao_e_reproduzivel_com_a_mesma_seed():
    df = pd.DataFrame({"HH Apropriado": [1.0, 2.0, 3.0]})
    a = ingest._perturbar_hh(df)["hh_apropriado"].tolist()
    b = ingest._perturbar_hh(df)["hh_apropriado"].tolist()
    assert a == b


def test_perturbacao_nao_altera_a_planilha_de_entrada():
    df = pd.DataFrame({"HH Apropriado": [1.0, 2.0]})
    ingest._perturbar_hh(df)
    assert df["HH Apropriado"].tolist() == [1.0, 2.0]
    assert "hh_apropriado" not in df.columns


# ---------- Pipeline completo: nada identificável sobra ----------

def test_planilha_processada_nao_tem_nenhum_dado_identificavel(planilha_ficticia):
    df, mapa = ingest.processar_planilha(planilha_ficticia, "Planilha1")
    texto = " ".join(df["descricao_tarefa_redigida"].dropna()).lower()

    for tag in fabrica.TAGS_REAIS_FICTICIOS:
        assert tag.lower() not in texto
        assert tag[:2].lower() + " " + tag[2:] not in texto
    assert fabrica.TECNICO_FICTICIO.lower() not in texto
    assert set(df["tag_code"]) == {"EQ001", "EQ002", "EQ003"}


def test_banco_nao_guarda_colunas_identificaveis(banco_com_dados):
    from sqlalchemy import inspect

    from app.data.db import engine

    colunas = {c["name"].lower() for c in inspect(engine).get_columns("eventos_manutencao")}
    for proibida in ["centro", "matricula", "ofic", "tag_original", "descricao_original"]:
        assert not any(proibida in c for c in colunas), proibida


def test_tag_real_so_fica_na_tabela_interna_de_mapeamento(banco_com_dados):
    from app.processing.dataset import carregar_eventos

    eventos = carregar_eventos()
    tudo = eventos.astype(str).to_string().lower()
    for tag in fabrica.TAGS_REAIS_FICTICIOS:
        assert tag.lower() not in tudo
    assert fabrica.TECNICO_FICTICIO.lower() not in tudo
