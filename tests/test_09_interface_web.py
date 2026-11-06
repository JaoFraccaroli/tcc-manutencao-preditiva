"""
Aplicação Flask, testada pelas páginas (como o usuário vê), com banco
fictício e um modelo falso no lugar dos arquivos .joblib.
"""
import datetime

import pandas as pd
import pytest

from app.presentation import _data_br
from tests import fabrica

PAGINAS = [
    "/",
    "/equipamentos",
    "/equipamentos/EQ001",
    "/modelos",
    "/analise-textual",
    "/analise-exploratoria",
    "/maquinas",
    "/maquinas/EQ001/editar",
    "/componentes",
    "/componentes/exemplos?nome=Portas / travas",
]


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_abre(cliente, pagina):
    assert cliente.get(pagina).status_code == 200


@pytest.mark.parametrize("pagina", PAGINAS)
def test_nenhuma_pagina_mostra_dado_identificavel(cliente, pagina):
    html = cliente.get(pagina).get_data(as_text=True)
    for tag in fabrica.TAGS_REAIS_FICTICIOS:
        assert tag not in html
        assert f"{tag[:2]} {tag[2:]}" not in html
    assert fabrica.TECNICO_FICTICIO not in html
    assert "998877" not in html  # matrícula
    assert "CC-123" not in html  # centro de custo


@pytest.mark.parametrize("pagina", ["/equipamentos/EQ999", "/maquinas/EQ999/editar"])
def test_equipamento_inexistente_da_404(cliente, pagina):
    assert cliente.get(pagina).status_code == 404


def test_lista_de_equipamentos_nao_e_ordenada_por_risco(cliente):
    html = cliente.get("/equipamentos").get_data(as_text=True)
    assert html.index("EQ001") < html.index("EQ002") < html.index("EQ003")


def test_usuario_pode_ordenar_pela_probabilidade(cliente):
    assert cliente.get("/equipamentos?sort=prob_30d&dir=desc").status_code == 200


def test_ordenacao_desconhecida_volta_para_o_codigo(cliente):
    html = cliente.get("/equipamentos?sort=coluna_que_nao_existe").get_data(as_text=True)
    assert html.index("EQ001") < html.index("EQ002")


@pytest.mark.parametrize("pagina", ["0", "-5", "99"])
def test_pagina_fora_do_intervalo_nao_quebra(cliente, pagina):
    assert cliente.get(f"/equipamentos?page={pagina}").status_code == 200


@pytest.mark.defeito
@pytest.mark.xfail(strict=True, reason=(
    "DEFEITO: /equipamentos?page=abc faz int('abc') sem tratamento e a página dá erro 500. "
    "Deveria voltar para a página 1."))
def test_pagina_com_texto_nao_derruba_a_aplicacao(cliente):
    assert cliente.get("/equipamentos?page=abc").status_code == 200


def test_cadastrar_maquina_pela_tela(cliente):
    resposta = cliente.post("/maquinas", data={"tag_code": "EQ900", "tipo": "Monta-carga novo",
                                               "status": "Ativo"}, follow_redirects=True)
    html = resposta.get_data(as_text=True)
    assert "cadastrada com sucesso" in html
    assert "EQ900" in html


def test_cadastro_invalido_mostra_o_erro_na_tela(cliente):
    resposta = cliente.post("/maquinas", data={"tag_code": "EQ001", "tipo": "Repetida"}, follow_redirects=True)
    assert "Já existe" in resposta.get_data(as_text=True)


def test_editar_maquina_pela_tela(cliente):
    resposta = cliente.post("/maquinas/EQ001/editar", data={"tipo": "Monta-carga", "fabricante": "Fabricante Y",
                                                             "status": "Ativo"})
    assert resposta.status_code == 302
    assert resposta.headers["Location"].endswith("/equipamentos/EQ001")
    assert "Fabricante Y" in cliente.get("/equipamentos/EQ001").get_data(as_text=True)


def test_cadastrar_componente_pela_tela(cliente):
    resposta = cliente.post("/componentes", data={"nome": "Botoeira"}, follow_redirects=True)
    assert "cadastrado com sucesso" in resposta.get_data(as_text=True)


def test_detalhe_explica_a_data_sugerida(cliente):
    html = cliente.get("/equipamentos/EQ001").get_data(as_text=True)
    assert ("ultrapassa o limiar" in html) or ("Nenhum horizonte avaliado" in html)


def test_atualizar_limpa_o_cache(cliente):
    cliente.get("/")
    assert cliente.get("/atualizar").get_json() == {"status": "cache limpo"}


# ---------- Datas no formato brasileiro ----------

@pytest.mark.parametrize("valor, esperado", [
    (datetime.date(2025, 1, 31), "31/01/2025"),
    (datetime.datetime(2025, 1, 31, 14, 30), "31/01/2025"),
    (pd.Timestamp("2025-01-31"), "31/01/2025"),
    ("2025-01-31", "31/01/2025"),
    (None, "—"),
    (pd.NaT, "—"),
    (float("nan"), "—"),
    ("texto qualquer", "texto qualquer"),
])
def test_data_no_formato_brasileiro(valor, esperado):
    assert _data_br(valor) == esperado
