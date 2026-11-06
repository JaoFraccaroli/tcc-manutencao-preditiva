"""
Cadastro de máquinas e componentes pela aplicação (banco temporário).
"""
import pytest

from app.data import cadastro
from app.processing.dataset import carregar_componentes, carregar_maquina


def _form(**campos):
    base = {"tipo": "Monta-carga teste"}
    base.update(campos)
    return base


# ---------- Valores de fronteira e formatos ----------

@pytest.mark.parametrize("ano, valido", [
    ("1899", False),
    ("1900", True),
    ("2015", True),
    ("2100", True),
    ("2101", False),
    ("20a5", False),
    ("2015.5", False),
    ("", True),       # vazio = "não informado"
])
def test_ano_de_instalacao(ano, valido):
    campos, erro = cadastro._montar_campos_maquina(_form(ano_instalacao=ano))
    assert (erro is None) is valido
    if valido and ano:
        assert campos["ano_instalacao"] == int(ano)


@pytest.mark.parametrize("texto, valor", [
    ("1500", 1500.0),
    ("1,5", 1.5),
    ("1.5", 1.5),
    ("  800  ", 800.0),
    ("", None),
])
def test_capacidade_de_carga_aceita(texto, valor):
    campos, erro = cadastro._montar_campos_maquina(_form(capacidade_carga_kg=texto))
    assert erro is None
    assert campos["capacidade_carga_kg"] == valor


@pytest.mark.parametrize("texto", ["mil kg", "1.500,00", "abc"])
def test_capacidade_de_carga_invalida(texto):
    _, erro = cadastro._montar_campos_maquina(_form(capacidade_carga_kg=texto))
    assert "Capacidade de carga inválida" in erro


@pytest.mark.parametrize("status, valido", [
    ("Ativo", True), ("Em manutenção", True), ("Inativo", True),
    ("ativo", False), ("Quebrado", False),
])
def test_status(status, valido):
    _, erro = cadastro._montar_campos_maquina(_form(status=status))
    assert (erro is None) is valido


def test_status_vazio_vira_ativo():
    campos, _ = cadastro._montar_campos_maquina(_form(status=""))
    assert campos["status"] == "Ativo"


def test_campo_em_branco_fica_nao_informado():
    campos, _ = cadastro._montar_campos_maquina(_form(fabricante="   ", modelo=None))
    assert campos["fabricante"] is None and campos["modelo"] is None


# ---------- Inserir e editar máquina ----------

def test_inserir_maquina(banco_limpo):
    assert cadastro.inserir_maquina(" EQ900 ", _form(fabricante="Fabricante X", ano_instalacao="2015")) is None
    maquina = carregar_maquina("EQ900")
    assert maquina["fabricante"] == "Fabricante X"
    assert maquina["ano_instalacao"] == 2015
    assert maquina["total_eventos"] == 0


@pytest.mark.parametrize("tag", ["", "   ", None])
def test_inserir_sem_tag(banco_limpo, tag):
    assert cadastro.inserir_maquina(tag, _form()) == "Informe o código (TAG) da máquina."


def test_inserir_sem_nome(banco_limpo):
    assert cadastro.inserir_maquina("EQ900", {"tipo": ""}) == "Informe o nome/apelido do equipamento."


def test_tag_repetida_nao_grava_duplicado(banco_limpo):
    assert cadastro.inserir_maquina("EQ900", _form()) is None
    assert "Já existe" in cadastro.inserir_maquina("EQ900", _form(fabricante="Outro"))
    assert carregar_maquina("EQ900")["fabricante"] is None


def test_erro_de_validacao_nao_grava_nada(banco_limpo):
    assert "Ano" in cadastro.inserir_maquina("EQ900", _form(ano_instalacao="1800"))
    assert carregar_maquina("EQ900") is None


def test_editar_maquina_nao_mexe_nos_dados_da_ingestao(banco_com_dados):
    antes = carregar_maquina("EQ001")
    assert cadastro.atualizar_maquina("EQ001", _form(status="Em manutenção", modelo="M-1")) is None
    depois = carregar_maquina("EQ001")
    assert depois["status"] == "Em manutenção"
    assert depois["modelo"] == "M-1"
    for campo in ["primeiro_evento", "ultimo_evento", "total_eventos"]:
        assert depois[campo] == antes[campo]


def test_editar_maquina_que_nao_existe(banco_limpo):
    assert cadastro.atualizar_maquina("EQ999", _form()) == 'Máquina "EQ999" não encontrada.'


def test_editar_com_dado_invalido_nao_grava(banco_com_dados):
    assert "Status inválido" in cadastro.atualizar_maquina("EQ001", _form(status="Quebrado"))
    assert carregar_maquina("EQ001")["status"] != "Quebrado"


# ---------- Componentes ----------

def test_inserir_componente(banco_limpo):
    assert cadastro.inserir_componente("Botoeira", fabricante_referencia="  ") is None
    linha = carregar_componentes().set_index("nome").loc["Botoeira"]
    assert linha["fabricante_referencia"] is None


@pytest.mark.parametrize("nome", ["", "   ", None])
def test_componente_sem_nome(banco_limpo, nome):
    assert cadastro.inserir_componente(nome) == "Informe o nome do componente."


@pytest.mark.parametrize("repetido", ["Botoeira", "botoeira", "BOTOEIRA", " Botoeira "])
def test_componente_repetido_ignora_maiusculas(banco_limpo, repetido):
    cadastro.inserir_componente("Botoeira")
    assert "Já existe" in cadastro.inserir_componente(repetido)
    assert len(carregar_componentes()) == 1
