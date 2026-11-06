"""
Limpeza da planilha bruta e classificação do tipo de manutenção.
"""
import pandas as pd
import pytest

import config
from app.data import ingest
from app.processing.tipos_manutencao import descricao_tipo


def _bruto(**colunas):
    base = {
        "Início": [" 31/12/2025"],
        "Situação": ["C"],
        "Tipo de Manutenção": ["MC"],
        "TAG": ["MC01"],
        "Área": ["A"],
        "Sist Oper": ["S"],
    }
    base.update(colunas)
    return pd.DataFrame(base)


# ---------- Situação ----------

@pytest.mark.parametrize("situacao, fica", [
    ("C", True),
    (" C ", True),
    ("A", False),
    ("c", False),
    (None, False),
    ("", False),
])
def test_so_ordens_concluidas_entram(situacao, fica):
    assert (len(ingest._limpar(_bruto(**{"Situação": [situacao]}))) == 1) is fica


# ---------- Datas ----------

@pytest.mark.parametrize("texto, esperado", [
    ("31/12/2025", "2025-12-31"),
    (" 31/12/2025", "2025-12-31"),
    ("01/01/2008  ", "2008-01-01"),
    ("29/02/2024", "2024-02-29"),
])
def test_datas_no_formato_brasileiro(texto, esperado):
    assert ingest._limpar(_bruto(**{"Início": [texto]}))["Início"].iloc[0] == pd.Timestamp(esperado)


@pytest.mark.parametrize("texto", ["31/02/2025", "2025-12-31", "12/31/2025", "ontem"])
def test_data_invalida_interrompe_a_ingestao(texto):
    with pytest.raises(ValueError):
        ingest._limpar(_bruto(**{"Início": [texto]}))


def test_sem_coluna_fim_o_evento_dura_um_dia():
    df = ingest._limpar(_bruto())
    assert df["Fim"].iloc[0] == df["Início"].iloc[0]


def test_com_coluna_fim_ela_e_respeitada():
    df = ingest._limpar(_bruto(Fim=[" 02/01/2026"]))
    assert df["Fim"].iloc[0] == pd.Timestamp("2026-01-02")


def test_coluna_malha_vazia_e_descartada():
    assert "Malha" not in ingest._limpar(_bruto(Malha=[None])).columns


def test_limpeza_nao_altera_a_planilha_original():
    bruto = _bruto()
    ingest._limpar(bruto)
    assert bruto["Início"].iloc[0] == " 31/12/2025"


# ---------- Escopo (monta-cargas) ----------

@pytest.mark.parametrize("tag, fica", [
    ("MC01", True),
    ("mc06", True),
    ("MC99", True),
    ("EL01", False),
    ("XMC01", False),
])
def test_filtro_de_escopo_pelo_prefixo_do_tag(tag, fica, monkeypatch):
    monkeypatch.setattr(config, "TAG_PREFIXOS_ESCOPO", ["MC"])
    assert (len(ingest._filtrar_escopo(pd.DataFrame({"TAG": [tag]}))) == 1) is fica


@pytest.mark.parametrize("prefixos", [[], None])
def test_sem_prefixo_configurado_entram_todos(prefixos, monkeypatch):
    monkeypatch.setattr(config, "TAG_PREFIXOS_ESCOPO", prefixos)
    assert len(ingest._filtrar_escopo(pd.DataFrame({"TAG": ["MC01", "EL01"]}))) == 2


def test_varios_prefixos(monkeypatch):
    monkeypatch.setattr(config, "TAG_PREFIXOS_ESCOPO", ["MC", "EL"])
    assert len(ingest._filtrar_escopo(pd.DataFrame({"TAG": ["MC01", "EL01", "PT01"]}))) == 2


# ---------- Tipo de manutenção ----------

@pytest.mark.parametrize("tipo, falha, preventiva, classificado", [
    ("MC", True, False, True),
    ("CP", True, False, True),
    ("MP", False, True, True),
    ("IN", False, False, False),
    ("ME", False, False, False),
    ("EV", False, False, False),
    ("mc", False, False, False),
])
def test_classificacao_do_tipo(tipo, falha, preventiva, classificado):
    linha = ingest._classificar_tipo(pd.DataFrame({"Tipo de Manutenção": [tipo]})).iloc[0]
    assert (bool(linha["is_falha"]), bool(linha["is_preventiva"]), bool(linha["classificado"])) == \
        (falha, preventiva, classificado)


def test_falha_e_preventiva_nunca_ao_mesmo_tempo():
    df = ingest._classificar_tipo(pd.DataFrame({"Tipo de Manutenção": ["MC", "CP", "MP", "IN"]}))
    assert not (df["is_falha"] & df["is_preventiva"]).any()


@pytest.mark.parametrize("codigo, texto", [
    ("MC", "Corretiva"),
    ("CP", "Corretiva Programada"),
    ("MP", "Preventiva"),
    ("IN", "Outro (não classificado) — IN"),
    ("XX", "Outro (não classificado) — XX"),
])
def test_tipo_por_extenso(codigo, texto):
    assert descricao_tipo(codigo) == texto


# ---------- Pipeline completo da planilha fictícia ----------

def test_pipeline_remove_o_que_deve_sair(planilha_ficticia):
    df, _ = ingest.processar_planilha(planilha_ficticia, "Planilha1")
    # 60 linhas válidas + 4 que precisam sair (não concluída, situação vazia,
    # fora do escopo e duplicata)
    assert len(df) == 60
    assert df["Situação"].eq("C").all()
    assert df["TAG"].str.startswith("MC").all()
    assert not df.duplicated(subset=["Nº OS", "NºTarefa", "Início"]).any()


def test_pipeline_estima_componente_so_para_falhas(planilha_ficticia):
    df, _ = ingest.processar_planilha(planilha_ficticia, "Planilha1")
    assert df.loc[~df["is_falha"], "componente_estimado"].isna().all()
    assert df.loc[df["is_falha"], "componente_estimado"].notna().all()


def test_carga_no_banco_e_idempotente(banco_limpo, planilha_ficticia):
    from app.processing.dataset import carregar_eventos, carregar_maquinas

    df, mapa = ingest.processar_planilha(planilha_ficticia, "Planilha1")
    ingest.carregar_no_banco(df, mapa)
    ingest.carregar_no_banco(df, mapa)  # rodar de novo não duplica nada

    assert len(carregar_eventos()) == 60
    maquinas = carregar_maquinas()
    assert maquinas["tag_code"].tolist() == ["EQ001", "EQ002", "EQ003"]
    assert maquinas["total_eventos"].sum() == 60
