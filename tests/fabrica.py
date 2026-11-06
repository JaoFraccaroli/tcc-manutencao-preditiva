"""
Fábrica de dados FICTÍCIOS para os testes.

Nada aqui vem da planilha real: TAGs, nomes e descrições são inventados.
Os TAGs seguem o mesmo formato da planilha real (MC01, MC02...) para que
o código de anonimização seja exercitado do mesmo jeito.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Nomes de técnicos INVENTADOS (não constam da planilha real)
TECNICO_FICTICIO = "Zebedeu"
EMPRESA_FICTICIA = "Manutec Ficticia"

TAGS_REAIS_FICTICIOS = ["MC01", "MC02", "MC03"]


def _linha(os_num, tarefa, inicio, tag, tipo="MC", situacao="C", descricao="porta não abre",
           hh=2.0, area="Area A", sist="Sist 1"):
    return {
        "Nº OS": os_num,
        "NºTarefa": tarefa,
        "Início": inicio,
        "Situação": situacao,
        "Tipo de Manutenção": tipo,
        "TAG": tag,
        "Área": area,
        "Sist Oper": sist,
        "Descrição Tarefa": descricao,
        "HH Apropriado": hh,
        # Colunas identificáveis que a ingestão precisa DESCARTAR
        "Centro de Custo": "CC-123",
        "NºMatrícula": "998877",
        "Ofic Coordenadora": "Oficina Ficticia",
        "Ofic. Executante": "Equipe Ficticia",
        "Malha": None,
    }


def planilha_bruta() -> pd.DataFrame:
    """Planilha fictícia no formato da real (datas como texto dd/mm/aaaa,
    às vezes com espaço antes), com casos que a limpeza precisa tratar."""
    linhas = []
    os_num = 1000
    inicio = date(2020, 1, 6)
    descricoes = [
        "porta não abre - " + TECNICO_FICTICIO,
        "motor do MC01 queimado",
        "sensor da porta com defeito (ok " + TECNICO_FICTICIO + ")",
        "monta carga parou de funcionar",
        "cabo de aço desgastado",
        "Lâmpada do MC 02 queimada",
    ]
    for i in range(60):
        tag = TAGS_REAIS_FICTICIOS[i % 3]
        dia = inicio + timedelta(days=11 * i)
        tipo = ["MC", "MP", "CP", "MC", "IN"][i % 5]
        linhas.append(_linha(os_num + i, 1, " " + dia.strftime("%d/%m/%Y"), tag, tipo=tipo,
                             descricao=descricoes[i % len(descricoes)], hh=1.0 + (i % 4)))

    # OS não concluída (Situação != C) -> deve sair
    linhas.append(_linha(5000, 1, "10/10/2021", "MC01", situacao="A"))
    # Situação vazia -> deve sair
    linhas.append(_linha(5001, 1, "11/10/2021", "MC01", situacao=None))
    # Equipamento fora do escopo (não é monta-cargas) -> deve sair
    linhas.append(_linha(5002, 1, "12/10/2021", "EL01"))
    # Duplicata exata de (Nº OS, NºTarefa, Início) -> uma das duas sai
    linhas.append(_linha(1000, 1, " 06/01/2020", "MC01"))
    return pd.DataFrame(linhas)


def evento(tag, dia, tipo="MC", hh=1.0, area="Area A", sist="Sist 1", descricao="porta não abre"):
    """Uma linha no formato de `dataset.carregar_eventos()`."""
    falha = tipo in ("MC", "CP")
    return {
        "tag_code": tag,
        "inicio": pd.Timestamp(dia),
        "fim": pd.Timestamp(dia),
        "tipo_manutencao": tipo,
        "is_falha": falha,
        "is_preventiva": tipo == "MP",
        "classificado": tipo in ("MC", "CP", "MP"),
        "hh_apropriado": hh,
        "area": area,
        "sist_oper": sist,
        "descricao_tarefa": descricao,
        "componente_estimado": None,
    }


def eventos(*linhas) -> pd.DataFrame:
    df = pd.DataFrame(list(linhas))
    return df.sort_values(["tag_code", "inicio"]).reset_index(drop=True)


class ModeloFalso:
    """Modelo previsível: quanto mais falhas nos últimos 90 dias, maior a
    probabilidade. Horizontes maiores somam um pouco mais de risco."""

    def __init__(self, horizonte):
        self.horizonte = horizonte

    def predict_proba(self, X):
        base = 0.2 + 0.15 * X["freq_falhas_90d"].to_numpy(dtype=float) + (self.horizonte - 30) / 300
        p = np.clip(base, 0.0, 0.99)
        return np.column_stack([1 - p, p])
