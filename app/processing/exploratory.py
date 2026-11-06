"""
Análise exploratória e descritiva — MTBF por equipamento, tendência
do tempo de reparo ao longo do tempo, sazonalidade de falhas e
comparação consolidada entre equipamentos (briefing complementar de
análise exploratória, itens 2 a 5).

Módulo independente da modelagem preditiva (`app/modeling/`): nada aqui
vira feature do modelo de classificação, é conteúdo complementar
descritivo/exploratório.
"""
import numpy as np
import pandas as pd

MESES_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def mtbf_por_equipamento(eventos: pd.DataFrame) -> pd.DataFrame:
    """MTBF (tempo médio entre falhas) por equipamento, calculado sobre
    os eventos classificados como falha (MC/CP) — fundamentação teórica
    já discutida no capítulo 2 do TCC (Tsarouhas, 2020; Alsyouf et al.,
    2016).

    Fórmula: para cada equipamento, ordenam-se as datas "Início" dos
    eventos de falha e toma-se a média das diferenças entre datas
    consecutivas, em DIAS DE CALENDÁRIO (não horas de operação; a
    planilha não traz horímetro). Falhas no mesmo dia geram intervalo 0 e
    entram na média. Equipamentos com menos de 2 falhas ficam com
    `mtbf_dias = None` (intervalo indefinido, não zero)."""
    linhas = []
    for tag_code, grupo in eventos[eventos["is_falha"]].groupby("tag_code"):
        datas = grupo["inicio"].sort_values()
        n_falhas = len(datas)
        if n_falhas >= 2:
            mtbf = datas.diff().dropna().dt.days.mean()
        else:
            mtbf = None
        linhas.append({"tag_code": tag_code, "n_falhas": int(n_falhas), "mtbf_dias": mtbf})
    return pd.DataFrame(linhas).sort_values("tag_code").reset_index(drop=True)


def tendencia_tempo_reparo(eventos: pd.DataFrame) -> pd.DataFrame:
    """Regressão linear simples (ano-calendário x HH Apropriado) por
    equipamento. Inclinação positiva sugere tempo de reparo crescendo ao
    longo dos anos — sinal possível de degradação/fim de vida útil
    (briefing complementar, item 3). Usa todos os eventos do equipamento
    (não só falhas), já que tempo de reparo se aplica a qualquer
    intervenção. Equipamentos com poucos anos distintos de histórico
    (<3 pontos) ou variação nula no eixo do tempo ficam com
    `inclinacao_h_por_ano = None`."""
    linhas = []
    for tag_code, grupo in eventos.groupby("tag_code"):
        anos = grupo["inicio"].dt.year.to_numpy(dtype=float)
        y = grupo["hh_apropriado"].to_numpy(dtype=float)
        if len(grupo) >= 3 and anos.std() > 0:
            inclinacao, intercepto = np.polyfit(anos, y, 1)
        else:
            inclinacao, intercepto = None, None
        linhas.append(
            {
                "tag_code": tag_code,
                "n_eventos": len(grupo),
                "inclinacao_h_por_ano": inclinacao,
                "intercepto": intercepto,
            }
        )
    return pd.DataFrame(linhas).sort_values("tag_code").reset_index(drop=True)


def sazonalidade_mensal(eventos: pd.DataFrame) -> pd.DataFrame:
    """Contagem de falhas agregada por mês do calendário, somando todos
    os anos e todos os equipamentos do escopo (briefing complementar,
    item 4)."""
    falhas = eventos[eventos["is_falha"]]
    contagem = falhas["inicio"].dt.month.value_counts()
    tabela = pd.DataFrame({"mes": range(1, 13), "mes_nome": MESES_PT})
    tabela["falhas"] = tabela["mes"].map(contagem).fillna(0).astype(int)
    return tabela


def comparacao_consolidada(eventos: pd.DataFrame) -> pd.DataFrame:
    """Tabela final por equipamento, lado a lado (briefing complementar,
    item 5):

    - `pct_eventos_corretivos` = eventos de falha (MC/CP) / total de
      eventos do equipamento — uma PROPORÇÃO de eventos corretivos, não
      uma taxa por unidade de tempo (ex.: 218/398 = 54,8%).
    - `mtbf_dias` = intervalo médio de CALENDÁRIO, em dias, entre as
      datas ("Início") de falhas consecutivas do mesmo equipamento — não
      é tempo de operação.
    - `tempo_medio_reparo_h` = média de HH Apropriado (horas apropriadas
      por apontamento, já perturbadas), de todos os tipos de evento.
    """
    mtbf = mtbf_por_equipamento(eventos)
    linhas = []
    for tag_code, grupo in eventos.groupby("tag_code"):
        linhas.append(
            {
                "tag_code": tag_code,
                "n_eventos": len(grupo),
                "n_falhas": int(grupo["is_falha"].sum()),
                "pct_eventos_corretivos": round(grupo["is_falha"].mean() * 100, 1),
                "tempo_medio_reparo_h": round(grupo["hh_apropriado"].mean(), 2),
            }
        )
    tabela = pd.DataFrame(linhas)
    tabela = tabela.merge(mtbf[["tag_code", "mtbf_dias"]], on="tag_code", how="left")
    tabela["mtbf_dias"] = tabela["mtbf_dias"].round(1)
    return tabela.sort_values("tag_code").reset_index(drop=True)
