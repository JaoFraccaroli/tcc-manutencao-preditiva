"""
Divisão treino/teste combinando as duas exigências do briefing (seção 7):

1. Sem vazamento entre grupos: nenhum TAG aparece simultaneamente em
   treino e teste — reaproveita a ideia do GroupShuffleSplit do Ex1.py
   (agrupar por TAG), só que em vez de sortear os grupos aleatoriamente,
   ordena-os no tempo.
2. Escolha dos TAGs de teste por ordem temporal agregada: entram no teste
   os TAGs cuja data de referência mediana é mais recente. NÃO existe corte
   por data: cada TAG tem atividade espalhada por todo o histórico, e o
   teste são equipamentos inteiros fora do treino.

Como cada equipamento tem eventos espalhados por todo o histórico (não dá
para "cortar" um TAG no meio sem violar o requisito 1), a divisão é feita
no nível do GRUPO, não da linha:

  a) Calcula-se, para cada TAG, a data de referência mediana das suas
     linhas — um resumo de "quando" aquele equipamento concentrou
     atividade.
  b) Os TAGs são ordenados por essa mediana.
  c) Os TAGs com atividade mais recente entram no teste, acumulando
     linhas até atingir ~test_size do total; o restante (TAGs com
     atividade mais antiga) forma o treino.

Isso satisfaz o requisito 1 de forma exata (grupo inteiro de um lado só)
e o requisito 2 de forma agregada (o conjunto de teste é, em conjunto,
mais recente que o de treino, pela mediana) — sem descartar nenhuma linha.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class DivisaoTreinoTeste:
    treino: pd.DataFrame
    teste: pd.DataFrame
    data_mediana_corte: pd.Timestamp


def dividir_treino_teste(dataset: pd.DataFrame, test_size: float = 0.2) -> DivisaoTreinoTeste:
    mediana_por_tag = dataset.groupby("tag_code")["data_referencia"].median()
    contagem_por_tag = dataset.groupby("tag_code").size()

    ordem = mediana_por_tag.sort_values(ascending=False)  # mais recente primeiro
    total_linhas = contagem_por_tag.sum()
    alvo_teste = total_linhas * test_size

    acumulado = 0
    tags_teste = []
    for tag_code in ordem.index:
        if acumulado >= alvo_teste:
            break
        tags_teste.append(tag_code)
        acumulado += contagem_por_tag[tag_code]

    tags_teste = set(tags_teste)
    mask_teste = dataset["tag_code"].isin(tags_teste)

    treino = dataset[~mask_teste].reset_index(drop=True)
    teste = dataset[mask_teste].reset_index(drop=True)

    overlap = set(treino["tag_code"]).intersection(teste["tag_code"])
    assert not overlap, f"Vazamento de grupo detectado para TAGs: {overlap}"

    data_mediana_corte = ordem.loc[list(tags_teste)].min() if tags_teste else ordem.min()

    return DivisaoTreinoTeste(treino=treino, teste=teste, data_mediana_corte=data_mediana_corte)
