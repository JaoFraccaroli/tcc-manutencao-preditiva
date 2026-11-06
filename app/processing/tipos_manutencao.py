"""
Tradução de código -> descrição por extenso do campo `Tipo de Manutenção`
(briefing complementar 3, item 2). Só formatação/rotulagem para
exibição — não muda a classificação de falha (`config.TIPOS_FALHA`/
`TIPO_PREVENTIVA`), que continua baseada nos códigos originais.

Só MC, CP e MP têm significado confirmado no TCC (briefing seção 3.2 /
tópico 3.2 do TCC: são exatamente os códigos usados para definir falha e
preventiva). Os demais códigos (IN, ME, EV, IP, CA, LB, PT, ...) são
tratados no TCC como "sem significado claramente definido no contexto
deste trabalho" — por isso usam um rótulo genérico, em vez de arriscar
uma tradução não confirmada; o código original fica entre parênteses
para continuar distinguível em tabelas e gráficos.
"""

TIPO_MANUTENCAO_DESCRICAO = {
    "MC": "Corretiva",
    "CP": "Corretiva Programada",
    "MP": "Preventiva",
}


def descricao_tipo(codigo: str) -> str:
    if codigo in TIPO_MANUTENCAO_DESCRICAO:
        return TIPO_MANUTENCAO_DESCRICAO[codigo]
    return f"Outro (não classificado) — {codigo}"
