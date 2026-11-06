"""Lista de features usadas na modelagem (briefing seção 6).

Diferente do Ex1.py (sensores contínuos simulados: temperatura, vibração,
RPM...), aqui as features vêm inteiramente do histórico de OS.
"""

NUMERIC_FEATURES = [
    "tempo_desde_ultima_falha",
    "tempo_medio_reparo",
    "n_eventos_historico",
    "contagem_preventivas_recentes",
    "freq_falhas_90d",
    "freq_falhas_180d",
]

CATEGORICAL_FEATURES = [
    "area",
    "sist_oper",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
