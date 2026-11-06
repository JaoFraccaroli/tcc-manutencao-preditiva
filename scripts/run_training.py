"""
Executa o módulo de modelagem para os dois horizontes (falha_30d e
falha_60d): comparação de modelos, tuning, otimização de threshold,
avaliação no teste e geração dos gráficos.

Uso:
    python scripts/run_training.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.modeling.train import executar_tudo

if __name__ == "__main__":
    executar_tudo()
