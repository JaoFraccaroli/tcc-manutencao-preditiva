"""
Executa o módulo de dados: lê a planilha bruta, limpa, anonimiza e carrega
no banco (SQLite por padrão; ver config.py / DATABASE_URL).

Uso:
    python scripts/run_ingest.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.ingest import executar

if __name__ == "__main__":
    executar()
