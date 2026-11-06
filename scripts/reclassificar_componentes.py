"""
Reaplica `component_mapping.classificar_componente` aos eventos de falha já
gravados no banco, SEM refazer a ingestão (que apagaria o cadastro manual
das máquinas). A regra roda sobre `descricao_tarefa`, o mesmo texto redigido
que a ingestão usa, então o resultado é idêntico ao de uma reingestão.

Uso:
    python scripts/reclassificar_componentes.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

from app.data.db import get_session
from app.data.models import EventoManutencao
from app.processing import component_mapping


def main() -> None:
    session = get_session()
    try:
        falhas = session.query(EventoManutencao).filter(EventoManutencao.is_falha.is_(True)).all()
        antes = Counter(e.componente_estimado for e in falhas)
        mudou = 0
        for e in falhas:
            novo = component_mapping.classificar_componente(e.descricao_tarefa)
            if novo != e.componente_estimado:
                e.componente_estimado = novo
                mudou += 1
        session.commit()
        depois = Counter(e.componente_estimado for e in falhas)
    finally:
        session.close()

    print(f"Eventos de falha: {len(falhas)} | reclassificados: {mudou}")
    for comp in sorted(set(antes) | set(depois), key=lambda c: -depois[c]):
        print(f"  {comp:28s} {antes[comp]:5d} -> {depois[comp]:5d}")


if __name__ == "__main__":
    main()
