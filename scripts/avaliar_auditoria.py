"""
Avalia a auditoria manual de rótulos de componente (evidências, item 23).

Entrada: `evidencias/csv/auditoria_rotulos_amostra.csv`, com a coluna
`rotulo_manual_componente` preenchida por uma pessoa (nome do componente
do catálogo, exatamente como em `componente_estimado`, ou `nenhum` quando a
descrição não permite apontar componente).

Saída (stdout): precisão dos eventos que a regra identificou, fração dos
"não identificado" que na verdade tinham componente e matriz de confusão.

Uso:
    python scripts/avaliar_auditoria.py [caminho_do_csv]
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAO_IDENTIFICADO = "não identificado"


def main() -> None:
    caminho = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "evidencias", "csv", "auditoria_rotulos_amostra.csv")
    df = pd.read_csv(caminho, encoding="utf-8-sig").fillna("")
    df["rotulo_manual_componente"] = df["rotulo_manual_componente"].astype(str).str.strip()
    pendentes = int((df["rotulo_manual_componente"] == "").sum())
    if pendentes:
        raise SystemExit(f"{pendentes} de {len(df)} linhas ainda sem rótulo manual — preencha a coluna antes de avaliar.")

    manual = df["rotulo_manual_componente"].str.lower().replace({"nenhum": NAO_IDENTIFICADO})
    df["manual"] = manual.where(manual == NAO_IDENTIFICADO, df["rotulo_manual_componente"])

    ident = df[df["grupo_amostra"] == "identificado"]
    nao = df[df["grupo_amostra"] == "não identificado"]

    acertos = (ident["manual"] == ident["componente_estimado"]).sum()
    print(f"Identificados pela regra (n={len(ident)}): rótulo da regra confere com o manual em {acertos} ({100 * acertos / len(ident):.1f}%).")
    com_comp = (nao["manual"] != NAO_IDENTIFICADO).sum()
    print(f"'Não identificado' pela regra (n={len(nao)}): {com_comp} ({100 * com_comp / len(nao):.1f}%) tinham componente apontável no julgamento manual.")
    print("\nMatriz de confusão (linhas = regra, colunas = manual):")
    print(pd.crosstab(df["componente_estimado"], df["manual"]).to_string())
    print("\nAmostras pequenas (50 por grupo): intervalos de confiança largos; reporte n junto com os percentuais.")


if __name__ == "__main__":
    main()
