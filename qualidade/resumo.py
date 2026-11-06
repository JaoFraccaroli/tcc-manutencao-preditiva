"""
Resumo de qualidade em Markdown (usado no resumo do GitHub Actions):
quantidade de testes, cobertura de código e teste de mutação.

Uso:  python qualidade/resumo.py
"""
import xml.etree.ElementTree as ET
from pathlib import Path

RELATORIOS = Path(__file__).resolve().parent.parent / "relatorios"


def testes():
    arquivo = RELATORIOS / "junit.xml"
    if not arquivo.exists():
        return "| Testes | sem relatório |"
    raiz = ET.parse(arquivo).getroot()
    suites = raiz.findall("testsuite") if raiz.tag == "testsuites" else [raiz]
    total = sum(int(s.get("tests", 0)) for s in suites)
    falhas = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    pulados = sum(int(s.get("skipped", 0)) for s in suites)
    passaram = total - falhas - pulados
    return (f"| Testes | {passaram} passando, {falhas} falhando, "
            f"{pulados} defeitos/limitações documentados (xfail) |")


def cobertura():
    arquivo = RELATORIOS / "cobertura.xml"
    if not arquivo.exists():
        return "| Cobertura de linhas | sem relatório |"
    taxa = float(ET.parse(arquivo).getroot().get("line-rate", 0)) * 100
    return f"| Cobertura de linhas | {taxa:.1f}% (mínimo exigido: 90%) |"


def main():
    print("## Qualidade do código do TCC\n")
    print("| Métrica | Resultado |\n|---|---|")
    print(testes())
    print(cobertura())
    mutacao = RELATORIOS / "mutacao.md"
    if mutacao.exists():
        print()
        print(mutacao.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
