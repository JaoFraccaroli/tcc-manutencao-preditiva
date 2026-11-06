"""
Teste de mutação (mesma ideia da aula 06 de Qualidade de Software).

Cada "mutante" é uma pequena alteração proposital no código do TCC,
simulando um erro que um programador poderia cometer (trocar >= por >,
mudar uma janela de data, esquecer uma regra...). Para cada mutante:

1. o arquivo é alterado;
2. a bateria de testes roda;
3. o arquivo volta ao original (sempre, mesmo se der erro).

Se algum teste falhar, o mutante foi "morto": os testes perceberam o
erro. Se todos passarem, o mutante "sobreviveu": existe um erro que os
testes não pegariam.

Mutation score = mutantes mortos / total de mutantes.

Uso:  python qualidade/mutacao.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
MINIMO_ACEITO = 90.0  # mutation score mínimo (%)


@dataclass
class Mutante:
    nome: str
    arquivo: str
    original: str
    mutado: str


MUTANTES = [
    # --- Rótulo e janelas (app/processing/dataset.py) ---
    Mutante("Falha no dia da referência passa a contar como futura", "app/processing/dataset.py",
            'lo = np.searchsorted(grupo_falhas, refs_np, side="right")  # estritamente após a ref',
            'lo = np.searchsorted(grupo_falhas, refs_np, side="left")  # estritamente após a ref'),
    Mutante("Último dia da janela deixa de contar", "app/processing/dataset.py",
            'hi = np.searchsorted(grupo_falhas, lim_sup, side="right")',
            'hi = np.searchsorted(grupo_falhas, lim_sup, side="left")'),
    Mutante("Janela de rótulo com um dia a mais", "app/processing/dataset.py",
            'lim_sup = refs_np + np.timedelta64(horizonte, "D")',
            'lim_sup = refs_np + np.timedelta64(horizonte + 1, "D")'),
    Mutante("Sem descarte das janelas incompletas (censura à direita)", "app/processing/dataset.py",
            "fim = data_max_base - timedelta(days=horizonte)",
            "fim = data_max_base"),
    Mutante("Janela de frequência de falhas inclui a fronteira", "app/processing/dataset.py",
            'lo = np.searchsorted(falha_datas, lim_inf, side="right")',
            'lo = np.searchsorted(falha_datas, lim_inf, side="left")'),
    Mutante("Tempo médio de reparo ignora o evento mais recente", "app/processing/dataset.py",
            "[hh[:i].mean() if i > 0 else np.nan for i in idx_hist]",
            "[hh[:i - 1].mean() if i > 1 else np.nan for i in idx_hist]"),
    Mutante("Área vem do evento errado", "app/processing/dataset.py",
            "[area[i - 1] if i > 0 else None for i in idx_hist], dtype=object",
            "[area[0] if i > 0 else None for i in idx_hist], dtype=object"),

    # --- Anonimização e limpeza (app/data/ingest.py) ---
    Mutante("Código do equipamento começa em EQ000", "app/data/ingest.py",
            'f"EQ{i+1:03d}"', 'f"EQ{i:03d}"'),
    Mutante("Perturbação do HH fora de 90%-110%", "app/data/ingest.py",
            "rng.uniform(config.PERTURBACAO_MIN, config.PERTURBACAO_MAX, size=len(df))",
            "rng.uniform(0.5, 1.5, size=len(df))"),
    Mutante("Esquece de remover o nome entre parênteses", "app/data/ingest.py",
            'texto = re.sub(r"\\s*\\([^)]*\\)\\s*$", "", texto).strip()',
            "texto = texto.strip()"),
    Mutante("TAG com espaço ('MC 01') deixa de ser redigido", "app/data/ingest.py",
            'return re.escape(prefixo) + r"[\\s-]*" + re.escape(sufixo)',
            "return re.escape(prefixo) + re.escape(sufixo)"),
    Mutante("Redação do TAG passa a diferenciar maiúsculas", "app/data/ingest.py",
            'r"\\b(" + "|".join(_padrao_tag_flexivel(t) for t in tags_reais) + r")\\b", flags=re.IGNORECASE',
            'r"\\b(" + "|".join(_padrao_tag_flexivel(t) for t in tags_reais) + r")\\b"'),
    Mutante("Ordens não concluídas passam a entrar", "app/data/ingest.py",
            'df = df[df["Situação"] == "C"]', "df = df"),
    Mutante("CP deixa de ser falha", "app/data/ingest.py",
            'df["is_falha"] = tipo.isin(config.TIPOS_FALHA)', 'df["is_falha"] = tipo == "MC"'),
    Mutante("Duplicatas não são removidas", "app/data/ingest.py",
            'df = df.drop_duplicates(subset=["Nº OS", "NºTarefa", "Início"], keep="first")', "df = df"),

    # --- Componentes (app/processing/component_mapping.py) ---
    Mutante("'porta' passa a casar dentro de 'exportação'", "app/processing/component_mapping.py",
            'inicio = r"(?<![a-z0-9])" + re.escape(termo)', "inicio = re.escape(termo)"),
    Mutante("'travado' passa a contar como trava", "app/processing/component_mapping.py",
            'return re.compile(inicio + r"s?(?![a-z0-9])")', "return re.compile(inicio)"),

    # --- Data de alerta, divisão e threshold ---
    Mutante("Limiar exato deixa de gerar alerta", "app/modeling/predict.py",
            "prob >= limiar", "prob > limiar"),
    Mutante("Alerta escolhe o maior horizonte em vez do primeiro", "app/modeling/predict.py",
            "for h in sorted(config.HORIZONTES_DIAS):", "for h in sorted(config.HORIZONTES_DIAS, reverse=True):"),
    Mutante("Teste fica com os equipamentos mais antigos", "app/modeling/split.py",
            "ordem = mediana_por_tag.sort_values(ascending=False)", "ordem = mediana_por_tag.sort_values(ascending=True)"),
    Mutante("Threshold escolhe a menor precisão", "app/modeling/train.py",
            "melhor_idx = candidatos[np.argmax(precision_t[candidatos])]",
            "melhor_idx = candidatos[np.argmin(precision_t[candidatos])]"),

    # --- Análise exploratória, cadastro e interface ---
    Mutante("MTBF passa a contar as preventivas", "app/processing/exploratory.py",
            'for tag_code, grupo in eventos[eventos["is_falha"]].groupby("tag_code"):',
            'for tag_code, grupo in eventos.groupby("tag_code"):'),
    Mutante("Sazonalidade conta preventivas como falha", "app/processing/exploratory.py",
            'falhas = eventos[eventos["is_falha"]]\n    contagem', 'falhas = eventos\n    contagem'),
    Mutante("Ano 1900 deixa de ser aceito", "app/data/cadastro.py",
            "if not (1900 <= ano_int <= 2100):", "if not (1900 < ano_int <= 2100):"),
    Mutante("Capacidade com vírgula deixa de ser aceita", "app/data/cadastro.py",
            'float(capacidade.replace(",", "."))', "float(capacidade)"),
    Mutante("Data no formato americano", "app/presentation/__init__.py",
            'return valor.strftime("%d/%m/%Y")', 'return valor.strftime("%m/%d/%Y")'),
]


def rodar_testes() -> bool:
    comando = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--no-header", "-m", "not lento"]
    processo = subprocess.run(comando, cwd=RAIZ, text=True, capture_output=True, env=os.environ.copy())
    return processo.returncode == 0


def main() -> int:
    print("=== TESTE DE MUTAÇÃO ===\n")
    if not rodar_testes():
        print("Os testes precisam passar no código original antes da mutação.")
        return 1

    resultados = []
    for mutante in MUTANTES:
        caminho = RAIZ / mutante.arquivo
        # Lê e grava em bytes para preservar o fim de linha original (CRLF no Windows)
        bytes_originais = caminho.read_bytes()
        original = bytes_originais.decode("utf-8").replace("\r\n", "\n")
        if original.count(mutante.original) != 1:
            print(f"[ERRO] trecho do mutante '{mutante.nome}' não encontrado (ou repetido) em {mutante.arquivo}")
            return 1
        inicio = time.time()
        try:
            caminho.write_bytes(original.replace(mutante.original, mutante.mutado).encode("utf-8"))
            morto = not rodar_testes()
        finally:
            caminho.write_bytes(bytes_originais)
        resultados.append((mutante, morto))
        situacao = "MORTO     " if morto else "SOBREVIVEU"
        print(f"[{situacao}] {mutante.nome} ({mutante.arquivo}, {time.time() - inicio:.1f}s)")

    mortos = sum(1 for _, morto in resultados if morto)
    score = 100 * mortos / len(resultados)
    print(f"\nMutation score: {mortos}/{len(resultados)} = {score:.1f}% (mínimo: {MINIMO_ACEITO:.0f}%)")

    relatorio = RAIZ / "relatorios" / "mutacao.md"
    relatorio.parent.mkdir(exist_ok=True)
    linhas = ["| Mutante | Arquivo | Resultado |", "|---|---|---|"]
    linhas += [f"| {m.nome} | `{m.arquivo}` | {'✅ morto' if morto else '❌ sobreviveu'} |" for m, morto in resultados]
    relatorio.write_text(
        f"### Teste de mutação\n\n**Mutation score: {mortos}/{len(resultados)} = {score:.1f}%**\n\n"
        + "\n".join(linhas) + "\n", encoding="utf-8")

    return 0 if score >= MINIMO_ACEITO else 1


if __name__ == "__main__":
    sys.exit(main())
