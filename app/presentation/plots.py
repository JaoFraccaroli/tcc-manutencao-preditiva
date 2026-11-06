"""
Geração de gráficos Matplotlib embutidos na interface (briefing seção 11:
Matplotlib é a ferramenta de gráficos da aplicação). Cada função devolve
uma string base64 (PNG) pronta para um <img src="data:image/png;base64,...">.

São todos gráficos DESCRITIVOS (distribuições, contagens) — nenhum aqui
ordena ou destaca equipamentos como prioritários (briefing seção 10).
"""
import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def histograma_probabilidades(snapshot: pd.DataFrame, horizonte: int) -> str:
    """Eixo em percentual (0-100%) — o valor interno usado pelo modelo e
    pelo limiar continua em decimal (0-1), isto é só formatação de
    exibição (briefing complementar 3, item 1)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    valores_pct = snapshot[f"prob_falha_{horizonte}d"] * 100
    ax.hist(valores_pct, bins=20, range=(0, 100))
    ax.set_xlabel(f"Probabilidade estimada de falha em {horizonte} dias (%)")
    ax.set_ylabel("Número de equipamentos")
    ax.set_title(f"Distribuição das probabilidades — falha_{horizonte}d")
    ax.xaxis.set_major_formatter(lambda x, _pos: f"{x:.0f}%")
    fig.tight_layout()
    return _fig_to_base64(fig)


def contagem_por_tipo(eventos: pd.DataFrame) -> str:
    """Rótulos por extenso (não a sigla) — briefing complementar 3, item 2."""
    from app.processing.tipos_manutencao import descricao_tipo

    contagem = eventos["tipo_manutencao"].value_counts().sort_values(ascending=True)
    rotulos = [descricao_tipo(t) for t in contagem.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(rotulos, contagem.values)
    ax.set_xlabel("Número de eventos")
    ax.set_title("Eventos de manutenção por tipo")
    fig.tight_layout()
    return _fig_to_base64(fig)


def categorias_falha(tabela_categorias: pd.DataFrame) -> str:
    """Barras com as categorias de falha descobertas na análise de texto
    (briefing complementar, item 1) — "outro / não identificado" sempre
    por último, para não confundir com uma categoria de conteúdo real."""
    from app.processing.text_analysis import NAO_IDENTIFICADO

    tabela = tabela_categorias.copy()
    tabela["ordem"] = (tabela["categoria"] == NAO_IDENTIFICADO).astype(int)
    tabela = tabela.sort_values(["ordem", "percentual"], ascending=[True, True])

    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(tabela))))
    cores = ["tab:gray" if c == NAO_IDENTIFICADO else "tab:blue" for c in tabela["categoria"]]
    ax.barh(tabela["categoria"], tabela["percentual"], color=cores)
    ax.set_xlabel("% das descrições de falha")
    ax.set_title("Categorias de falha descobertas na descrição da OS")
    fig.tight_layout()
    return _fig_to_base64(fig)


def frequencia_texto(tabela: pd.DataFrame, coluna_rotulo: str, titulo: str) -> str:
    """Barras genéricas de frequência (usada para palavras e bigramas)."""
    tabela = tabela.sort_values("ocorrencias", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(tabela))))
    ax.barh(tabela[coluna_rotulo], tabela["ocorrencias"])
    ax.set_xlabel("Nº de descrições de falha em que aparece")
    ax.set_title(titulo)
    fig.tight_layout()
    return _fig_to_base64(fig)


def mtbf_equipamentos(tabela_mtbf: pd.DataFrame) -> str:
    """MTBF por equipamento — barras cinza para quem tem menos de 2
    falhas (MTBF indefinido, mostrado como 0 só para deixar a barra
    visível, com nota separada no texto)."""
    tabela = tabela_mtbf.sort_values("tag_code")
    valores = tabela["mtbf_dias"].fillna(0)
    cores = ["tab:gray" if pd.isna(v) else "tab:blue" for v in tabela["mtbf_dias"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(tabela["tag_code"], valores, color=cores)
    ax.set_ylabel("MTBF (dias de calendário)")
    ax.set_title("Intervalo médio de calendário entre falhas (MTBF) por equipamento")
    fig.tight_layout()
    return _fig_to_base64(fig)


def tendencia_reparo_grid(eventos: pd.DataFrame, tabela_tendencia: pd.DataFrame) -> str:
    """Um painel por equipamento: HH Apropriado ao longo do tempo, com
    reta de tendência (regressão linear ano x HH Apropriado)."""
    tags = sorted(eventos["tag_code"].unique())
    n = len(tags)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)

    for i, tag_code in enumerate(tags):
        ax = axes[i // cols][i % cols]
        grupo = eventos[eventos["tag_code"] == tag_code].sort_values("inicio")
        ax.scatter(grupo["inicio"], grupo["hh_apropriado"], s=10, alpha=0.5)

        linha = tabela_tendencia[tabela_tendencia["tag_code"] == tag_code].iloc[0]
        if pd.notna(linha["inclinacao_h_por_ano"]):
            anos = grupo["inicio"].dt.year
            x_anos = pd.Series(sorted(anos.unique()))
            y_pred = linha["inclinacao_h_por_ano"] * x_anos + linha["intercepto"]
            x_datas = pd.to_datetime(x_anos.astype(str) + "-07-01")
            ax.plot(x_datas, y_pred, color="tab:red", linewidth=2)

        ax.set_title(tag_code, fontsize=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Tempo de reparo (HH Apropriado) ao longo do tempo, por equipamento")
    fig.tight_layout()
    return _fig_to_base64(fig)


def sazonalidade(tabela_sazonalidade: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(tabela_sazonalidade["mes_nome"], tabela_sazonalidade["falhas"])
    ax.set_ylabel("Nº de falhas (todos os equipamentos, todos os anos)")
    ax.set_title("Sazonalidade de falhas por mês do calendário")
    fig.tight_layout()
    return _fig_to_base64(fig)


def componentes_falha(tabela_componentes: pd.DataFrame) -> str:
    """Barras com a distribuição de componente estimado entre os eventos
    de falha (briefing complementar 2, seção 3) — "não identificado"
    sempre por último, mesma convenção de `categorias_falha`."""
    from app.processing.component_mapping import NAO_IDENTIFICADO

    tabela = tabela_componentes.copy()
    tabela["ordem"] = (tabela["componente"] == NAO_IDENTIFICADO).astype(int)
    tabela = tabela.sort_values(["ordem", "percentual"], ascending=[True, True])

    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(tabela))))
    cores = ["tab:gray" if c == NAO_IDENTIFICADO else "tab:orange" for c in tabela["componente"]]
    ax.barh(tabela["componente"], tabela["percentual"], color=cores)
    ax.set_xlabel("% dos eventos de falha")
    ax.set_title("Componente estimado por palavra-chave (eventos de falha)")
    fig.tight_layout()
    return _fig_to_base64(fig)


def historico_equipamento(eventos_tag: pd.DataFrame) -> str:
    eventos_tag = eventos_tag.sort_values("inicio")
    cores = eventos_tag["is_falha"].map({True: "tab:red", False: "tab:blue"})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(eventos_tag["inicio"], eventos_tag["hh_apropriado"], c=cores)
    ax.set_xlabel("Data")
    ax.set_ylabel("HH Apropriado (horas)")
    ax.set_title("Histórico de manutenção do equipamento (vermelho = falha)")
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_base64(fig)
