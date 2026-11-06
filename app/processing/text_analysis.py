"""
Análise exploratória de texto — briefing complementar de análise
exploratória (item 1), módulo independente do pipeline de modelagem
preditiva (`app/modeling/`). Usa só a coluna de texto livre
`descricao_tarefa` dos eventos já filtrados para o escopo de
monta-cargas; não participa de nenhuma feature do modelo de
classificação.

O objetivo é deixar os PRÓPRIOS DADOS revelarem os agrupamentos de
padrões de falha, em vez de impor categorias definidas a priori: as
"categorias" desta análise são, literalmente, os bigramas (ou palavras
isoladas, quando a descrição é curta demais para gerar bigrama) mais
frequentes entre as descrições de falha, usados como "sementes" às
quais cada evento é associado. O que não casa com nenhuma semente
suficientemente frequente vai para "outro / não identificado" — não é
forçado em nenhuma categoria.

Sobre anonimização: o texto usado aqui já vem com qualquer menção ao TAG
real substituída por um marcador neutro na ingestão (`app/data/ingest.py`,
`_redigir_descricao`) — este módulo nunca vê o TAG original.
"""
import re
import unicodedata
from collections import Counter

import pandas as pd

# Palavras funcionais removidas da contagem — deliberadamente NÃO inclui
# negações/advérbios que carregam sentido de falha (ex.: "não", "sem"),
# porque bigramas como "não sobe" / "sem funcionar" são exatamente o tipo
# de padrão que esta análise busca revelar (briefing complementar, item 1).
STOPWORDS = {
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "e", "ou", "ao", "aos", "por", "com",
    "que", "se", "foi", "ser", "esta", "estao",
}

# Termos contextuais específicos deste dataset: "monta carga" é o nome do
# próprio tipo de equipamento (aparece em ~60% das descrições só por
# identificar do que se trata, não por descrever um sintoma de falha), e
# "linha"/"nacional"/"exportação"/"térreo"/"sala"/"micro" são referências
# de localização (qual linha de produção, qual andar/sala), equivalentes
# em função a um campo estruturado como Área — não a um sintoma. Excluídos
# da mesma forma que o código do equipamento (briefing complementar, item
# 1: "para não poluir a contagem"), decisão tomada após inspecionar a
# saída real (sem eles, "monta"/"carga"/"linha" dominavam as categorias
# como se fossem os padrões mais informativos, quando na verdade só
# identificam o equipamento/local, presentes em quase toda descrição).
STOPWORDS |= {
    "monta", "montas", "carga", "cargas",
    "linha", "linhas", "nacional", "exportacao", "terreo", "sala", "micro",
}

NAO_IDENTIFICADO = "outro / não identificado"

# Nº de categorias (sementes de bigrama/palavra) e frequência mínima
# (fração das descrições de falha) para uma semente virar categoria.
# N_CATEGORIAS_PADRAO=15 foi escolhido depois de testar valores entre 8 e
# 25 nos dados reais: com poucas sementes, quase tudo cai em "não
# identificado" (73% com 8 sementes, já que só ~9 bigramas alcançam o
# limiar de frequência mínima); a partir de ~15 o ganho marginal de
# reduzir ainda mais o "não identificado" fica pequeno, então o valor
# escolhido equilibra cobertura com sementes ainda razoavelmente
# específicas (bigramas como "parou funcionar", "porta não", "trava
# porta" antes de recorrer a palavras isoladas mais genéricas).
N_CATEGORIAS_PADRAO = 15
FRACAO_MINIMA_PADRAO = 0.02


def _normalizar(texto: str) -> str:
    """minúsculas, remove acentuação e pontuação, colapsa espaços."""
    texto = texto.lower()
    texto = re.sub(r"\[equip\]", " ", texto)  # marcador de TAG redigido (ingest.py)
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def _tokenizar(texto_normalizado: str) -> list[str]:
    tokens = texto_normalizado.split(" ") if texto_normalizado else []
    # Tokens puramente numéricos (nº de linha, andar, etc.) identificam
    # localização, não sintoma de falha — mesma lógica da exclusão acima.
    return [t for t in tokens if t and t not in STOPWORDS and len(t) > 1 and not t.isdigit()]


def _bigramas(tokens: list[str]) -> list[str]:
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def carregar_descricoes_falha(eventos: pd.DataFrame) -> pd.Series:
    """Descrições das intervenções classificadas como falha (MC/CP) — as
    demais (preventiva, não classificado) descrevem tarefas planejadas,
    não sintomas de falha, e ficam fora desta análise especificamente."""
    falhas = eventos[eventos["is_falha"] & eventos["descricao_tarefa"].notna()]
    return falhas["descricao_tarefa"]


def frequencia_palavras(descricoes: pd.Series, top_n: int = 25) -> pd.DataFrame:
    """Frequência de palavras isoladas (nº de descrições em que aparecem)."""
    contador = Counter()
    for texto in descricoes:
        contador.update(set(_tokenizar(_normalizar(texto))))
    return pd.DataFrame(contador.most_common(top_n), columns=["palavra", "ocorrencias"])


def frequencia_bigramas(descricoes: pd.Series, top_n: int = 25) -> pd.DataFrame:
    """Frequência de bigramas (nº de descrições em que aparecem)."""
    contador = Counter()
    for texto in descricoes:
        tokens = _tokenizar(_normalizar(texto))
        contador.update(set(_bigramas(tokens)))
    return pd.DataFrame(contador.most_common(top_n), columns=["bigrama", "ocorrencias"])


def categorizar(
    eventos: pd.DataFrame,
    n_categorias: int = N_CATEGORIAS_PADRAO,
    fracao_minima: float = FRACAO_MINIMA_PADRAO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa as descrições de falha em categorias descobertas a partir
    dos bigramas/palavras mais frequentes (nenhuma categoria definida a
    priori — briefing complementar, item 1).

    Cada descrição é atribuída à semente (bigrama ou, na falta de um bom
    bigrama, palavra isolada) de maior frequência que aparece em seu
    texto; quem não contém nenhuma semente suficientemente frequente vai
    para "outro / não identificado".

    Devolve (tabela de categorias com percentual, tabela das sementes
    usadas para formar as categorias).
    """
    descricoes = carregar_descricoes_falha(eventos)
    total = len(descricoes)
    if total == 0:
        vazio = pd.DataFrame(columns=["categoria", "ocorrencias", "percentual"])
        return vazio, pd.DataFrame(columns=["padrao", "tipo", "ocorrencias"])

    tokenizados = descricoes.map(lambda t: _tokenizar(_normalizar(t)))
    textos_filtrados = tokenizados.map(lambda toks: " " + " ".join(toks) + " ")

    contador_bigramas = Counter()
    contador_unigramas = Counter()
    for tokens in tokenizados:
        contador_bigramas.update(set(_bigramas(tokens)))
        contador_unigramas.update(set(tokens))

    # Bigramas entram com prioridade sobre palavras isoladas: um bigrama
    # ("parou de funcionar") é sempre mais específico/informativo que um
    # verbo genérico isolado ("parou"), que sozinho tende a "engolir"
    # quase tudo antes de um padrão mais específico ter chance de casar.
    # Palavras isoladas só completam a lista de sementes até
    # `n_categorias` (útil para descrições curtas demais para gerar
    # bigrama, ex.: "travado", "alarme").
    minimo = max(2, int(total * fracao_minima))
    bigramas_ok = sorted(
        ((bg, c, "bigrama") for bg, c in contador_bigramas.items() if c >= minimo),
        key=lambda x: -x[1],
    )
    unigramas_ok = sorted(
        ((uni, c, "palavra") for uni, c in contador_unigramas.items() if c >= minimo),
        key=lambda x: -x[1],
    )
    sementes = (bigramas_ok + unigramas_ok)[:n_categorias]

    def atribuir(texto_filtrado: str) -> str:
        for padrao, _, _ in sementes:
            if f" {padrao} " in texto_filtrado:
                return padrao
        return NAO_IDENTIFICADO

    categorias = textos_filtrados.map(atribuir)
    contagem = categorias.value_counts()

    tabela = contagem.rename_axis("categoria").reset_index(name="ocorrencias")
    tabela["percentual"] = (tabela["ocorrencias"] / total * 100).round(1)
    tabela = tabela.sort_values("ocorrencias", ascending=False).reset_index(drop=True)

    sementes_df = pd.DataFrame(sementes, columns=["padrao", "ocorrencias_descricoes", "tipo"])

    return tabela, sementes_df
