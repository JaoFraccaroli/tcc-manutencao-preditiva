"""
Classificação estimada de componente por evento de falha (briefing
complementar 2, seção 3).

**Isto é uma estimativa por palavra-chave, não um campo medido pelo
sistema de origem.** O CMMS da empresa não registra qual componente
falhou em cada evento — só existe a descrição textual livre
(`Descrição Tarefa`). A correspondência abaixo foi validada contra a
base real (2.929 eventos, 1.990 classificados como falha) comparando os
percentuais resultantes com uma inspeção manual de amostra — não é uma
classificação de IA sem verificação (isso já foi tentado e descartado:
os números não batiam com a base real).

Regras (documentadas aqui para quem for ler o resultado):
- Restrito a eventos `is_falha=True` (MC/CP). Eventos de preventiva ou
  não classificados não recebem `componente_estimado`.
- Texto normalizado (minúsculas, sem acento) antes da busca. O termo deve
  começar uma palavra: "porta" não casa com "exportação" nem "comporta"
  (correção do falso positivo descoberto na auditoria de evidências).
  Além disso, "trava" só casa como palavra inteira ("trava"/"travas"):
  "travado", "travando" etc. não contam como Portas / travas.
- Um evento pode bater com mais de um termo (ex.: menciona porta E
  sensor). Quando isso acontece, o componente atribuído é o PRIMEIRO que
  bate, na ordem de `TERMOS_COMPONENTE` abaixo — critério de desempate
  simples, para manter uma categoria única por evento (documentado no
  README). Nada impede reprocessar guardando todas as correspondências
  se um uso futuro precisar disso.
- Sem nenhuma correspondência -> `NAO_IDENTIFICADO` (esperado ficar perto
  de metade da base — não é bug, é o texto livre não trazendo
  informação suficiente para estimar o componente).

O catálogo (`CATALOGO_COMPONENTES`) é conhecimento técnico geral sobre os
tipos de componente de um monta-carga (fabricantes de referência, vida
útil típica, critério de substituição) — não é dado específico da
empresa em estudo.
"""
import re
import unicodedata

import pandas as pd

NAO_IDENTIFICADO = "não identificado"

# Ordem importa: é o critério de desempate quando uma descrição bate com
# mais de um componente (o primeiro da lista abaixo que casar "vence").
TERMOS_COMPONENTE = {
    "Motor elétrico": ["motor"],
    "Redutor": ["redutor", "engrenagem"],
    "Rolamentos": ["rolamento"],
    "Cabos de aço": ["cabo de aco", "cabo aco", "cabos de aco"],
    "Polias / roletes": ["polia", "rolete"],
    "Freio eletromagnético": ["freio"],
    "Sensores / fim de curso": ["sensor", "fim de curso", "fim curso"],
    "Contatores / comandos": ["contator", "comando eletric"],
    "Painel elétrico": ["painel"],
    "Trilhos / guias": ["trilho", "guia"],
    "Portas / travas": ["porta", "trava"],
    "Cabos elétricos móveis": ["cabo eletric", "fiacao", "curto circuito", "curto-circuito"],
    "Tambor de enrolamento": ["tambor"],
    "Estrutura da plataforma": ["plataforma", "estrutura"],
}

# Catálogo de referência técnica (briefing complementar 2, seção 1).
# Fabricantes citados são exemplos de mercado para o TIPO de componente
# (conhecimento geral de engenharia), não fornecedores confirmados da
# empresa em estudo.
CATALOGO_COMPONENTES = {
    "Motor elétrico": {
        "fabricante_referencia": "WEG / SEW-EURODRIVE",
        "vida_util_indicada": "~20.000 a 40.000 horas de operação, conforme carga e manutenção",
        "unidade": "Horas de operação",
        "criterio_substituicao": "Falha de isolamento, superaquecimento recorrente ou desgaste de rolamento interno",
    },
    "Redutor": {
        "fabricante_referencia": "SEW-EURODRIVE / WEG",
        "vida_util_indicada": "~15.000 a 30.000 horas, conforme fator de serviço",
        "unidade": "Horas de operação",
        "criterio_substituicao": "Ruído/vibração anormal, folga excessiva ou vazamento de óleo",
    },
    "Rolamentos": {
        "fabricante_referencia": "SKF",
        "vida_util_indicada": "Centenas de milhares a milhões de ciclos, conforme carga e lubrificação",
        "unidade": "Ciclos",
        "criterio_substituicao": "Ruído, vibração ou aquecimento acima do esperado",
    },
    "Cabos de aço": {
        "fabricante_referencia": "Fabricantes especializados em cabos de elevação (ex.: Cimaf, Usiminas Cabos)",
        "vida_util_indicada": "Conforme nº de ciclos de içamento e critério de descarte por rompimento de fios (NBR)",
        "unidade": "Ciclos de içamento",
        "criterio_substituicao": "Fios rompidos acima do limite normativo, corrosão ou deformação",
    },
    "Polias / roletes": {
        "fabricante_referencia": "SKF / fabricantes de transmissão mecânica",
        "vida_util_indicada": "Vida útil associada ao rolamento interno (centenas de milhares de ciclos)",
        "unidade": "Ciclos",
        "criterio_substituicao": "Desgaste da pista de rolamento, folga ou desalinhamento",
    },
    "Freio eletromagnético": {
        "fabricante_referencia": "Mayr",
        "vida_util_indicada": "Conforme nº de frenagens e desgaste do disco de fricção",
        "unidade": "Nº de frenagens",
        "criterio_substituicao": "Desgaste do disco de fricção além do limite ou falha de acionamento",
    },
    "Sensores / fim de curso": {
        "fabricante_referencia": "Schneider / Siemens / Omron",
        "vida_util_indicada": "Centenas de milhares a milhões de ciclos, conforme modelo",
        "unidade": "Ciclos",
        "criterio_substituicao": "Falha de detecção, desgaste mecânico ou falha elétrica",
    },
    "Contatores / comandos": {
        "fabricante_referencia": "Schneider / Siemens",
        "vida_util_indicada": "Centenas de milhares de manobras, conforme categoria de uso",
        "unidade": "Manobras",
        "criterio_substituicao": "Desgaste de contatos, falha de acionamento ou aquecimento",
    },
    "Painel elétrico": {
        "fabricante_referencia": "Schneider / Siemens (componentes internos do painel)",
        "vida_util_indicada": "Vida útil associada aos componentes internos (contatores, disjuntores, etc.)",
        "unidade": "—",
        "criterio_substituicao": "Falha de componente interno, mau contato ou dano por umidade/poeira",
    },
    "Trilhos / guias": {
        "fabricante_referencia": "igus (guias e cabos móveis) / fabricantes de perfis estruturais",
        "vida_util_indicada": "Conforme desgaste mecânico acumulado, sem ciclo padronizado de fabricante",
        "unidade": "—",
        "criterio_substituicao": "Desalinhamento, desgaste visível ou ruído excessivo na movimentação",
    },
    "Portas / travas": {
        "fabricante_referencia": "Fabricantes de portas e intertravamentos de elevadores/monta-cargas",
        "vida_util_indicada": "Centenas de milhares de ciclos de abertura/fechamento",
        "unidade": "Ciclos",
        "criterio_substituicao": "Falha de abertura/fechamento, desalinhamento ou falha do intertravamento",
    },
    "Cabos elétricos móveis": {
        "fabricante_referencia": "igus (cabos e esteiras porta-cabos)",
        "vida_util_indicada": "Milhões de ciclos de flexão, conforme aplicação",
        "unidade": "Ciclos de flexão",
        "criterio_substituicao": "Ruptura de fios internos, curto-circuito ou dano visível no isolamento",
    },
    "Tambor de enrolamento": {
        "fabricante_referencia": "Fabricantes especializados em guinchos/tambores de elevação",
        "vida_util_indicada": "Conforme desgaste do sulco do tambor e nº de ciclos de içamento",
        "unidade": "Ciclos de içamento",
        "criterio_substituicao": "Desgaste do sulco, deformação ou dano estrutural",
    },
    "Estrutura da plataforma": {
        "fabricante_referencia": "Estrutura própria do equipamento (fabricação sob projeto)",
        "vida_util_indicada": "Vida útil estrutural longa; depende de inspeção visual e de manutenção corretiva pontual",
        "unidade": "—",
        "criterio_substituicao": "Deformação, trinca ou corrosão estrutural identificada em inspeção",
    },
}


# O termo precisa COMEÇAR uma palavra (não pode estar no meio dela): "porta"
# casa com "porta", "portas", mas não com "exportação" (nome de linha de
# produção, usado só como localização) nem com "comporta". Flexões que
# continuam depois do termo ("travado", "sensores") seguem casando.
#
# Exceção: "trava" só casa como palavra inteira ("trava", "travas"). Flexões
# do verbo travar ("travado", "travando", "travou") costumam descrever a
# máquina travada, não o componente trava da porta, e ficam de fora.
TERMOS_PALAVRA_INTEIRA = {"trava"}


def _compilar(termo: str) -> re.Pattern:
    inicio = r"(?<![a-z0-9])" + re.escape(termo)
    if termo in TERMOS_PALAVRA_INTEIRA:
        return re.compile(inicio + r"s?(?![a-z0-9])")
    return re.compile(inicio)


_PADROES_COMPONENTE = {
    componente: [_compilar(termo) for termo in termos]
    for componente, termos in TERMOS_COMPONENTE.items()
}


def _normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"\[equip\]", " ", texto)
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto


def classificar_componente(descricao: str) -> str:
    """Devolve o nome do componente estimado para uma descrição (já em
    texto bruto — a normalização é feita aqui), ou NAO_IDENTIFICADO."""
    if not descricao or (isinstance(descricao, float) and pd.isna(descricao)):
        return NAO_IDENTIFICADO
    texto = _normalizar(str(descricao))
    for componente, padroes in _PADROES_COMPONENTE.items():
        if any(p.search(texto) for p in padroes):
            return componente
    return NAO_IDENTIFICADO


def classificar_serie(descricoes: pd.Series) -> pd.Series:
    return descricoes.map(classificar_componente)


def tabela_resumo(eventos: pd.DataFrame) -> pd.DataFrame:
    """componente x contagem x percentual, sobre os eventos de falha
    (is_falha=True) do DataFrame de eventos já carregado."""
    falhas = eventos[eventos["is_falha"]]
    total = len(falhas)
    if total == 0:
        return pd.DataFrame(columns=["componente", "ocorrencias", "percentual"])

    contagem = falhas["componente_estimado"].fillna(NAO_IDENTIFICADO).value_counts()
    tabela = contagem.rename_axis("componente").reset_index(name="ocorrencias")
    tabela["percentual"] = (tabela["ocorrencias"] / total * 100).round(1)
    return tabela.sort_values("ocorrencias", ascending=False).reset_index(drop=True)


def exemplos_por_componente(eventos: pd.DataFrame, componente: str, limite: int = 15) -> pd.DataFrame:
    """Amostra de eventos de falha reais cujo `componente_estimado` bateu
    com `componente` — demonstra concretamente a correspondência
    palavra-chave -> componente (briefing complementar 3, item 9), em vez
    de deixar a tabela de distribuição como um número solto sem
    rastreabilidade visível.

    A descrição devolvida é a já redigida na ingestão (sem TAG real, sem
    nome de técnico/empresa) — segura para exibir.
    """
    falhas = eventos[eventos["is_falha"]].copy()
    falhas["componente_estimado"] = falhas["componente_estimado"].fillna(NAO_IDENTIFICADO)
    filtrado = falhas[falhas["componente_estimado"] == componente]
    filtrado = filtrado.sort_values("inicio", ascending=False).head(limite)
    return filtrado[["tag_code", "inicio", "descricao_tarefa"]].reset_index(drop=True)
