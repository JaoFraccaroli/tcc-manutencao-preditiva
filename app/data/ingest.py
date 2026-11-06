"""
Módulo de dados (i) — extração da planilha bruta, limpeza mínima,
anonimização e carga no banco tratado.

Ordem das operações (cada uma documentada com a seção do briefing que a
justifica):

1. Ler a planilha bruta (Sheet1) e normalizar nomes de coluna.
2. Remover a coluna `Malha` (100% nula) e as linhas com `Situação` nula
   (3 registros) — briefing seção 2/3.2.
3. Tirar espaços em branco de `Início`/`Fim` e converter para data
   (`%d/%m/%Y`) — briefing seção 2. Fontes sem coluna `Fim` própria (ex.:
   planilha dedicada de monta-cargas, só tem `Início` + horários no mesmo
   dia) usam `Fim = Início`.
4. Manter apenas `Situação == "C"` (Concluída) — briefing seção 3.2.
4b. Restringir ao escopo definido em `config.TAG_PREFIXOS_ESCOPO` (TAG
    ORIGINAL, antes da anonimização). Escopo atual: monta-cargas
    (MC01..MC06). Lista vazia/None = sem filtro (todos os equipamentos).
5. Classificar `Tipo de Manutenção`:
   - MC/CP -> falha (is_falha=True)
   - MP -> preventiva, não é falha, mas é sinal (is_preventiva=True)
   - demais (IN, ME, IP, EV, LB, PT) -> não classificado
     (classificado=False); ficam no histórico geral mas não entram na
     construção de rótulo (feito no módulo de processamento).
6. Anonimizar TAG -> código genérico sequencial (EQ001, EQ002, ...). O
   mapeamento fica só em `tag_mapping_interno`, nunca é exposto.
6b. Redigir menções ao TAG real dentro do texto livre de `Descrição
    Tarefa` (ex.: "Lâmpada do MC01 queimada" -> "Lâmpada do [equip]
    queimada"), usada pela análise exploratória de texto (briefing
    complementar). Sem isso, o TAG real vazaria embutido no texto mesmo
    com a coluna estruturada já anonimizada.
7. Remover colunas potencialmente identificáveis que não são usadas como
   feature: `Centro de Custo`, `NºMatrícula`, `Ofic Coordenadora`,
   `Ofic. Executante` — briefing seção 4.
8. Perturbar `HH Apropriado` com fator aleatório uniforme em
   [PERTURBACAO_MIN, PERTURBACAO_MAX], seed fixa — briefing seção 4. O
   valor perturbado é o único que fica armazenado; não existe cópia do
   valor original no banco.
9. Estimar `componente_estimado` por palavra-chave em cima da descrição
   já redigida, só para eventos de falha (briefing complementar 2, seção
   3) — ESTIMATIVA, não um campo medido pelo sistema de origem. Ver
   `app/processing/component_mapping.py`.

Este módulo não decide o que é "falha nos próximos N dias" — isso é
responsabilidade do módulo de processamento (janela deslizante por TAG),
porque depende da data de referência escolhida, não de cada evento
isoladamente.
"""
import re

import numpy as np
import pandas as pd

import config
from app.data.db import get_session, init_db
from app.data.models import TagMapping, Equipamento, Componente, EventoManutencao
from app.processing import component_mapping


def _carregar_bruto(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.columns = [c.strip() for c in df.columns]
    return df


def _limpar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Coluna 100% vazia — descartar (briefing seção 2).
    if "Malha" in df.columns:
        df = df.drop(columns=["Malha"])

    # Algumas fontes não têm uma coluna "Fim" separada — só "Início" (data)
    # + "Hr. Início:"/"Hr. Fim:" (hora, mesmo dia). É o caso da planilha
    # dedicada de monta-cargas. Nesse caso Fim = Início (evento de um dia
    # só); se "Fim" existir, ele é respeitado normalmente.
    tem_coluna_fim = "Fim" in df.columns

    # Colunas de texto usadas adiante precisam de strip (datas vêm com
    # espaço em branco antes do valor em algumas fontes, ex: " 31/12/2025").
    colunas_texto = ["Início", "Situação", "Tipo de Manutenção", "TAG", "Área", "Sist Oper"]
    if tem_coluna_fim:
        colunas_texto.append("Fim")
    for col in colunas_texto:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Registros com Situação nula -> descartar (viravam string "nan" após
    # o cast acima, então tratamos isso explicitamente).
    df = df[df["Situação"].notna() & (df["Situação"].str.lower() != "nan")]

    # Manter apenas Situação == "C" (briefing seção 3.2).
    df = df[df["Situação"] == "C"]

    # Datas.
    df["Início"] = pd.to_datetime(df["Início"], format="%d/%m/%Y", errors="raise")
    if tem_coluna_fim:
        df["Fim"] = pd.to_datetime(df["Fim"], format="%d/%m/%Y", errors="raise")
    else:
        df["Fim"] = df["Início"]

    return df


def _filtrar_escopo(df: pd.DataFrame) -> pd.DataFrame:
    """Restringe a ingestão a um subconjunto de equipamentos, pelo prefixo
    do TAG ORIGINAL (`config.TAG_PREFIXOS_ESCOPO`). Escopo atual do TCC:
    monta-cargas (TAG original MC01..MC06). Lista vazia/None = sem filtro,
    mantém o comportamento genérico (todos os equipamentos).
    """
    prefixos = config.TAG_PREFIXOS_ESCOPO
    if not prefixos:
        return df

    tag_upper = df["TAG"].str.upper()
    mascara = pd.Series(False, index=df.index)
    for prefixo in prefixos:
        mascara |= tag_upper.str.startswith(prefixo.upper())

    return df[mascara].copy()


def _classificar_tipo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tipo = df["Tipo de Manutenção"]
    df["is_falha"] = tipo.isin(config.TIPOS_FALHA)
    df["is_preventiva"] = tipo == config.TIPO_PREVENTIVA
    df["classificado"] = tipo.isin(config.TIPOS_FALHA + [config.TIPO_PREVENTIVA])
    return df


def _anonimizar_tag(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Gera código genérico sequencial por TAG, ordenado alfabeticamente
    pelo TAG original para que a atribuição seja determinística/reprodutível.
    """
    tags_unicos = sorted(df["TAG"].unique())
    mapeamento = {tag: f"EQ{i+1:03d}" for i, tag in enumerate(tags_unicos)}

    df = df.copy()
    df["tag_code"] = df["TAG"].map(mapeamento)
    return df, mapeamento


# Nomes de técnicos e de empresas terceirizadas encontrados na inspeção
# real da planilha de monta-cargas, que sobrevivem à remoção estrutural
# (hífen/parênteses/"ok Fulano") por aparecerem sem separador claro (ex.:
# "monta carga não está subindo Fulano", "Teste monta carga Empresa X").
#
# VERSÃO PÚBLICA: os nomes reais foram trocados por nomes fictícios para
# não expor pessoas nem empresas. Na versão usada com a planilha real, esta
# lista tem os nomes encontrados na inspeção dos dados.
# Lista fechada e auditável — construída inspecionando os dados reais,
# não uma lista genérica de nomes comuns (o que arriscaria remover
# palavras legítimas da descrição por coincidência).
NOMES_ATRIBUICAO_CONHECIDOS = {
    "fulano", "ciclano", "beltrano", "deltrano", "joaozinho",
    "empresax", "terceirizadaz",
}


def _remover_atribuicao_final(texto: str) -> str:
    """Remove a atribuição de nome de técnico / empresa terceirizada no
    final da descrição (ex.: "porta não abre - Fulano" -> "porta não
    abre"; "monta carga com problema (ok Ciclano)" -> "monta carga com
    problema"; "monta carga não está subindo Beltrano" -> idem).

    Investigação sobre a planilha real de monta-cargas mostrou que mais da
    metade das descrições de falha tem algum padrão de atribuição desse
    tipo no final — quase sempre o primeiro nome (ou iniciais) de um
    técnico, ou o nome de uma empresa terceirizada, nunca parte da
    descrição do problema em si. Trata-se do mesmo tipo de informação
    identificável que o restante do projeto já exclui (matrícula, oficina
    executante etc.), só que embutida em texto livre em vez de estar em
    coluna própria — por isso é removida aqui, antes de qualquer
    persistência, e não apenas na análise (`app/processing/text_analysis.py`).

    Estratégia em camadas: primeiro remove os padrões estruturais mais
    comuns (parênteses no final; "ok NOME" no final; segmento após o
    último hífen); depois, como rede de segurança, remove qualquer
    ocorrência isolada dos nomes específicos já identificados nesta base
    (`NOMES_ATRIBUICAO_CONHECIDOS`), que não seguem nenhum separador
    estrutural.
    """
    texto = texto.strip()

    # Parênteses no final: "(ok Ciclano)", "(Ciclano)".
    texto = re.sub(r"\s*\([^)]*\)\s*$", "", texto).strip()

    # "ok Fulano" / "OK FULANO" no final (com ou sem pontuação residual).
    texto = re.sub(
        r"[\s,-]*\bo\.?k\.?[\s.]+[A-Za-zÀ-ÿ/\\]+(?:\s+[A-Za-zÀ-ÿ]+)?[\s.)]*$",
        "",
        texto,
        flags=re.IGNORECASE,
    ).strip()

    # Segmento após o último hífen.
    partes = texto.rsplit("-", 1)
    texto = partes[0].strip() if len(partes) == 2 else texto

    # Rede de segurança: nomes conhecidos sem separador estrutural.
    if NOMES_ATRIBUICAO_CONHECIDOS:
        padrao_nomes = re.compile(
            r"\b(" + "|".join(re.escape(n) for n in NOMES_ATRIBUICAO_CONHECIDOS) + r")\b",
            flags=re.IGNORECASE,
        )
        texto = padrao_nomes.sub("", texto)
        texto = re.sub(r"[\\/]", " ", texto)  # separadores "Nome1/Nome2" que sobraram
        texto = re.sub(r"\s+", " ", texto).strip(" -,.")

    return texto


def _padrao_tag_flexivel(tag: str) -> str:
    """Fragmento de regex para um TAG que casa também com variantes soltas
    no texto livre — "MC02" escrito como "MC 02", "MC-02", "Mc  02" etc.
    (achado real: 37 descrições tinham o TAG com espaço/hífen entre a
    parte alfabética e a numérica, que o casamento por igualdade exata
    não pegava — descoberto ao inspecionar exemplos reais na tela de
    componentes, briefing complementar 3 item 9).

    Separa o TAG em prefixo alfabético + sufixo numérico e permite
    espaços/hífen entre os dois; TAGs que não seguem esse formato caem
    para o escape literal simples.
    """
    m = re.match(r"^([A-Za-z]+)(\d+)$", tag)
    if not m:
        return re.escape(tag)
    prefixo, sufixo = m.groups()
    return re.escape(prefixo) + r"[\s-]*" + re.escape(sufixo)


def _redigir_descricao(df: pd.DataFrame) -> pd.DataFrame:
    """Redige o texto livre de `Descrição Tarefa` antes de persistir
    (briefing complementar de análise exploratória, item 1): remove a
    atribuição de nome/empresa após o último hífen (ver
    `_remover_atribuicao_final`) e qualquer menção ao TAG ORIGINAL do
    equipamento embutida na frase, inclusive variantes com espaço/hífen
    (ex.: "Lâmpada do MC01 queimada", "MC 01 travado", "MC-02").

    A coluna `TAG` já está anonimizada à parte (`tag_code`), mas o texto
    livre frequentemente repete o TAG real dentro da própria frase. Sem
    essa redação, tanto o TAG quanto nomes de pessoas/empresas
    acabariam persistidos no banco em texto livre, contornando a
    anonimização já aplicada às colunas estruturadas. A substituição do
    TAG usa a lista de TAGs reais efetivamente presentes no escopo (não
    só o da própria linha), para cobrir eventuais menções cruzadas entre
    equipamentos na mesma descrição.
    """
    df = df.copy()
    if "Descrição Tarefa" not in df.columns:
        df["descricao_tarefa_redigida"] = None
        return df

    tags_reais = sorted(df["TAG"].dropna().unique(), key=len, reverse=True)
    padrao = None
    if tags_reais:
        padrao = re.compile(
            r"\b(" + "|".join(_padrao_tag_flexivel(t) for t in tags_reais) + r")\b", flags=re.IGNORECASE
        )

    def redigir(texto):
        if pd.isna(texto):
            return None
        texto = _remover_atribuicao_final(str(texto))
        return padrao.sub("[equip]", texto) if padrao else texto

    df["descricao_tarefa_redigida"] = df["Descrição Tarefa"].map(redigir)
    return df


def _estimar_componente(df: pd.DataFrame) -> pd.DataFrame:
    """Estimativa por palavra-chave do componente associado a cada evento
    de falha (briefing complementar 2, seção 3) — ver
    `app/processing/component_mapping.py` para a lógica e as ressalvas.
    Calculada só para `is_falha=True`; `None` para os demais eventos
    (não é aplicável/não faz sentido estimar componente de uma
    preventiva ou de um tipo não classificado).
    """
    df = df.copy()
    df["componente_estimado"] = None
    mask_falha = df["is_falha"]
    df.loc[mask_falha, "componente_estimado"] = df.loc[mask_falha, "descricao_tarefa_redigida"].map(
        component_mapping.classificar_componente
    )
    return df


def _perturbar_hh(df: pd.DataFrame) -> pd.DataFrame:
    """Perturbação aleatória controlada de HH Apropriado (briefing seção 4).

    Fator uniforme em [PERTURBACAO_MIN, PERTURBACAO_MAX], seed fixa. Só o
    valor perturbado é mantido — é o valor usado tanto no treinamento
    quanto em qualquer exibição.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    fator = rng.uniform(config.PERTURBACAO_MIN, config.PERTURBACAO_MAX, size=len(df))

    df = df.copy()
    df["hh_apropriado"] = df["HH Apropriado"].to_numpy() * fator
    return df


def processar_planilha(xlsx_path: str = None, sheet_name: str = None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Executa o pipeline de limpeza + anonimização e devolve o DataFrame
    final (pronto para persistir) e o mapeamento TAG real -> código.
    """
    xlsx_path = xlsx_path or config.RAW_XLSX_PATH
    sheet_name = sheet_name or config.RAW_XLSX_SHEET

    df = _carregar_bruto(xlsx_path, sheet_name)
    n_bruto = len(df)

    df = _limpar(df)
    n_pos_limpeza = len(df)

    df = _filtrar_escopo(df)
    n_pos_escopo = len(df)

    df = _classificar_tipo(df)
    df, mapeamento = _anonimizar_tag(df)
    df = _redigir_descricao(df)
    df = _estimar_componente(df)
    df = _perturbar_hh(df)

    # Remove duplicatas exatas de (Nº OS, NºTarefa, Início) — mesma
    # apontamento repetida; ver briefing seção 2 e investigação no chat.
    antes = len(df)
    df = df.drop_duplicates(subset=["Nº OS", "NºTarefa", "Início"], keep="first")
    n_dup_removidas = antes - len(df)

    print(f"Linhas brutas:                       {n_bruto}")
    print(f"Linhas após limpeza (C, datas):      {n_pos_limpeza}")
    print(f"Linhas após filtro de escopo {config.TAG_PREFIXOS_ESCOPO}: {n_pos_escopo}")
    print(f"Duplicatas exatas removidas:         {n_dup_removidas}")
    print(f"Linhas finais (eventos):             {len(df)}")
    print(f"Equipamentos (TAGs) distintos:       {df['tag_code'].nunique()}")

    n_falha = int(df["is_falha"].sum())
    n_nao_identificado = int((df.loc[df["is_falha"], "componente_estimado"] == component_mapping.NAO_IDENTIFICADO).sum())
    if n_falha:
        print(f"Eventos de falha (MC/CP):            {n_falha}")
        print(f"  componente_estimado='{component_mapping.NAO_IDENTIFICADO}': "
              f"{n_nao_identificado} ({100 * n_nao_identificado / n_falha:.1f}%)")

    return df, mapeamento


def carregar_no_banco(df: pd.DataFrame, mapeamento: dict[str, str]) -> None:
    """Persiste o DataFrame tratado no banco (SQLite/SQL Server via config)."""
    init_db()
    session = get_session()
    try:
        # Zera as tabelas para permitir reprocessamento idempotente.
        session.query(EventoManutencao).delete()
        session.query(Equipamento).delete()
        session.query(Componente).delete()
        session.query(TagMapping).delete()
        session.commit()

        session.bulk_save_objects(
            [TagMapping(tag_original=orig, tag_code=code) for orig, code in mapeamento.items()]
        )

        eventos = [
            EventoManutencao(
                tag_code=row["tag_code"],
                os_num=int(row["Nº OS"]),
                tarefa_num=int(row["NºTarefa"]),
                inicio=row["Início"].date(),
                fim=row["Fim"].date(),
                tipo_manutencao=row["Tipo de Manutenção"],
                is_falha=bool(row["is_falha"]),
                is_preventiva=bool(row["is_preventiva"]),
                classificado=bool(row["classificado"]),
                hh_apropriado=float(row["hh_apropriado"]),
                area=row["Área"] if pd.notna(row["Área"]) else None,
                sist_oper=row["Sist Oper"] if pd.notna(row["Sist Oper"]) else None,
                descricao_tarefa=(
                    row["descricao_tarefa_redigida"] if pd.notna(row.get("descricao_tarefa_redigida")) else None
                ),
                componente_estimado=(
                    row["componente_estimado"] if pd.notna(row.get("componente_estimado")) else None
                ),
            )
            for row in df.to_dict("records")
        ]
        session.bulk_save_objects(eventos)
        session.commit()

        # Dimensão de equipamentos (resumo por tag_code).
        resumo = (
            df.groupby("tag_code")["Início"]
            .agg(["min", "max", "count"])
            .rename(columns={"min": "primeiro", "max": "ultimo", "count": "total"})
        )
        equipamentos = [
            Equipamento(
                tag_code=tag_code,
                tipo=config.TIPO_EQUIPAMENTO_ESCOPO,
                primeiro_evento=row.primeiro.date(),
                ultimo_evento=row.ultimo.date(),
                total_eventos=int(row.total),
            )
            for tag_code, row in resumo.iterrows()
        ]
        session.bulk_save_objects(equipamentos)

        # Cadastro de componentes (catálogo de referência técnica geral —
        # briefing complementar 2, seção 1). `termos_busca` é a mesma
        # lista usada para estimar `componente_estimado`, guardada aqui só
        # para exibir com transparência o critério da estimativa.
        componentes = [
            Componente(
                nome=nome,
                fabricante_referencia=dados["fabricante_referencia"],
                vida_util_indicada=dados["vida_util_indicada"],
                unidade=dados["unidade"],
                criterio_substituicao=dados["criterio_substituicao"],
                termos_busca=", ".join(component_mapping.TERMOS_COMPONENTE[nome]),
            )
            for nome, dados in component_mapping.CATALOGO_COMPONENTES.items()
        ]
        session.bulk_save_objects(componentes)
        session.commit()
    finally:
        session.close()


def executar(xlsx_path: str = None, sheet_name: str = None) -> None:
    df, mapeamento = processar_planilha(xlsx_path, sheet_name)
    carregar_no_banco(df, mapeamento)
    print("\nCarga concluída no banco:", config.DATABASE_URL)


if __name__ == "__main__":
    executar()
