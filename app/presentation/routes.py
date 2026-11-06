"""
Módulo de apresentação (iv) — rotas Flask.

Princípio geral (briefing seção 1): a aplicação é descritiva e preditiva.
Mostra indicadores e probabilidades; NUNCA ordena equipamentos por risco
por padrão nem os rotula como "prioritários" — a tabela de equipamentos
é ordenada por código (ou pelo critério que o usuário escolher
explicitamente na URL), nunca por probabilidade de falha por padrão.
"""
import json
import math
import os

import pandas as pd
from flask import Blueprint, render_template, request, send_from_directory, abort, redirect, url_for, flash

import config
from app.processing.dataset import carregar_eventos, carregar_maquinas, carregar_maquina, carregar_componentes
from app.processing import text_analysis, exploratory, component_mapping
from app.modeling.predict import snapshot_atual
from app.data import cadastro
from app.data.models import STATUS_EQUIPAMENTO_OPCOES
from app.presentation import plots

bp = Blueprint("main", __name__)

FIGURES_DIR = os.path.join(config.BASE_DIR, "reports", "figures")
MODELS_DIR = os.path.join(config.BASE_DIR, "models")

_cache = {}


def _get_eventos():
    if "eventos" not in _cache:
        _cache["eventos"] = carregar_eventos()
    return _cache["eventos"]


def _get_snapshot():
    if "snapshot" not in _cache:
        _cache["snapshot"] = snapshot_atual(_get_eventos())
    return _cache["snapshot"]


@bp.route("/atualizar")
def atualizar():
    """Limpa o cache em memória — força reler o banco e reaplicar os
    modelos no próximo acesso (útil após reingestão/re-treino)."""
    _cache.clear()
    return {"status": "cache limpo"}


def _resumo_por_equipamento(eventos, snapshot) -> list[dict]:
    """Uma linha por equipamento juntando cadastro da máquina + componente
    mais frequente + data sugerida com o motivo — visão consolidada
    pedida para aparecer logo na visão geral (briefing complementar 4),
    em vez de só no detalhe de cada equipamento."""
    maquinas_por_tag = {m["tag_code"]: m for m in carregar_maquinas().to_dict("records")}
    linhas = []
    for linha_snap in snapshot.to_dict("records"):
        tag_code = linha_snap["tag_code"]
        maquina = maquinas_por_tag.get(tag_code, {})

        eventos_tag = eventos[eventos["tag_code"] == tag_code]
        resumo_comp = component_mapping.tabela_resumo(eventos_tag)

        # "não identificado" NUNCA é apresentado como o componente mais
        # frequente (briefing de evidências, item 25): o componente de
        # destaque é o mais frequente ENTRE OS IDENTIFICADOS, com o
        # denominador explícito (total de falhas do equipamento); a parcela
        # sem componente identificado aparece separada.
        total_falhas_tag = int(resumo_comp["ocorrencias"].sum()) if len(resumo_comp) else 0
        identificados = resumo_comp[resumo_comp["componente"] != component_mapping.NAO_IDENTIFICADO]
        nao_ident = resumo_comp[resumo_comp["componente"] == component_mapping.NAO_IDENTIFICADO]

        if total_falhas_tag == 0:
            componente_top = "sem falhas registradas"
            sem_componente = "—"
        else:
            if len(identificados):
                top = identificados.iloc[0]
                componente_top = (
                    f'{top["componente"]}: {int(top["ocorrencias"])} de {total_falhas_tag} '
                    f'falhas ({top["ocorrencias"] / total_falhas_tag * 100:.1f}%)'
                )
            else:
                componente_top = "nenhum componente identificado"
            n_nao = int(nao_ident["ocorrencias"].iloc[0]) if len(nao_ident) else 0
            sem_componente = f"{n_nao} de {total_falhas_tag} ({n_nao / total_falhas_tag * 100:.1f}%)"

        linha_snap_limpa = {
            k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in linha_snap.items()
        }

        linhas.append(
            {
                "tag_code": tag_code,
                "nome": maquina.get("tipo") or "—",
                "status": maquina.get("status") or "—",
                "componente_top": componente_top,
                "sem_componente": sem_componente,
                "data_sugerida_alerta": linha_snap_limpa.get("data_sugerida_alerta"),
                "frase_data_sugerida": _frase_data_sugerida(linha_snap_limpa),
            }
        )
    return linhas


@bp.route("/")
def index():
    eventos = _get_eventos()
    snapshot = _get_snapshot()

    data_ref = eventos["inicio"].max()
    contexto = {
        "data_referencia": data_ref.date(),
        "total_equipamentos": snapshot["tag_code"].nunique(),
        "total_eventos": len(eventos),
        "total_falhas": int(eventos["is_falha"].sum()),
        "periodo_inicio": eventos["inicio"].min().date(),
        "periodo_fim": eventos["inicio"].max().date(),
        "graficos_prob": [
            (h, plots.histograma_probabilidades(snapshot, h)) for h in config.HORIZONTES_DIAS
        ],
        "grafico_tipo": plots.contagem_por_tipo(eventos),
        "resumo_equipamentos": _resumo_por_equipamento(eventos, snapshot),
    }
    return render_template("index.html", **contexto)


COLUNAS_ORDENAVEIS = {
    "tag_code": "tag_code",
    **{f"prob_{h}d": f"prob_falha_{h}d" for h in config.HORIZONTES_DIAS},
}

# Área/Sist. Oper. não aparecem nesta listagem (briefing complementar 3,
# item 5) — continuam existindo no banco e como feature do modelo, só
# não são exibidas aqui. Ainda visíveis no detalhe de cada equipamento.
COLUNAS_TABELA_BASE = [("tag_code", "TAG")]


@bp.route("/equipamentos")
def equipamentos():
    snapshot = _get_snapshot()

    ordenar_por = request.args.get("sort", "tag_code")
    direcao = request.args.get("dir", "asc")
    coluna = COLUNAS_ORDENAVEIS.get(ordenar_por, "tag_code")

    tabela = snapshot.sort_values(coluna, ascending=(direcao == "asc"))

    pagina = max(int(request.args.get("page", 1)), 1)
    por_pagina = 50
    total = len(tabela)
    inicio = (pagina - 1) * por_pagina
    tabela_pagina = tabela.iloc[inicio : inicio + por_pagina]

    colunas = COLUNAS_TABELA_BASE + [
        (f"prob_{h}d", f"Prob. falha {h}d") for h in config.HORIZONTES_DIAS
    ]

    return render_template(
        "equipamentos.html",
        equipamentos=tabela_pagina.to_dict("records"),
        colunas=colunas,
        horizontes=config.HORIZONTES_DIAS,
        limiar_alerta=config.LIMIAR_ALERTA,
        pagina=pagina,
        total_paginas=max((total - 1) // por_pagina + 1, 1),
        ordenar_por=ordenar_por,
        direcao=direcao,
        total=total,
    )


def _frase_data_sugerida(linha_snapshot: dict) -> str:
    """Frase curta explicando o motivo da data sugerida (briefing
    complementar 4, seção 1, item 3) — gerada a partir do horizonte/
    probabilidade já calculados, não é um novo texto de modelo."""
    if linha_snapshot.get("data_sugerida_alerta"):
        h = int(linha_snapshot["horizonte_sugerido"])
        prob = linha_snapshot.get(f"prob_falha_{h}d")
        return (
            f"A probabilidade estimada de falha ultrapassa o limiar de "
            f"{config.LIMIAR_ALERTA * 100:.0f}% no horizonte de {h} dias "
            f"({prob * 100:.0f}%)."
        )
    return "Nenhum horizonte avaliado ultrapassa o limiar de alerta configurado."


@bp.route("/equipamentos/<tag_code>")
def equipamento_detalhe(tag_code):
    """Página de detalhe do equipamento — desde o briefing complementar 4
    (seção 1), consolida em um único lugar: dados cadastrais da máquina,
    componentes estimados associados a ESTE equipamento (com amostra de
    descrições reais) e probabilidade + data sugerida com a frase
    explicando o motivo. Antes essas três coisas apareciam espalhadas em
    `/maquinas`, `/componentes` e aqui."""
    eventos = _get_eventos()
    snapshot = _get_snapshot()

    eventos_tag = eventos[eventos["tag_code"] == tag_code].sort_values("inicio", ascending=False)
    if eventos_tag.empty:
        abort(404)

    linha_snapshot = snapshot[snapshot["tag_code"] == tag_code].iloc[0].to_dict()
    linha_snapshot = {
        k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in linha_snapshot.items()
    }
    if linha_snapshot.get("tempo_desde_ultima_falha") is not None:
        linha_snapshot["tempo_desde_ultima_falha"] = int(linha_snapshot["tempo_desde_ultima_falha"])

    maquina = carregar_maquina(tag_code)

    # Componentes estimados SÓ deste equipamento (briefing complementar 4,
    # seção 1, item 2) — mesma lógica de /componentes, só que filtrada.
    resumo_componentes = component_mapping.tabela_resumo(eventos_tag)
    componentes_com_exemplos = []
    for linha in resumo_componentes.to_dict("records"):
        exemplos = component_mapping.exemplos_por_componente(eventos_tag, linha["componente"], limite=5)
        componentes_com_exemplos.append({**linha, "exemplos": exemplos.to_dict("records")})

    grafico = plots.historico_equipamento(eventos_tag)

    return render_template(
        "equipamento_detalhe.html",
        tag_code=tag_code,
        maquina=maquina,
        snapshot=linha_snapshot,
        frase_data_sugerida=_frase_data_sugerida(linha_snapshot),
        componentes_com_exemplos=componentes_com_exemplos,
        eventos=eventos_tag.to_dict("records"),
        grafico=grafico,
        horizontes=config.HORIZONTES_DIAS,
        limiar_alerta=config.LIMIAR_ALERTA,
    )


@bp.route("/modelos")
def modelos():
    metas = {}
    for h in config.HORIZONTES_DIAS:
        caminho = os.path.join(MODELS_DIR, f"modelo_falha_{h}d.meta.json")
        if os.path.exists(caminho):
            with open(caminho, "r", encoding="utf-8") as f:
                metas[h] = json.load(f)
    return render_template("modelos.html", metas=metas, horizontes=config.HORIZONTES_DIAS)


@bp.route("/analise-textual")
def analise_textual():
    """Categorização exploratória das descrições de falha (briefing
    complementar de análise exploratória, item 1). Análise descritiva,
    separada da modelagem preditiva — as categorias aqui não são
    features de nenhum modelo."""
    eventos = _get_eventos()
    descricoes = text_analysis.carregar_descricoes_falha(eventos)

    tabela, sementes = text_analysis.categorizar(eventos)
    palavras = text_analysis.frequencia_palavras(descricoes, top_n=20)
    bigramas = text_analysis.frequencia_bigramas(descricoes, top_n=20)

    nao_identificado = tabela[tabela["categoria"] == text_analysis.NAO_IDENTIFICADO]
    pct_nao_identificado = float(nao_identificado["percentual"].iloc[0]) if len(nao_identificado) else 0.0

    return render_template(
        "analise_textual.html",
        total_descricoes=len(descricoes),
        categorias=tabela.to_dict("records"),
        pct_nao_identificado=pct_nao_identificado,
        grafico_categorias=plots.categorias_falha(tabela),
        grafico_palavras=plots.frequencia_texto(palavras, "palavra", "Palavras mais frequentes"),
        grafico_bigramas=plots.frequencia_texto(bigramas, "bigrama", "Bigramas mais frequentes"),
    )


@bp.route("/analise-exploratoria")
def analise_exploratoria():
    """MTBF, tendência de tempo de reparo, sazonalidade e comparação
    consolidada entre equipamentos (briefing complementar, itens 2-5).
    Análise descritiva, separada da modelagem preditiva."""
    eventos = _get_eventos()

    tabela_mtbf = exploratory.mtbf_por_equipamento(eventos)
    tabela_tendencia = exploratory.tendencia_tempo_reparo(eventos)
    tabela_sazonalidade = exploratory.sazonalidade_mensal(eventos)
    tabela_comparacao = exploratory.comparacao_consolidada(eventos)

    return render_template(
        "analise_exploratoria.html",
        comparacao=tabela_comparacao.to_dict("records"),
        grafico_mtbf=plots.mtbf_equipamentos(tabela_mtbf),
        grafico_tendencia=plots.tendencia_reparo_grid(eventos, tabela_tendencia),
        grafico_sazonalidade=plots.sazonalidade(tabela_sazonalidade),
    )


@bp.route("/maquinas", methods=["GET", "POST"])
def maquinas():
    """Cadastro de máquinas (briefing complementar 2, seção 2; formulário
    de inserção — briefing complementar 3, item 6; campos cadastrais
    ampliados — briefing complementar 4, seção 2)."""
    if request.method == "POST":
        tag_code = request.form.get("tag_code", "")
        erro = cadastro.inserir_maquina(tag_code, request.form)
        if erro:
            flash(erro, "erro")
        else:
            flash(f'Máquina "{tag_code.strip()}" cadastrada com sucesso.', "sucesso")
        return redirect(url_for("main.maquinas"))

    tabela = carregar_maquinas()
    return render_template(
        "maquinas.html", maquinas=tabela.to_dict("records"), status_opcoes=STATUS_EQUIPAMENTO_OPCOES
    )


@bp.route("/maquinas/<tag_code>/editar", methods=["GET", "POST"])
def maquina_editar(tag_code):
    """Edição dos campos cadastrais de uma máquina já existente (briefing
    complementar 4, seção 2) — `primeiro_evento`/`ultimo_evento`/
    `total_eventos` não são editáveis aqui, só vêm da ingestão."""
    maquina = carregar_maquina(tag_code)
    if maquina is None:
        abort(404)

    if request.method == "POST":
        erro = cadastro.atualizar_maquina(tag_code, request.form)
        if erro:
            flash(erro, "erro")
            return redirect(url_for("main.maquina_editar", tag_code=tag_code))
        flash(f'Máquina "{tag_code}" atualizada com sucesso.', "sucesso")
        return redirect(url_for("main.equipamento_detalhe", tag_code=tag_code))

    return render_template("maquina_editar.html", maquina=maquina, status_opcoes=STATUS_EQUIPAMENTO_OPCOES)


@bp.route("/componentes", methods=["GET", "POST"])
def componentes():
    """Cadastro de componentes + classificação estimada de componente por
    evento de falha (briefing complementar 2, seções 1 e 3; formulário de
    inserção — briefing complementar 3, item 6). A classificação é uma
    ESTIMATIVA por palavra-chave sobre texto livre — documentado na
    própria tela, não um campo medido pelo sistema de origem."""
    if request.method == "POST":
        erro = cadastro.inserir_componente(
            nome=request.form.get("nome"),
            fabricante_referencia=request.form.get("fabricante_referencia"),
            vida_util_indicada=request.form.get("vida_util_indicada"),
            unidade=request.form.get("unidade"),
            criterio_substituicao=request.form.get("criterio_substituicao"),
            termos_busca=request.form.get("termos_busca"),
        )
        if erro:
            flash(erro, "erro")
        else:
            flash(f'Componente "{request.form.get("nome").strip()}" cadastrado com sucesso.', "sucesso")
        return redirect(url_for("main.componentes"))

    eventos = _get_eventos()
    catalogo = carregar_componentes()
    tabela_resumo = component_mapping.tabela_resumo(eventos)

    nao_identificado = tabela_resumo[tabela_resumo["componente"] == component_mapping.NAO_IDENTIFICADO]
    pct_nao_identificado = float(nao_identificado["percentual"].iloc[0]) if len(nao_identificado) else 0.0
    total_falhas = int(eventos["is_falha"].sum())

    return render_template(
        "componentes.html",
        catalogo=catalogo.to_dict("records"),
        resumo=tabela_resumo.to_dict("records"),
        pct_nao_identificado=pct_nao_identificado,
        total_falhas=total_falhas,
        grafico_componentes=plots.componentes_falha(tabela_resumo),
    )


@bp.route("/componentes/exemplos")
def componente_exemplos():
    """Amostra de descrições reais que geraram a estimativa de um
    componente (briefing complementar 3, item 9) — demonstra a
    correspondência palavra-chave -> componente, em vez de deixar a
    distribuição como número solto sem rastreabilidade visível.

    `nome` vem por query string (não por segmento de URL) porque vários
    nomes de componente têm "/" (ex.: "Portas / travas"), o que quebraria
    o roteamento padrão do Flask se fosse parte do caminho.
    """
    nome = request.args.get("nome", "")
    eventos = _get_eventos()
    exemplos = component_mapping.exemplos_por_componente(eventos, nome, limite=20)
    return render_template("componente_exemplos.html", nome=nome, exemplos=exemplos.to_dict("records"))


@bp.route("/figuras/<path:filename>")
def figuras(filename):
    return send_from_directory(FIGURES_DIR, filename)
