"""
Módulo de processamento (ii) — linha do tempo por TAG, rótulos por janela
deslizante (falha_30d / falha_60d) e engenharia de variáveis.

Este módulo NÃO decide qual é o "evento de falha" (isso já veio pronto do
módulo de dados, campo ``is_falha``). O que ele faz é:

1. Gerar, para cada equipamento (TAG), uma grade de "datas de referência"
   ao longo do histórico observado daquele equipamento (briefing seção 5).
2. Para cada data de referência, calcular as features descritas na
   seção 6 do briefing, olhando só para o passado daquele TAG (nunca para
   o futuro — isso evitaria vazamento de informação).
3. Rotular cada data de referência com ``falha_30d`` / ``falha_60d``:
   houve algum evento is_falha=True nos N dias SEGUINTES à data de
   referência?

Limitação assumida e documentada (briefing seções 3.2 e 7): uma data de
referência só é gerada se `data_referencia + horizonte` couber dentro do
período coberto pela base (hoje, jul/2008 a set/2025). Sem isso, um "não
houve falha" nos dados seria ambíguo — poderia ser que a falha
simplesmente ainda não tinha sido registrada porque a base acabou ali
(censura à direita). Quanto maior o horizonte, mais janelas finais são
descartadas (90 dias descarta mais que 60, que descarta mais que 30); com
~17 anos de histórico a perda é pequena em termos relativos.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

import config
from app.data.db import get_session
from app.data.models import EventoManutencao, Equipamento, Componente


def carregar_eventos() -> pd.DataFrame:
    """Lê todos os eventos de manutenção tratados do banco."""
    session = get_session()
    try:
        rows = session.query(EventoManutencao).all()
        df = pd.DataFrame(
            [
                {
                    "tag_code": r.tag_code,
                    "inicio": pd.Timestamp(r.inicio),
                    "fim": pd.Timestamp(r.fim),
                    "tipo_manutencao": r.tipo_manutencao,
                    "is_falha": r.is_falha,
                    "is_preventiva": r.is_preventiva,
                    "classificado": r.classificado,
                    "hh_apropriado": r.hh_apropriado,
                    "area": r.area,
                    "sist_oper": r.sist_oper,
                    "descricao_tarefa": r.descricao_tarefa,
                    "componente_estimado": r.componente_estimado,
                }
                for r in rows
            ]
        )
        return df.sort_values(["tag_code", "inicio"]).reset_index(drop=True)
    finally:
        session.close()


def _equipamento_para_dict(r) -> dict:
    return {
        "tag_code": r.tag_code,
        "tipo": r.tipo,
        "primeiro_evento": pd.Timestamp(r.primeiro_evento) if r.primeiro_evento else None,
        "ultimo_evento": pd.Timestamp(r.ultimo_evento) if r.ultimo_evento else None,
        "total_eventos": r.total_eventos,
        # Cadastro manual (briefing complementar 4, seção 2) — não vem da
        # planilha histórica, não entra em nenhuma feature do modelo.
        "fabricante": r.fabricante,
        "modelo": r.modelo,
        "capacidade_carga_kg": r.capacidade_carga_kg,
        "ano_instalacao": r.ano_instalacao,
        "setor_localizacao": r.setor_localizacao,
        "status": r.status,
        "responsavel_tecnico": r.responsavel_tecnico,
        "observacoes_tecnicas": r.observacoes_tecnicas,
    }


def carregar_maquinas() -> pd.DataFrame:
    """Cadastro de máquinas (briefing complementar 2, seção 2) — dimensão
    de equipamentos já calculada pelo módulo de dados na ingestão, mais os
    campos de cadastro manual (briefing complementar 4, seção 2)."""
    session = get_session()
    try:
        rows = session.query(Equipamento).order_by(Equipamento.tag_code).all()
        return pd.DataFrame([_equipamento_para_dict(r) for r in rows])
    finally:
        session.close()


def carregar_maquina(tag_code: str) -> dict | None:
    """Um único registro do cadastro de máquinas, ou `None` se o TAG não
    existir (usado na página de detalhe do equipamento e no formulário de
    edição — briefing complementar 4)."""
    session = get_session()
    try:
        r = session.get(Equipamento, tag_code)
        return _equipamento_para_dict(r) if r is not None else None
    finally:
        session.close()


def carregar_componentes() -> pd.DataFrame:
    """Cadastro de componentes (briefing complementar 2, seção 1) —
    catálogo de referência técnica geral, não dado específico da
    empresa."""
    session = get_session()
    try:
        rows = session.query(Componente).order_by(Componente.nome).all()
        return pd.DataFrame(
            [
                {
                    "nome": r.nome,
                    "fabricante_referencia": r.fabricante_referencia,
                    "vida_util_indicada": r.vida_util_indicada,
                    "unidade": r.unidade,
                    "criterio_substituicao": r.criterio_substituicao,
                    "termos_busca": r.termos_busca,
                }
                for r in rows
            ]
        )
    finally:
        session.close()


def _grade_datas_referencia(datas_tag: np.ndarray, data_max_base: pd.Timestamp, horizonte: int) -> pd.DatetimeIndex:
    """Datas de referência para um TAG: do primeiro evento observado até
    (data_max_base - horizonte), em passos de REFERENCE_STEP_DAYS dias.
    """
    inicio = pd.Timestamp(datas_tag.min())
    fim = data_max_base - timedelta(days=horizonte)
    if fim < inicio:
        return pd.DatetimeIndex([])
    return pd.date_range(inicio, fim, freq=f"{config.REFERENCE_STEP_DAYS}D")


def _features_para_grupo(grupo: pd.DataFrame, refs: pd.DatetimeIndex) -> pd.DataFrame:
    """Calcula as features da seção 6 do briefing para todas as datas de
    referência de um único TAG, olhando apenas para trás no tempo.
    """
    datas = grupo["inicio"].to_numpy()
    falha_datas = grupo.loc[grupo["is_falha"], "inicio"].to_numpy()
    prev_datas = grupo.loc[grupo["is_preventiva"], "inicio"].to_numpy()
    hh = grupo["hh_apropriado"].to_numpy()
    area = grupo["area"].to_numpy()
    sist_oper = grupo["sist_oper"].to_numpy()

    refs_np = refs.to_numpy()

    # -- tempo desde a última falha (dias) --------------------------------
    if len(falha_datas) == 0:
        tempo_desde_ultima_falha = np.full(len(refs_np), np.nan)
    else:
        idx_falha = np.searchsorted(falha_datas, refs_np, side="right") - 1
        tempo_desde_ultima_falha = np.where(
            idx_falha >= 0,
            (refs_np - falha_datas[np.clip(idx_falha, 0, None)]).astype("timedelta64[D]").astype(float),
            np.nan,
        )

    # -- frequência de falhas em janelas anteriores ------------------------
    freq_cols = {}
    for janela in config.JANELAS_FREQ_FALHAS:
        lim_inf = refs_np - np.timedelta64(janela, "D")
        hi = np.searchsorted(falha_datas, refs_np, side="right")
        lo = np.searchsorted(falha_datas, lim_inf, side="right")
        freq_cols[f"freq_falhas_{janela}d"] = hi - lo

    # -- tempo médio de reparo histórico (todas as intervenções até a ref) -
    idx_hist = np.searchsorted(datas, refs_np, side="right")
    tempo_medio_reparo = np.array(
        [hh[:i].mean() if i > 0 else np.nan for i in idx_hist]
    )
    n_eventos_historico = idx_hist  # quantidade de eventos conhecidos até a ref

    # -- contagem de intervenções preventivas recentes ---------------------
    lim_inf_prev = refs_np - np.timedelta64(config.JANELA_PREVENTIVAS, "D")
    hi_p = np.searchsorted(prev_datas, refs_np, side="right")
    lo_p = np.searchsorted(prev_datas, lim_inf_prev, side="right")
    contagem_preventivas = hi_p - lo_p

    # -- área / sistema operacional conhecidos na data de referência -------
    # (valor do evento mais recente até a ref; None se ainda não há histórico)
    area_ref = np.array(
        [area[i - 1] if i > 0 else None for i in idx_hist], dtype=object
    )
    sist_oper_ref = np.array(
        [sist_oper[i - 1] if i > 0 else None for i in idx_hist], dtype=object
    )

    out = pd.DataFrame(
        {
            "data_referencia": refs,
            "tempo_desde_ultima_falha": tempo_desde_ultima_falha,
            "tempo_medio_reparo": tempo_medio_reparo,
            "n_eventos_historico": n_eventos_historico,
            "contagem_preventivas_recentes": contagem_preventivas,
            "area": area_ref,
            "sist_oper": sist_oper_ref,
            **freq_cols,
        }
    )
    return out


def _rotular(grupo_falhas: np.ndarray, refs: pd.DatetimeIndex, horizonte: int) -> np.ndarray:
    """falha_Nd = 1 se existe evento is_falha em (ref, ref + horizonte]."""
    refs_np = refs.to_numpy()
    lim_sup = refs_np + np.timedelta64(horizonte, "D")
    hi = np.searchsorted(grupo_falhas, lim_sup, side="right")
    lo = np.searchsorted(grupo_falhas, refs_np, side="right")  # estritamente após a ref
    return (hi - lo) > 0


def construir_dataset(horizonte: int, eventos: pd.DataFrame = None) -> pd.DataFrame:
    """Monta o dataset de modelagem (features + rótulo) para um horizonte.

    Uma linha = (tag_code, data_referencia) com as features da seção 6 e o
    rótulo binário falha_{horizonte}d.
    """
    if horizonte not in config.HORIZONTES_DIAS:
        raise ValueError(f"Horizonte {horizonte} não está em config.HORIZONTES_DIAS")

    if eventos is None:
        eventos = carregar_eventos()

    data_max_base = eventos["inicio"].max()

    blocos = []
    for tag_code, grupo in eventos.groupby("tag_code", sort=False):
        grupo = grupo.sort_values("inicio")
        refs = _grade_datas_referencia(grupo["inicio"].to_numpy(), data_max_base, horizonte)
        if len(refs) == 0:
            continue

        feats = _features_para_grupo(grupo, refs)
        falha_datas = grupo.loc[grupo["is_falha"], "inicio"].to_numpy()
        feats[f"falha_{horizonte}d"] = _rotular(falha_datas, refs, horizonte).astype(int)
        feats.insert(0, "tag_code", tag_code)
        blocos.append(feats)

    dataset = pd.concat(blocos, ignore_index=True)
    return dataset


if __name__ == "__main__":
    eventos = carregar_eventos()
    for h in config.HORIZONTES_DIAS:
        ds = construir_dataset(h, eventos)
        print(f"\n=== Horizonte {h}d ===")
        print("Linhas (tag, data_referencia):", len(ds))
        print("TAGs distintos:", ds["tag_code"].nunique())
        print("Proporção de falha:", ds[f"falha_{h}d"].mean().round(4))
        print(ds.head(3).to_string())
