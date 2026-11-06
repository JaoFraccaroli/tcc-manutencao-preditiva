"""
Modelos ORM (SQLAlchemy) para o banco de dados tratado.

Duas tabelas fazem o trabalho:

- ``TagMapping``: mapeamento TAG real -> código genérico (EQ001, EQ002...).
  É a ÚNICA tabela que contém o TAG real. Nunca é lida pelos módulos de
  processamento, modelagem ou apresentação — existe só para permitir
  reprocessar a planilha de forma consistente. Nunca deve ser exportada,
  logada ou exibida (briefing seção 4).

- ``EventoManutencao``: uma linha por evento de manutenção concluído
  (Situação == "C"), já anonimizado e com HH Apropriado perturbado.
  É a tabela usada por todo o resto da aplicação.

Sobre "rastreabilidade" (briefing seção 11): a planilha bruta não tem um
campo explícito de "componente" do equipamento; usamos
``tipo_manutencao`` + ``area`` como o par que descreve a natureza da
intervenção, já que é o que os dados realmente oferecem.
"""
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    Boolean,
    Text,
    UniqueConstraint,
    Index,
)

from app.data.db import Base


class TagMapping(Base):
    """Mapeamento interno TAG real -> código genérico. NUNCA expor."""

    __tablename__ = "tag_mapping_interno"

    id = Column(Integer, primary_key=True)
    tag_original = Column(String(64), unique=True, nullable=False)
    tag_code = Column(String(16), unique=True, nullable=False)


#: Valores permitidos para `Equipamento.status` (briefing complementar 4,
#: seção 2). Validado em `app/data/cadastro.py`.
STATUS_EQUIPAMENTO_OPCOES = ["Ativo", "Em manutenção", "Inativo"]


class Equipamento(Base):
    """Cadastro de máquinas — dimensão de equipamentos (briefing
    complementar 2, seção 2; campos cadastrais ampliados no briefing
    complementar 4, seção 2).

    `tag_code`, `tipo`, `primeiro_evento`, `ultimo_evento` e
    `total_eventos` vêm do processamento da planilha histórica (ingestão
    — `app/data/ingest.py`) e são recalculados a cada reprocessamento.

    Os demais campos (`fabricante` em diante) são **cadastro manual**,
    preenchidos pelo usuário no formulário da aplicação
    (`app/data/cadastro.py::inserir_maquina`/`atualizar_maquina`) — NÃO
    vêm da planilha de manutenção, NÃO são recalculados na ingestão e
    NÃO entram em nenhuma feature do modelo preditivo nem em nenhum
    indicador calculado. São puramente descritivos, exibidos só no
    cadastro de máquinas e no detalhe do equipamento. Nunca são
    pré-preenchidos com valor inventado — ausência fica `None` (exibido
    como "não informado"), nunca um chute.
    """

    __tablename__ = "equipamentos"

    tag_code = Column(String(16), primary_key=True)
    tipo = Column(String(32), nullable=True)  # ex.: "Monta-carga" (config.TIPO_EQUIPAMENTO_ESCOPO)
    primeiro_evento = Column(Date, nullable=True)
    ultimo_evento = Column(Date, nullable=True)
    total_eventos = Column(Integer, default=0)

    # --- Cadastro manual (briefing complementar 4, seção 2) -----------------
    fabricante = Column(String(128), nullable=True)
    modelo = Column(String(128), nullable=True)
    capacidade_carga_kg = Column(Float, nullable=True)
    ano_instalacao = Column(Integer, nullable=True)
    setor_localizacao = Column(String(128), nullable=True)
    status = Column(String(32), nullable=False, default="Ativo")
    responsavel_tecnico = Column(String(128), nullable=True)
    observacoes_tecnicas = Column(Text, nullable=True)


class Componente(Base):
    """Cadastro de componentes (briefing complementar 2, seção 1).

    Catálogo de referência TÉCNICA GERAL (fabricantes, vida útil típica,
    critério de substituição) — não é dado específico da empresa, é
    conhecimento de engenharia sobre os tipos de componente que compõem
    um monta-carga. Não muda por evento.

    `termos_busca` é a MESMA lista usada por
    `app/processing/component_mapping.py` para estimar qual componente
    está associado a cada evento de falha — guardada aqui só para exibir
    de forma transparente com que critério a estimativa foi feita.
    """

    __tablename__ = "componentes"

    id = Column(Integer, primary_key=True)
    nome = Column(String(64), unique=True, nullable=False)
    fabricante_referencia = Column(String(128), nullable=True)
    vida_util_indicada = Column(String(128), nullable=True)
    unidade = Column(String(32), nullable=True)
    criterio_substituicao = Column(String(256), nullable=True)
    termos_busca = Column(String(256), nullable=True)  # separados por vírgula


class EventoManutencao(Base):
    """Um evento de manutenção concluído (Situação == 'C'), anonimizado."""

    __tablename__ = "eventos_manutencao"
    __table_args__ = (
        UniqueConstraint("os_num", "tarefa_num", "inicio", name="uq_evento_natural"),
        Index("ix_evento_tag_inicio", "tag_code", "inicio"),
    )

    id = Column(Integer, primary_key=True)
    tag_code = Column(String(16), nullable=False)

    os_num = Column(Integer, nullable=False)
    tarefa_num = Column(Integer, nullable=False)

    inicio = Column(Date, nullable=False)
    fim = Column(Date, nullable=False)

    tipo_manutencao = Column(String(8), nullable=False)
    # True apenas para MC/CP (briefing 3.2). Calculado na ingestão para
    # evitar que cada consumidor reimplemente a regra de negócio.
    is_falha = Column(Boolean, nullable=False, default=False)
    # True para MP — não é falha, mas é sinal de manutenção preventiva
    # recente (briefing seção 6).
    is_preventiva = Column(Boolean, nullable=False, default=False)
    # False para os códigos não classificados (IN, ME, IP, EV, LB, PT).
    # Esses eventos são mantidos no histórico geral mas não participam da
    # construção de rótulos (briefing seção 3.2, limitação assumida).
    classificado = Column(Boolean, nullable=False, default=False)

    hh_apropriado = Column(Float, nullable=False)  # já perturbado

    area = Column(String(16), nullable=True)
    sist_oper = Column(String(16), nullable=True)

    # Texto livre da OS (briefing complementar de análise exploratória).
    # Qualquer menção ao TAG ORIGINAL do equipamento é substituída por um
    # marcador neutro ("[equip]") já na ingestão (ver ingest.py,
    # _redigir_descricao) — nunca o TAG real chega a ser persistido aqui,
    # mesmo que apareça embutido no texto da descrição na planilha bruta
    # (ex.: "Lâmpada do MC01 queimada").
    descricao_tarefa = Column(Text, nullable=True)

    # ESTIMATIVA por palavra-chave sobre `descricao_tarefa`, não um campo
    # medido pelo sistema de origem (que não registra qual componente
    # falhou) — briefing complementar 2, seção 3. Calculado só para
    # eventos com is_falha=True; None para os demais. Ver
    # `app/processing/component_mapping.py` para a lógica e o critério de
    # desempate quando a descrição bate com mais de um componente.
    componente_estimado = Column(String(64), nullable=True)
