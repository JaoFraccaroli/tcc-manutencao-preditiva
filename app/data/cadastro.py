"""
Inserção/edição de máquinas e componentes pela própria aplicação
(briefing complementar 3, item 6; campos cadastrais ampliados no
briefing complementar 4, seção 2) — antes só era possível via
script/seed de ingestão. Cada função valida campos obrigatórios,
unicidade e formato antes de gravar, devolvendo uma mensagem de erro
(string) em vez de lançar exceção quando a validação falha, para a
camada de apresentação exibir ao usuário.

Os campos cadastrais de máquina (fabricante, modelo, capacidade de
carga, ano de instalação, setor/localização, status, responsável
técnico, observações) são CADASTRO MANUAL — preenchidos pelo usuário,
nunca inferidos ou inventados. Ausência fica `None` ("não informado"),
nunca um valor chutado. Não vêm da planilha histórica e não entram em
nenhuma feature do modelo preditivo.
"""
from app.data.db import get_session
from app.data.models import Equipamento, Componente, STATUS_EQUIPAMENTO_OPCOES


def _texto_ou_none(valor: str) -> str | None:
    valor = (valor or "").strip()
    return valor or None


def _montar_campos_maquina(form: dict) -> tuple[dict, str | None]:
    """Valida e converte os campos de cadastro de máquina vindos do
    formulário. Devolve (campos_prontos_para_gravar, erro); `erro` não é
    `None` se algum campo estiver em formato inválido — nesse caso o
    dict de campos deve ser ignorado."""
    campos = {
        "tipo": _texto_ou_none(form.get("tipo")),
        "fabricante": _texto_ou_none(form.get("fabricante")),
        "modelo": _texto_ou_none(form.get("modelo")),
        "setor_localizacao": _texto_ou_none(form.get("setor_localizacao")),
        "responsavel_tecnico": _texto_ou_none(form.get("responsavel_tecnico")),
        "observacoes_tecnicas": _texto_ou_none(form.get("observacoes_tecnicas")),
    }

    capacidade = _texto_ou_none(form.get("capacidade_carga_kg"))
    if capacidade is not None:
        try:
            campos["capacidade_carga_kg"] = float(capacidade.replace(",", "."))
        except ValueError:
            return {}, f'Capacidade de carga inválida: "{capacidade}" (use só números).'
    else:
        campos["capacidade_carga_kg"] = None

    ano = _texto_ou_none(form.get("ano_instalacao"))
    if ano is not None:
        try:
            ano_int = int(ano)
        except ValueError:
            return {}, f'Ano de instalação inválido: "{ano}" (use só números, ex.: 2015).'
        if not (1900 <= ano_int <= 2100):
            return {}, f'Ano de instalação fora do intervalo esperado: "{ano}".'
        campos["ano_instalacao"] = ano_int
    else:
        campos["ano_instalacao"] = None

    status = _texto_ou_none(form.get("status")) or "Ativo"
    if status not in STATUS_EQUIPAMENTO_OPCOES:
        return {}, f'Status inválido: "{status}". Use um de: {", ".join(STATUS_EQUIPAMENTO_OPCOES)}.'
    campos["status"] = status

    return campos, None


def inserir_maquina(tag_code: str, form: dict = None) -> str | None:
    """Insere uma nova máquina no cadastro. `form` traz os campos
    cadastrais opcionais (briefing complementar 4, seção 2) — mesmas
    chaves de `_montar_campos_maquina`. Devolve `None` em caso de
    sucesso, ou uma mensagem de erro (string) se a validação falhar —
    nesse caso nada é gravado."""
    form = form or {}
    tag_code = (tag_code or "").strip()
    if not tag_code:
        return "Informe o código (TAG) da máquina."

    campos, erro = _montar_campos_maquina(form)
    if erro:
        return erro
    if not campos["tipo"]:
        return "Informe o nome/apelido do equipamento."

    session = get_session()
    try:
        existente = session.get(Equipamento, tag_code)
        if existente is not None:
            return f'Já existe uma máquina cadastrada com o código "{tag_code}".'

        session.add(
            Equipamento(
                tag_code=tag_code,
                primeiro_evento=None,
                ultimo_evento=None,
                total_eventos=0,
                **campos,
            )
        )
        session.commit()
        return None
    finally:
        session.close()


def atualizar_maquina(tag_code: str, form: dict) -> str | None:
    """Atualiza os campos cadastrais (seção 2 do briefing complementar 4)
    de uma máquina já existente. Não mexe em `primeiro_evento`,
    `ultimo_evento` nem `total_eventos` — esses continuam vindo só da
    ingestão. Devolve `None` em caso de sucesso, ou mensagem de erro."""
    campos, erro = _montar_campos_maquina(form)
    if erro:
        return erro
    if not campos["tipo"]:
        return "Informe o nome/apelido do equipamento."

    session = get_session()
    try:
        existente = session.get(Equipamento, tag_code)
        if existente is None:
            return f'Máquina "{tag_code}" não encontrada.'

        for campo, valor in campos.items():
            setattr(existente, campo, valor)
        session.commit()
        return None
    finally:
        session.close()


def inserir_componente(
    nome: str,
    fabricante_referencia: str = None,
    vida_util_indicada: str = None,
    unidade: str = None,
    criterio_substituicao: str = None,
    termos_busca: str = None,
) -> str | None:
    """Insere um novo componente no catálogo. Devolve `None` em caso de
    sucesso, ou uma mensagem de erro (string) se a validação falhar."""
    nome = (nome or "").strip()
    if not nome:
        return "Informe o nome do componente."

    session = get_session()
    try:
        existente = session.query(Componente).filter(Componente.nome.ilike(nome)).first()
        if existente is not None:
            return f'Já existe um componente cadastrado com o nome "{nome}".'

        session.add(
            Componente(
                nome=nome,
                fabricante_referencia=_texto_ou_none(fabricante_referencia),
                vida_util_indicada=_texto_ou_none(vida_util_indicada),
                unidade=_texto_ou_none(unidade),
                criterio_substituicao=_texto_ou_none(criterio_substituicao),
                termos_busca=_texto_ou_none(termos_busca),
            )
        )
        session.commit()
        return None
    finally:
        session.close()
