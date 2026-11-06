import datetime

import pandas as pd
from flask import Flask

import config


def _data_br(valor):
    """Formata data como dd/mm/aaaa (briefing complementar 3, item 4).
    Aceita `date`/`datetime`/`pd.Timestamp` ou string ISO ("aaaa-mm-dd");
    usado como filtro Jinja `|data_br` em vez de cada template formatar a
    data à sua maneira, para propagar o formato de uma vez só.

    `pd.isna` cobre tanto `None` quanto `pd.NaT` — uma máquina/componente
    recém-cadastrado sem eventos ainda vira `NaT` ao entrar numa coluna
    datetime64 do pandas junto com datas reais, e `NaT` passa no
    `isinstance(..., datetime.datetime)` mas quebra em `.strftime()`.
    """
    if valor is None or (pd.isna(valor) if not isinstance(valor, str) else False):
        return "—"
    if isinstance(valor, str):
        try:
            valor = datetime.date.fromisoformat(valor)
        except ValueError:
            return valor
    if isinstance(valor, (datetime.date, datetime.datetime)):
        return valor.strftime("%d/%m/%Y")
    return valor


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = config.SECRET_KEY

    from app.presentation.routes import bp

    app.register_blueprint(bp)

    app.jinja_env.filters["data_br"] = _data_br

    from app.processing.tipos_manutencao import descricao_tipo

    app.jinja_env.filters["tipo_desc"] = descricao_tipo

    return app
