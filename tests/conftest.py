"""
Configuração comum dos testes.

IMPORTANTE: os testes NUNCA usam a planilha real nem o banco real.
- O banco é um SQLite temporário, criado numa pasta temporária.
- A planilha é gerada pelos próprios testes, com dados fictícios
  (ver tests/fabrica.py).

As variáveis de ambiente precisam ser definidas ANTES de qualquer
`import config` / `import app...`, porque o config lê o ambiente no
momento da importação. Por isso ficam no topo deste arquivo, que o
pytest carrega antes dos testes.
"""
import os
import tempfile

_PASTA_TEMPORARIA = tempfile.mkdtemp(prefix="tcc-testes-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_PASTA_TEMPORARIA, "teste.db").replace("\\", "/")
os.environ["RAW_XLSX_PATH"] = os.path.join(_PASTA_TEMPORARIA, "planilha-que-nao-existe.xlsx")
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

from tests import fabrica  # noqa: E402


@pytest.fixture
def banco_limpo():
    """Banco temporário zerado (todas as tabelas recriadas)."""
    from app.data.db import Base, engine, init_db

    Base.metadata.drop_all(bind=engine)
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def planilha_ficticia(tmp_path):
    """Caminho de uma planilha .xlsx fictícia no mesmo formato da real."""
    caminho = tmp_path / "planilha_ficticia.xlsx"
    fabrica.planilha_bruta().to_excel(caminho, sheet_name="Planilha1", index=False)
    return str(caminho)


@pytest.fixture
def banco_com_dados(banco_limpo, planilha_ficticia):
    """Banco temporário carregado a partir da planilha fictícia, pelo
    mesmo caminho da aplicação (ingest.processar_planilha + carregar_no_banco)."""
    from app.data import ingest

    df, mapeamento = ingest.processar_planilha(planilha_ficticia, "Planilha1")
    ingest.carregar_no_banco(df, mapeamento)
    return df, mapeamento


@pytest.fixture
def modelo_falso(monkeypatch):
    """Substitui os modelos treinados (.joblib) por um modelo simples e
    previsível, para testar a aplicação sem depender dos arquivos reais."""
    from app.modeling import predict

    def carregar(horizonte):
        return fabrica.ModeloFalso(horizonte), {"threshold_operacional": 0.5}

    monkeypatch.setattr(predict, "carregar_modelo", carregar)
    return carregar


@pytest.fixture
def cliente(banco_com_dados, modelo_falso):
    """Cliente HTTP da aplicação Flask, com banco fictício e modelo falso."""
    from app.presentation import create_app, routes

    routes._cache.clear()
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
    routes._cache.clear()
