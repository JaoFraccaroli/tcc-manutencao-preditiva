"""
Engine e sessão do SQLAlchemy.

Isolado num módulo próprio para que trocar SQLite -> SQL Server seja apenas
uma questão de mudar DATABASE_URL (config.py) — nenhum outro módulo depende
do dialeto do banco.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

import config

# `connect_args` com check_same_thread só é necessário/aceito para SQLite.
_connect_args = {}
if config.DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(config.DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_session():
    return SessionLocal()


def init_db():
    """Cria todas as tabelas que ainda não existem."""
    import app.data.models  # noqa: F401  garante que os modelos foram registrados
    Base.metadata.create_all(bind=engine)
