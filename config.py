"""
Configurações centrais da aplicação.

O caminho da planilha bruta e a connection string do banco são lidos de
variáveis de ambiente quando definidas, com um default sensato para
desenvolvimento local. Isso permite trocar SQLite -> SQL Server apenas
mudando DATABASE_URL, sem tocar em código (ver seção 11 do briefing).
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Planilha bruta (nunca é copiada para dentro do repositório).
# Escopo atual: planilha dedicada de monta-cargas (MC01..MC06), já
# restrita a esse equipamento pela própria fonte — sem "Fim" (só
# "Início" + horários no mesmo dia) e aba "Planilha1" em vez de "Sheet1".
RAW_XLSX_PATH = os.environ.get(
    "RAW_XLSX_PATH",
    r"C:\Codigo TCC\Historico 5 anos Monta Cargas.xlsx",
)
RAW_XLSX_SHEET = "Planilha1"

# Banco de dados. Default: SQLite local em instance/ (git-ignored).
# Para SQL Server, defina algo como:
#   mssql+pyodbc://usuario:senha@servidor/banco?driver=ODBC+Driver+17+for+SQL+Server
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'manutencao.db')}",
)

# Seed fixa para toda e qualquer perturbação/randomização (reprodutibilidade).
RANDOM_SEED = 42

# Fator de perturbação aleatória aplicado a HH Apropriado antes de qualquer
# uso/exposição (briefing seção 4).
PERTURBACAO_MIN = 0.9
PERTURBACAO_MAX = 1.1

# Horizontes de previsão (dias), briefing seção 5.
HORIZONTES_DIAS = [30, 60, 90]

# Passo (em dias) da grade de datas de referência geradas por equipamento
# para a janela deslizante de rótulo (briefing seção 5). Datas de
# referência vão do primeiro evento do TAG até (data_máxima_da_base -
# horizonte), para nunca avaliar uma janela de observação incompleta
# (censura à direita) — briefing seção 7, limitação do horizonte de 90d.
REFERENCE_STEP_DAYS = 7

# Janelas (dias) para a feature de frequência de falhas recentes
# (briefing seção 6).
JANELAS_FREQ_FALHAS = [90, 180]

# Janela (dias) para contagem de intervenções preventivas recentes
# (briefing seção 6).
JANELA_PREVENTIVAS = 90

# Escopo de equipamentos (briefing "Monta-cargas"): prefixos do TAG
# ORIGINAL (antes da anonimização) a incluir na ingestão. Lista vazia/None
# = inclui todos os equipamentos (comportamento genérico original).
#
# ATENÇÃO — coincidência de sigla, NÃO confundir com TIPOS_FALHA abaixo:
# "MC" aqui é prefixo de TAG de equipamento (monta-cargas: MC01..MC06).
# "MC" em TIPOS_FALHA é o código de Tipo de Manutenção (Corretiva). São
# duas colunas diferentes da planilha (TAG x Tipo de Manutenção) que só
# por acaso usam as mesmas duas letras.
TAG_PREFIXOS_ESCOPO = ["MC"]

# Tipos de manutenção considerados falha (briefing seção 3.2).
TIPOS_FALHA = ["MC", "CP"]

# Tipos que não são falha, mas são sinal relevante para features (MP = preventiva).
TIPO_PREVENTIVA = "MP"

# Tipo de equipamento exibido no cadastro de máquinas (briefing
# complementar 2). Só rótulo de exibição — não afeta filtro/ingestão
# (isso é `TAG_PREFIXOS_ESCOPO`).
TIPO_EQUIPAMENTO_ESCOPO = "Monta-carga"

# Limiar de probabilidade usado para derivar a "data sugerida" de alerta
# a partir das probabilidades já calculadas pelos modelos (briefing
# complementar 2, seção 4). NÃO é usado no treino — é só um corte
# aplicado depois, na apresentação, sobre uma saída de modelo já pronta.
LIMIAR_ALERTA = 0.5

# Chave usada só para mensagens flash (validação de formulário — briefing
# complementar 3, item 6). Aplicação local, sem autenticação de usuário;
# não protege nada sensível, então uma chave fixa é suficiente aqui.
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-tcc-manutencao-preditiva")
