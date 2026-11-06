# Manutenção Preditiva — Aplicação de apoio à decisão (TCC)

![Testes](https://github.com/JaoFraccaroli/tcc-manutencao-preditiva/actions/workflows/testes.yml/badge.svg)

> **Versão pública do código do meu TCC** (Engenharia da Computação). Para proteger a empresa e as pessoas envolvidas, **não estão aqui** a planilha de ordens de serviço, o banco de dados, os modelos treinados nem os gráficos gerados com os dados reais, e os nomes usados na anonimização foram trocados por nomes fictícios. Sem a planilha a aplicação não tem dados para mostrar, mas **todos os testes rodam**, porque usam dados fictícios gerados pelos próprios testes. Ver [Testes automatizados](#testes-automatizados-qa).

Ferramenta descritiva e preditiva que estima a probabilidade de falha
(MC/CP) de cada equipamento nos horizontes de **30, 60 e 90 dias**
(`config.HORIZONTES_DIAS`), a partir
do histórico de ordens de serviço. A aplicação mostra indicadores e
previsões; **não ordena nem recomenda automaticamente** qual equipamento
inspecionar — essa decisão é do usuário (ver `briefing_tecnico_aplicacao_tcc.md`).

**Escopo atual: Monta-cargas.** A fonte de dados é uma planilha dedicada
(`Historico 5 anos Monta Cargas.xlsx`, aba `Planilha1`), com **6
equipamentos** (TAG original `MC01`..`MC06`) e histórico de **2008 a
2025** (~2.900 eventos concluídos após limpeza) — bem mais denso que o
recorte equivalente dentro da planilha genérica original. Ver
[Limitações](#limitações-já-identificadas-para-os-capítulos-56-do-tcc)
para o que muda em relação a um dataset com muitos equipamentos.

## Módulos

| # | Módulo | Código | O que faz |
|---|--------|--------|-----------|
| i | Dados | `app/data/` | Lê a planilha bruta, limpa, anonimiza (TAG → EQ00N) e carga em banco (SQLite/SQL Server) |
| ii | Processamento | `app/processing/` | Linha do tempo por TAG, rótulos `falha_{N}d` por janela deslizante, engenharia de variáveis (`dataset.py`); análise exploratória de texto (`text_analysis.py`) e descritiva — MTBF, tendência, sazonalidade (`exploratory.py`); estimativa de componente por palavra-chave (`component_mapping.py`) — todos independentes da modelagem |
| iii | Modelagem | `app/modeling/` | Comparação de modelos, tuning, otimização de threshold (recall priorizado), avaliação |
| iv | Apresentação | `app/presentation/` | Aplicação Flask + Matplotlib |

## Como rodar

```bash
pip install -r requirements.txt

# 1) Ingestão (lê a planilha em config.RAW_XLSX_PATH, popula o banco)
python scripts/run_ingest.py

# 2) Treino de um modelo por horizonte (config.HORIZONTES_DIAS)
python scripts/run_training.py

# 3) Aplicação web
python run.py   # http://127.0.0.1:5000
```

Reprocessar dados ou retreinar não exige reiniciar o Flask: acesse
`/atualizar` para limpar o cache em memória.

## Testes automatizados (QA)

O projeto tem uma bateria de **testes automatizados com Pytest**, feita no mesmo estilo das aulas de Qualidade de Software: testes dirigidos por dados, valores de fronteira, cenários BDD em português, cobertura de código e teste de mutação.

| Métrica | Resultado |
|---|---|
| Testes | 376 passando + 4 problemas documentados (`xfail`) |
| Cenários BDD (Gherkin em português) | 26, em 3 funcionalidades |
| Cobertura de linhas (`pytest-cov`) | 96% |
| Teste de mutação | 26 de 26 mutantes mortos (100%) |

**Os testes nunca usam a planilha real nem o banco real.** A planilha de teste é gerada pelos próprios testes com dados fictícios (`tests/fabrica.py`), o banco é um SQLite temporário e os modelos de teste são gravados numa pasta temporária.

### O que é testado

| Arquivo | O que prova |
|---|---|
| `test_01_anonimizacao.py` | Nenhum TAG real, nome de técnico/empresa, matrícula ou centro de custo chega ao banco ou às telas; TAG → EQ00N sempre igual; HH perturbado entre 90% e 110% e reproduzível. Casos em [`tests/dados/casos_atribuicao.csv`](tests/dados/casos_atribuicao.csv) |
| `test_02_limpeza_e_classificacao.py` | Só OS concluídas; datas com espaço; data inválida interrompe; escopo de monta-cargas; MC/CP = falha, MP = preventiva; duplicatas; carga idempotente |
| `test_03_rotulos_e_variaveis.py` | Fronteiras da janela (dia 0 não conta, dia 1 e dia N contam, dia N+1 não) nos 3 horizontes; censura à direita; **eventos futuros não mudam as variáveis** (sem vazamento) |
| `test_04_divisao_e_modelagem.py` | Nenhum equipamento no treino e no teste ao mesmo tempo; os mais recentes vão para o teste; threshold com recall ≥ 0,90 e maior precisão; métricas com uma classe só |
| `test_05_data_sugerida.py` | Limiar em 0,49 / 0,50 / 0,51; primeiro horizonte que cruza; sem alerta não inventa data; frase explicativa |
| `test_06_componentes_e_texto.py` | Regras de palavra-chave ("porta" ≠ "exportação", "trava" ≠ "travado"), desempate e categorização do texto. Casos em [`tests/dados/casos_componentes.csv`](tests/dados/casos_componentes.csv) |
| `test_07_analise_exploratoria.py` | MTBF com 0, 1 e 2 falhas e falhas no mesmo dia; tendência; 12 meses na sazonalidade; comparação consolidada |
| `test_08_cadastro.py` | Ano 1899 ✗ / 1900 ✓ / 2100 ✓ / 2101 ✗; capacidade "1,5"; status inválido; TAG e componente repetidos |
| `test_09_interface_web.py` | Todas as páginas abrem; nenhuma página mostra dado identificável; 404; lista não ordenada por risco; cadastros pela tela; datas dd/mm/aaaa |
| `test_10_treinamento.py` | Treino completo com dados fictícios numa pasta temporária, sem tocar nos modelos reais |
| `bdd/features/*.feature` | Cenários em português: anonimização, rótulo de falha e data de alerta |
| `qualidade/mutacao.py` | 26 mutantes (ex.: trocar `>=` por `>` no limiar, mudar a janela do rótulo, esquecer a censura à direita) |

### Problemas encontrados pelos testes

Ficam marcados com `@pytest.mark.xfail(strict=True)`: o teste documenta o problema e, se alguém corrigir o código, o pytest avisa para atualizar o teste.

| Onde | Problema |
|---|---|
| `ingest._redigir_descricao` | **Defeito:** a remoção do nome do técnico corta tudo depois do último hífen *antes* de redigir o TAG. Com o TAG escrito com hífen, a frase perde o final: "Lâmpada do MC-02 queimada" vira "Lâmpada do MC". O TAG não vaza, mas a descrição fica incompleta |
| `ingest._remover_atribuicao_final` | **Limitação:** uma descrição legítima com hífen ("porta não abre - sensor com defeito") perde a parte depois do hífen |
| `ingest._anonimizar_tag` | **Risco:** o TAG não é padronizado em maiúsculas; "mc01" e "MC01" virariam dois equipamentos diferentes |
| `/equipamentos?page=abc` | **Defeito:** texto no número da página causa erro 500 em vez de voltar para a página 1 |

Também apareceu, pela cobertura, que o caminho alternativo por F2 em `train._escolher_threshold` nunca executa: a curva sempre tem um corte com recall 1,0. Isso não muda nenhum resultado.

### Como rodar os testes

```bash
pip install -r requirements.txt -r requirements-teste.txt

pytest                                   # todos os testes
pytest --cov=app --cov-report=html       # com cobertura (abre htmlcov/index.html)
python qualidade/mutacao.py              # teste de mutação
```

No **VS Code**, os testes aparecem no ícone do frasco (Testing): clique em **Run Tests**.

## Configuração

Tudo em `config.py`, sobrescrevível por variável de ambiente:

- `RAW_XLSX_PATH` — caminho da planilha bruta (nunca copiada para o repo).
  Atualmente aponta para a planilha dedicada de monta-cargas.
- `RAW_XLSX_SHEET` — nome da aba (`Planilha1` na planilha atual).
- `TAG_PREFIXOS_ESCOPO` — prefixos de TAG (original, antes da
  anonimização) a incluir na ingestão; `["MC"]` hoje. Lista vazia/`None`
  = sem filtro (todos os equipamentos da planilha). Redundante com a
  planilha atual (que já só tem monta-cargas), mas serve de trava caso
  a fonte volte a ser genérica.
- `DATABASE_URL` — `sqlite:///...` por padrão. Para SQL Server, algo como
  `mssql+pyodbc://usuario:senha@servidor/banco?driver=ODBC+Driver+17+for+SQL+Server`.
  Nenhum outro módulo depende do dialeto do banco.
- `RANDOM_SEED`, faixa de perturbação, horizontes, janelas de feature.

A planilha bruta pode ter ou não uma coluna `Fim` separada — a
[dedicada de monta-cargas](#escopo-atual-monta-cargas) só tem `Início`
(a ingestão assume `Fim = Início`, evento de um dia só); a genérica
original tinha `Início`/`Fim` distintos e ambos são suportados.

## Anonimização (briefing seção 4)

- TAG real → `EQ001, EQ002, ...`. O mapeamento fica isolado na tabela
  `tag_mapping_interno`, nunca lida pelos módulos de processamento,
  modelagem ou apresentação.
- `Centro de Custo`, `NºMatrícula`, `Ofic Coordenadora/Executante` são
  descartados na ingestão (não são features e podem ser identificáveis).
- `HH Apropriado` é perturbado (fator uniforme 0.9–1.1, seed fixa) antes
  de ser persistido — é o único valor armazenado, usado tanto no
  treinamento quanto na exibição.
- `Descrição Tarefa` (texto livre da OS, usado na análise exploratória de
  texto abaixo) passa por duas redações antes de ser persistido, em
  `app/data/ingest.py::_redigir_descricao`:
  - **TAG real embutido na frase** (ex.: "Lâmpada do MC01 queimada") →
    substituído por `[equip]`, mesma lógica da coluna estruturada. O
    casamento é flexível a espaço/hífen entre a parte alfabética e a
    numérica (`_padrao_tag_flexivel`) — achado real ao inspecionar
    exemplos na tela `/componentes/exemplos` (briefing complementar 3,
    item 9): 37 descrições tinham o TAG escrito como "MC 02"/"MC-02" em
    vez de "MC02", que o casamento por igualdade exata não pegava.
    Corrigido e reprocessado; verificado que zero variantes sobrevivem.
  - **Atribuição de nome de técnico ou de empresa terceirizada**: a
    inspeção da planilha real mostrou que mais da metade das descrições
    de falha termina com o nome de quem atendeu a OS (padrões "- Fulano",
    "(ok Fulano)", ou o nome solto no final, às vezes com o nome de uma
    empresa terceirizada) — removido por camadas (parênteses finais →
    padrão "ok Fulano" → segmento após o último hífen → lista fechada
    dos nomes específicos que sobreviveram às camadas anteriores nesta
    base, `NOMES_ATRIBUICAO_CONHECIDOS`). Verificado que zero ocorrências
    dos nomes identificados sobrevivem no banco após a redação.

## Resultados (conjunto de teste, planilha de monta-cargas)

Todos os três horizontes selecionaram **Logistic Regression** (entre
Logistic Regression / Random Forest / Gradient Boosting) por Average
Precision na validação cruzada. Threshold operacional = threshold que
maximiza precision entre os que atingem recall ≥ 0.90 na validação
(dev/OOF); quando nenhum atinge, cai no fallback F2 (ver
`_escolher_threshold` em `app/modeling/train.py`).

| Horizonte | Threshold operacional | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| 30d | 0.468 | 0.685 | 0.859 | 0.593 | 0.702 | 0.791 | 0.852 |
| 60d | 0.426 | 0.749 | 0.898 | 0.765 | 0.826 | 0.811 | 0.936 |
| 90d | 0.423 | 0.833 | 0.920 | **0.881** | 0.900 | 0.823 | 0.964 |

(Para referência, no threshold padrão de 0.50 o recall cai bastante em
troca de mais precision — ex.: 90d vai de 0.881 para 0.781 de recall, mas
ganha só 0.027 de precision. Os valores completos, incluindo threshold
0.50, estão em `models/modelo_falha_{30,60,90}d.meta.json`.)

**90 dias saiu com a melhor previsão dos três, de forma consistente em
todas as métricas** — e é o único horizonte que chega perto da meta de
recall ≥ 0.90 (0.881). Isso não é acaso: janela maior de observação =
mais chance de conter pelo menos uma falha dentro dela = fronteira mais
fácil de separar estatisticamente. O inverso acontece em 30d — janela
curta, há menos chance de a janela conter uma falha (a proporção de
falhas na janela é menor), então fica mais difícil manter recall alto
sem sacrificar bastante a precisão. Vale registrar
essa leitura no capítulo 5 do TCC como uma observação sobre a
granularidade alcançável com o volume de dados disponível (6
equipamentos), não como "o modelo de 90d é objetivamente melhor" sem
mais contexto.

## Análise exploratória e descritiva (briefing complementar)

Camada descritiva/exploratória adicional, **separada da modelagem
preditiva** (nada aqui é feature de nenhum modelo) — complementa a
probabilidade de falha com uma leitura de *por que* e *quando* as falhas
acontecem, usando só colunas que já existem na planilha real.

- **Categorização das descrições de falha** (`/analise-textual`,
  `app/processing/text_analysis.py`): em vez de categorias definidas de
  antemão, o texto livre da OS (`Descrição Tarefa`, já anonimizado — ver
  seção acima) é normalizado e as categorias emergem dos bigramas/palavras
  mais frequentes entre as 1.990 descrições de falha (MC/CP). Resultado
  atual: **34,8% ficam em "outro / não identificado"** (nenhum padrão
  frequente o suficiente); entre as identificadas, as maiores são "porta"
  (19,1%), "parou funcionar" (10,6%), "piso" (7,7%) e "não funcionando"
  (4,7%) — ver tabela completa na rota. A taxa de não identificação é
  esperada de uma categorização derivada de texto livre e não de um campo
  estruturado; documentar como limitação, não como falha da análise.
- **MTBF por equipamento** (`/analise-exploratoria`,
  `exploratory.mtbf_por_equipamento`): tempo médio entre falhas
  consecutivas (MC/CP) por TAG, de 7,8 dias (EQ003) a 175,8 dias (EQ005)
  — conecta diretamente com a fundamentação teórica do capítulo 2 do TCC
  (Tsarouhas, 2020; Alsyouf et al., 2016).
- **Tendência do tempo de reparo** (`exploratory.tendencia_tempo_reparo`):
  regressão linear simples (ano × HH Apropriado) por equipamento. Nos
  dados atuais, as inclinações são pequenas em todos os 6 equipamentos
  (entre -0,02 e +0,04 h/ano) — nenhum mostra sinal claro de degradação
  crescente no tempo de reparo até o momento.
- **Sazonalidade** (`exploratory.sazonalidade_mensal`): contagem de
  falhas por mês do calendário, somando todos os anos e equipamentos.
  Distribuição relativamente estável ao longo do ano (135 a 187 falhas
  por mês), sem concentração sazonal marcante.
- **Comparação consolidada**: tabela com proporção de eventos corretivos (falhas ÷ eventos, ex.: 218/398 = 54,8%; não é taxa por unidade de tempo), MTBF (intervalo médio de calendário entre falhas) e tempo
  médio de reparo lado a lado por equipamento, em `/analise-exploratoria`
  — fecha bem uma seção de resultados descritivos no capítulo 5 do TCC.

## Página unificada do equipamento e cadastro ampliado (briefing complementar 4)

1. **Consolidação no detalhe do equipamento** (`/equipamentos/<tag>`):
   passou a reunir, num lugar só, o que antes ficava espalhado em telas
   diferentes:
   - **Dados cadastrais da máquina** (seção abaixo) no topo, com link
     direto para editar.
   - **Componentes estimados só deste equipamento** — mesma distribuição
     de `/componentes`, mas filtrada por TAG, com cada categoria
     expansível (`<details>`) mostrando até 5 descrições reais que
     geraram aquela classificação.
   - **Data sugerida com frase dinâmica**: em vez de só mostrar a data,
     explica o motivo — ex.: *"A probabilidade estimada de falha
     ultrapassa o limiar de 50% no horizonte de 30 dias (81%)."*
     (`routes._frase_data_sugerida`, montada a partir do horizonte/
     probabilidade já calculados, não é texto gerado por modelo).
   - `/maquinas` e `/componentes` continuam existindo como telas de
     listagem/cadastro geral.
2. **Cadastro de máquina ampliado** (`Equipamento`, `app/data/cadastro.py`):
   fabricante, modelo, capacidade de carga (kg), ano de instalação,
   setor/localização, status (Ativo/Em manutenção/Inativo, obrigatório),
   responsável técnico, observações técnicas. **Todos cadastro manual** —
   preenchidos pelo usuário no formulário, nunca inferidos da planilha
   nem usados como feature do modelo; documentado no docstring do modelo
   e aqui. Campo em branco aparece como "não informado", nunca um valor
   chutado. Inclui edição (`/maquinas/<tag>/editar`) além da inclusão já
   prevista — sem editar, o cadastro ampliado não serviria para os 6
   equipamentos reais que já existem.

## Ajustes de interface e usabilidade (briefing complementar 3)

Ajustes de exibição/formatação/cadastro sobre a aplicação já construída —
nenhum modelo foi retreinado, nenhuma metodologia mudou:

1. **Probabilidade em percentual nos gráficos** (`plots.histograma_probabilidades`):
   eixo em 0–100% em vez de 0–1; o valor interno (decimal) usado pelo
   modelo e pelo limiar de alerta continua igual.
2. **Tipo de manutenção por extenso** (`app/processing/tipos_manutencao.py`):
   só MC/CP/MP têm tradução confirmada (são os únicos com significado
   definido no TCC); os demais códigos (IN, ME, EV, IP, CA, LB, PT, ...)
   mostram "Outro (não classificado) — `<código>`" em vez de uma tradução
   não verificada.
3. **"Eventos por área" removido** da visão geral (`plots.contagem_por_area`
   e a rota correspondente foram excluídos).
4. **Datas em `dd/mm/aaaa`** em toda a aplicação, via filtro Jinja único
   `|data_br` (`app/presentation/__init__.py`) — aceita `date`,
   `datetime`, `pd.Timestamp` e string ISO, propagando o formato para
   todos os templates de uma vez.
5. **Área/Sist. Oper. removidos da listagem `/equipamentos`** (continuam
   no banco, como feature do modelo, e visíveis no detalhe de cada
   equipamento).
6. **Inserção de máquinas e componentes pela aplicação** (`app/data/cadastro.py`,
   formulários em `/maquinas` e `/componentes`): valida campo obrigatório
   (TAG / nome) e unicidade antes de gravar, com mensagem de erro exibida
   na tela (flash) em vez de gravar duplicado. Cadastrar um componente
   pela tela adiciona ao catálogo exibido, mas **não** atualiza sozinho
   `TERMOS_COMPONENTE` nem reclassifica eventos — isso continua exigindo
   editar `app/processing/component_mapping.py` e reingerir (documentado
   na própria tela).
7. **Glossário das métricas** em `/modelos` (Acurácia, Precisão, Recall,
   F1, ROC-AUC, PR-AUC) — card dedicado + tooltip em cada cabeçalho da
   tabela de métricas.
8. **MTBF explicado por extenso** (Mean Time Between Failures) onde
   aparece em `/analise-exploratoria`.
9. **Rastreabilidade componente ↔ descrição** (`/componentes/exemplos?nome=...`,
   `component_mapping.exemplos_por_componente`): cada categoria na tabela
   de distribuição agora é um link que mostra até 20 descrições reais
   (mais recentes primeiro, já anonimizadas) que geraram aquela
   classificação — em vez de deixar o percentual como número solto sem
   rastreabilidade. **Foi inspecionando esses exemplos que apareceu o
   achado do item 1 da seção "Anonimização" acima** (variante "MC 02" do
   TAG não redigida) — o próprio ajuste de transparência pediu pela
   orientadora acabou funcionando como uma auditoria adicional da
   anonimização.

## Cadastro de máquinas e componentes (briefing complementar 2)

- **Cadastro de máquinas** (`/maquinas`): os 6 equipamentos do escopo,
  reaproveitando a dimensão `Equipamento` já calculada na ingestão
  (primeiro/último evento, total de eventos) — só ganhou uma coluna
  `tipo` ("Monta-carga").
- **Cadastro de componentes** (`/componentes`, tabela `Componente`):
  catálogo de referência TÉCNICA GERAL (fabricante de referência, vida
  útil indicada, unidade, critério de substituição) sobre os 14 tipos de
  componente típicos de um monta-carga — conhecimento de engenharia
  (WEG, SEW-EURODRIVE, SKF, Schneider, Siemens, Omron, Mayr, igus, entre
  outros), não dado específico da empresa. Populado uma vez na ingestão
  a partir de `component_mapping.CATALOGO_COMPONENTES`.
- **Classificação estimada de componente por evento** (coluna
  `componente_estimado` em `EventoManutencao`, calculada só para
  `is_falha=True`): **estimativa por palavra-chave sobre texto livre,
  não um campo medido pelo sistema de origem** — o CMMS não registra
  qual componente falhou. Metodologia validada contra a base real (não
  é uma classificação de IA sem verificação — isso foi tentado e
  descartado antes por não bater com os dados reais):
  - Texto normalizado (minúsculas, sem acento) comparado contra os
    termos de busca de cada componente (`TERMOS_COMPONENTE`).
  - O termo precisa **começar uma palavra** ("porta" casa com "portas",
    mas não com "exportação", nome de linha de produção, nem "comporta").
    Exceção: "trava" só casa como palavra inteira ("trava"/"travas"); flexões
    de "travar" ("travado", "travando") ficam de fora, pois costumam descrever a
    máquina travada e não o componente.
  - Quando uma descrição bate com mais de um componente, fica com o
    **primeiro que casar, na ordem definida em `TERMOS_COMPONENTE`**
    (critério de desempate simples, documentado no código).
  - Resultado real na base atual (1.990 eventos de falha): **69,2% em
    "não identificado"**, **24,0% em "Portas / travas"** (categoria
    isoladamente dominante), e o restante distribuído entre Sensores/fim
    de curso (3,1%), Trilhos/guias (1,4%), Painel elétrico (1,3%) e as
    demais categorias com menos de 0,5% cada. Esses percentuais são
    esperados de uma estimativa sobre texto livre — documentar como tal,
    nunca como taxonomia oficial da empresa.
- **Data sugerida de alerta** (`/equipamentos`, `/equipamentos/<tag>`,
  `predict.data_sugerida_alerta`): **derivada** das probabilidades já
  calculadas pelos modelos — não é uma nova estimativa, nenhum
  retreinamento envolvido. Para cada equipamento, é o primeiro horizonte
  (30, 60 ou 90 dias, em `config.HORIZONTES_DIAS`) em que a probabilidade
  estimada cruza `config.LIMIAR_ALERTA` (0.5 por padrão); se nenhum
  horizonte cruzar, mostra "sem alerta nos horizontes avaliados" em vez
  de forçar uma data. Representa quando o risco estimado cruza o limiar
  escolhido, não uma previsão determinística de quando a falha vai
  ocorrer.

## Limitações já identificadas (para os capítulos 5/6 do TCC)

- **Só 6 equipamentos no escopo (monta-cargas)** — isso é o fator que
  mais molda a metodologia de validação, bem diferente de um dataset com
  centenas de TAGs:
  - `StratifiedGroupKFold(5)` não é viável (exige ~5+ grupos por fold
    para estratificar de verdade). `app/modeling/train.py` troca
    automaticamente para **LeaveOneGroupOut** quando há menos de
    `GRUPOS_MIN_PARA_KFOLD` (8) equipamentos no treino — um fold por TAG,
    o máximo de validação agrupada possível com o volume disponível.
  - A calibração de probabilidade (`CalibratedClassifierCV`) só é
    aplicada se o pior fold de validação tiver exemplos suficientes da
    classe minoritária (`CALIBRACAO_CV`, hoje 3); caso contrário é
    desativada para aquele horizonte e a saída fica sem recalibração
    (documentado no `meta.json` de cada modelo, campo
    `calibracao_probabilidade`).
  - Se algum fold tiver o treino de uma única classe (um TAG que só
    "sempre falha" ou "nunca falha" na janela), o ajuste do modelo pode
    falhar para esse fold especificamente — `_oof_proba_seguro` cobre
    isso usando a taxa de falha do próprio fold como previsão, em vez de
    derrubar a execução inteira (fica registrado em
    `folds_com_fallback_oof` no `meta.json`).
  - **A calibração pode não transferir bem para o conjunto de teste**:
    ela é ajustada nos TAGs de treino, mas o teste usa TAGs
    *diferentes*, e a taxa de falha varia bastante entre monta-cargas
    (de ~15% a ~90%, dependendo do equipamento). O gráfico de calibração
    de cada horizonte mostra isso — vale conferir antes de tratar as
    probabilidades como definitivas.
  - **`área`/`sist_oper` aparecem entre as features mais importantes**
    (permutation importance) parcialmente porque, com só 6 equipamentos,
    essas categóricas acabam funcionando como um proxy da identidade do
    próprio TAG — o modelo pode estar aprendendo "esse padrão de área
    tende a falhar mais" mais do que um padrão genuinamente
    generalizável para um monta-cargas novo.
- **Quanto maior o horizonte, menos janelas completas disponíveis** (60d
  tem menos que 30d; 90d tem menos que 60d): uma data de referência só é
  usada se `data_referencia + horizonte` couber dentro do período coberto
  pela base, para não tratar ausência de falha registrada como "não
  falha" quando na verdade é falta de dados futuros (censura à direita).
  Com a planilha atual (histórico de 2008 a 2025) essa perda é pequena em
  termos relativos — mas ainda existe e é maior no horizonte de 90d.
- **Tipos de manutenção não classificados** (na planilha atual: IN, ME,
  IP, EV, CA) não entram na definição de falha nem de não-falha; ficam no
  histórico geral (contam para "eventos no histórico"), mas não geram
  data de referência própria.
- **Divisão treino/teste (por equipamento, sem corte por data)**: como
  cada equipamento tem atividade espalhada por quase 17 anos, não existe
  corte cronológico. O teste são equipamentos inteiros fora do treino:
  desenvolvimento = EQ001, EQ002, EQ003 e EQ005 (validação cruzada
  `LeaveOneGroupOut` por TAG); teste = EQ004 e EQ006. Em
  `app/modeling/split.py`, os TAGs são ordenados pela mediana de suas
  datas de referência e os de mediana mais recente vão para o teste (até
  ~20% das linhas) — sem overlap de TAG e sem descartar linhas; com só 6
  TAGs isso é grosseiro (o teste vira 2 equipamentos inteiros, não uma
  fatia proporcional). O threshold operacional é escolhido só nas
  predições out-of-fold do desenvolvimento (`_escolher_threshold`); o
  teste é usado depois, apenas para reportar.
- Probabilidades no `/equipamentos` são calculadas na data mais recente
  da base para todos os equipamentos ao mesmo tempo — é uma "foto" no fim
  do período coberto, não uma previsão em tempo real.
- **Categorização textual das falhas (34,8% "não identificado")**: as
  categorias vêm de padrões de texto livre descobertos automaticamente
  (bigramas/palavras mais frequentes), não de um campo estruturado do
  sistema de origem — tratar a tabela de `/analise-textual` como leitura
  indicativa da natureza das falhas, não como uma taxonomia exaustiva ou
  oficial da empresa.
- **Classificação de componente estimada (69,2% "não identificado")**:
  mesma limitação de fundo da categorização textual acima — depende de
  texto livre, sem padronização. "Portas / travas" dominar sozinha
  (24,0%) pode refletir tanto uma causa raiz real e recorrente quanto um
  viés do vocabulário mais comum nas descrições (a palavra "porta"
  aparece em contextos variados: trava, dobradiça, sensor de porta,
  intertravamento) — não foi feita uma subdivisão mais fina por falta de
  detalhe suficiente no texto para distinguir esses casos com confiança.

## O que foi deliberadamente omitido (briefing seção 10)

- Nenhum ranking automático de "equipamentos prioritários para
  inspeção". A tabela `/equipamentos` é ordenável pelo usuário, mas
  nunca ordenada por risco por padrão.
- Nenhum código do `Ex2.py` (simulação pygame, fora do escopo).
