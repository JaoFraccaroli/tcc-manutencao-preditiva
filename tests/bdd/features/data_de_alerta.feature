# language: pt
Funcionalidade: Data sugerida de alerta
  Para planejar a manutenção preditiva
  Como usuário da aplicação
  Quero ver a partir de quando o risco estimado passa do limiar de 50%

  Esquema do Cenário: Primeiro horizonte que cruza o limiar
    Dado as probabilidades de falha <p30> em 30 dias, <p60> em 60 dias e <p90> em 90 dias
    Quando a data sugerida é calculada a partir de "01/01/2025"
    Então o alerta deve ser "<alerta>"

    Exemplos:
      | p30  | p60  | p90  | alerta               |
      | 0.81 | 0.90 | 0.95 | em 31/01/2025        |
      | 0.20 | 0.55 | 0.90 | em 02/03/2025        |
      | 0.10 | 0.20 | 0.50 | em 01/04/2025        |
      | 0.49 | 0.49 | 0.49 | sem alerta           |
      | 0.50 | 0.10 | 0.10 | em 31/01/2025        |

  Cenário: A frase explica o motivo da data
    Dado as probabilidades de falha 0.81 em 30 dias, 0.90 em 60 dias e 0.95 em 90 dias
    Quando a data sugerida é calculada a partir de "01/01/2025"
    Então a frase deve ser "A probabilidade estimada de falha ultrapassa o limiar de 50% no horizonte de 30 dias (81%)."
