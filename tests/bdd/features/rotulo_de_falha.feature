# language: pt
Funcionalidade: Rótulo "houve falha nos próximos N dias"
  Para treinar o modelo sem vazamento de informação
  Como autor do TCC
  Quero que o rótulo considere só as falhas dentro da janela futura

  Esquema do Cenário: Fronteiras da janela de previsão
    Dado uma data de referência em "01/01/2024"
    E uma falha <dias> dias depois da referência
    Quando o rótulo de <horizonte> dias é calculado
    Então o rótulo deve ser "<rotulo>"

    Exemplos: Horizonte de 30 dias
      | horizonte | dias | rotulo   |
      | 30        | -1   | sem falha |
      | 30        | 0    | sem falha |
      | 30        | 1    | com falha |
      | 30        | 30   | com falha |
      | 30        | 31   | sem falha |

    Exemplos: Horizonte de 90 dias
      | horizonte | dias | rotulo   |
      | 90        | 0    | sem falha |
      | 90        | 1    | com falha |
      | 90        | 90   | com falha |
      | 90        | 91   | sem falha |

  Cenário: Eventos futuros não mudam as variáveis do passado
    Dado um equipamento com histórico de falhas até a referência
    Quando acontecem novas falhas depois da referência
    Então as variáveis calculadas na referência continuam iguais
