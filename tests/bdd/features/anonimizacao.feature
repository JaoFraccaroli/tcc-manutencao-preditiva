# language: pt
Funcionalidade: Anonimização das ordens de serviço
  Para cumprir a LGPD
  Como autor do TCC
  Quero que nenhum dado que identifique pessoas ou equipamentos chegue ao banco

  Esquema do Cenário: Nome de quem atendeu a OS é removido da descrição
    Dado a descrição "<original>" do equipamento "MC01"
    Quando a descrição é anonimizada
    Então a descrição gravada deve ser "<gravada>"

    Exemplos:
      | original                             | gravada                  |
      | porta não abre - Zebedeu             | porta não abre           |
      | sensor trocado (ok Zebedeu)          | sensor trocado           |
      | ajustada a porta ok Zebedeu          | ajustada a porta         |
      | motor com ruído - Manutec Ficticia   | motor com ruído          |
      | porta travada                        | porta travada            |

  Esquema do Cenário: TAG real escrito no texto vira um marcador
    Dado a descrição "<original>" do equipamento "MC01"
    Quando a descrição é anonimizada
    Então a descrição gravada deve ser "<gravada>"

    Exemplos:
      | original                  | gravada                   |
      | Lâmpada do MC01 queimada  | Lâmpada do [equip] queimada |
      | Lâmpada do mc01 queimada  | Lâmpada do [equip] queimada |
      | Lâmpada do MC 01 queimada | Lâmpada do [equip] queimada |
      | peça MC010 trocada        | peça MC010 trocada        |

  Cenário: A planilha inteira é processada sem vazar dados
    Dado uma planilha fictícia com TAGs reais e nomes de técnicos
    Quando a planilha é processada
    Então nenhum TAG real aparece no resultado
    E nenhum nome de técnico aparece no resultado
    E os equipamentos ficam com os códigos "EQ001, EQ002, EQ003"
    E as horas apropriadas ficam entre 90% e 110% das originais
