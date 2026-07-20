# Changelog

## v1.1.0

### Adicionado
- `actigraphy_replication.py` — replicação em segundo domínio (seções 7.9 e 7.10 do
  manuscrito). Reproduz a fração de sinal circadiana em actigrafia clínica (n = 162,
  quatro populações) e o teste de robustez ao número de ciclos observados.
- `rodar_grades_500.sh` — executor retomável e paralelo das grades de sigma_b a 500
  réplicas.

### Corrigido
- **`registered_test_power.py`: `R_X_REAL` e `R_Y_REAL` estavam codificados com os
  valores do estimador AR(1), que o manuscrito rejeita por enviesado (0,58 e 0,41).
  Corrigidos para os valores do estimador espectral validado (0,469 e 0,323).**
  Esta correção altera a tabela de alocação de orçamento do manuscrito; a conclusão
  qualitativa é preservada e reforçada.
- `wearable_fusion.py` — reescrito. A análise de componentes agora reporta a
  discordância entre quatro pré-processamentos defensáveis, em vez de escolher um.
  Acrescentada auditoria do viés de interpolação, executável sem dados
  (`--audit-only`).

## v1.0.0
- Versão inicial.

## v1.1.1

### Adicionado
- `budget_allocation.py` — reproduz as Tabelas 9 (sensibilidade da suposição
  FSS ≈ confiabilidade; analítica) e 17 (alocação de esforço de pesquisa; simulação).
  Ambas eram computadas ad hoc e não tinham script correspondente. A Tabela 17
  sustenta a afirmação prática central do manuscrito.

### Nota de reprodutibilidade
- Os valores da Tabela 17 no manuscrito correspondem exatamente à saída de
  `budget_allocation.py --table 17` com semente 2026 e 60 réplicas. São estimativas
  de Monte Carlo, sujeitas a variação de 0,02 a 0,05 nesse número de réplicas.
