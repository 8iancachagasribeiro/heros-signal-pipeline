#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Roda as grades de sigma_b restantes a 500 replicas.
#
# COMO USAR:
#   cd <pasta do repro>
#   bash rodar_grades_500.sh
#
# Roda em segundo plano e pode ser interrompido: cada chunk salva seu proprio
# CSV, entao ao reexecutar ele PULA os chunks ja concluidos.
#
# TEMPO: ~1h por sigma_b em 8 nucleos; ~3h no total. Deixe rodando e va fazer
# outra coisa. O progresso fica em progresso_<sigma>.log
# ---------------------------------------------------------------------------
set -u

SIGMAS="0.05 0.15 0.20"     # 0.10 ja foi rodado
NCHUNKS=8                    # ajuste ao numero de nucleos da sua maquina
OUT="$(pwd)"

echo "Grades a rodar: $SIGMAS  |  $NCHUNKS chunks cada  |  saida: $OUT"
echo

for SB in $SIGMAS; do
  echo "=== sigma_b = $SB ==="
  for CH in $(seq 0 $((NCHUNKS-1))); do
    ALVO="$OUT/h4_500_sb${SB}_c${CH}.csv"
    if [ -f "$ALVO" ]; then
      echo "  chunk $CH ja existe, pulando"
      continue
    fi
    python run_h4_500.py "$SB" "$CH" "$NCHUNKS" >> "progresso_${SB}.log" 2>&1 &
  done
  wait                      # espera todos os chunks deste sigma_b
  echo "  sigma_b=$SB concluido"
done

echo
echo "=== consolidando ==="
python - <<'PYEOF'
import glob, pandas as pd, os
arqs = sorted(glob.glob("h4_500_sb*_c*.csv"))
if not arqs:
    raise SystemExit("nenhum CSV encontrado")
df = pd.concat([pd.read_csv(a) for a in arqs], ignore_index=True)
df = df.sort_values(["sigma_b","obs_per_cycle","reliability"])
df.to_csv("h4_frontier_500reps_COMPLETO.csv", index=False)
print(f"{len(df)} celulas de {df.sigma_b.nunique()} valores de sigma_b")
print("\nRecuperacao mediana por sigma_b:")
print(df.groupby("sigma_b").recovery.describe()[["count","mean","min","max"]])
print("\n>>> CHECAGEM CRITICA: a recuperacao e NAO-MONOTONA na densidade?")
for sb, g in df.groupby("sigma_b"):
    m = g.groupby("obs_per_cycle").recovery.mean()
    monot = m.is_monotonic_increasing
    print(f"  sigma_b={sb}: {'MONOTONA' if monot else 'NAO-MONOTONA (confirma a retirada da fronteira)'}")
print("\nsalvo: h4_frontier_500reps_COMPLETO.csv")
PYEOF
