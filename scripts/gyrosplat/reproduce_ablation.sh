#!/usr/bin/env bash
# The 23 rows of the ablation ladder at the 302x budget on the 39 evaluation snapshots, one
# `python -m neugk.gyrosplat ablate --control` command each (neugk/gyrosplat/eval/ablation.py lists the controls and
# their overrides).  One GPU; about an hour per snapshot.
#   scripts/gyrosplat/reproduce_ablation.sh <data_root> <results_dir> [<raw_root>]
# CELLS="traj:t ..." restricts the run to some snapshots, CONTROLS="noky frac40 ..." to some
# controls; GYROSPLAT overrides the command.  The host `neugk` package must be importable.
set -uo pipefail
ROOT=${1:?data root (the preprocessed snapshots)}
OUT=${2:?results directory}
RAW=${3:-}
GG=${GYROSPLAT:-python -m neugk.gyrosplat}
mkdir -p "$OUT"
EXTRA=""
[ -n "$RAW" ] && EXTRA="data.raw_root=$RAW"
ALL="iteration_13:100 iteration_13:150 iteration_13:200 iteration_100:100 iteration_100:150 iteration_100:200
iteration_200:100 iteration_200:150 iteration_200:200 iteration_115:100 iteration_131:150 iteration_235:100
iteration_454:10 iteration_5:20 iteration_280:25 iteration_280:60 iteration_280:20 iteration_520:5 iteration_520:15
iteration_711:20 iteration_711:45 iteration_877:15 iteration_877:20 iteration_214:20
extra_iteration_0:150 extra_iteration_15:150 extra_iteration_19:150 extra_iteration_23:150 extra_iteration_26:150
extra_iteration_31:150 extra_iteration_36:150 extra_iteration_5:150
iteration_204:32 iteration_26:32 iteration_432:32 iteration_50:32 iteration_5:32 iteration_681:32 iteration_893:32"
CELLS=${CELLS:-$ALL}
CONTROLS=${CONTROLS:-"deployed floor zonal importance_init geometry_warmup potential_fit gabor_init field_solve
gn_without_joint_fit noky random_init random_full no_warmstart flux_during_potential_fit all_gabor all_gabor_full
frac40 frac60 s7_pack fp32match noky_full flux_refinement_all_splats"}
CFG="--config configs/gyrosplat/recipe/gkw_302x.yaml"

for cell in $CELLS; do
  traj=${cell%%:*}; t=${cell##*:}; tag=${traj}_t$t
  DATA="data.root=$ROOT data.trajectory=$traj data.timestep=$t $EXTRA"
  for c in $CONTROLS; do
    out="$OUT/ablation_${c}_302x_$tag.json"
    [ -f "$out" ] && continue
    $GG ablate --dataset gkw --control "$c" $CFG $DATA --out "$out" || echo "FAILED $out"
  done
done
echo "ladder done $(date -u)"
