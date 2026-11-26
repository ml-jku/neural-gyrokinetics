#!/usr/bin/env bash
# The stored rows (neugk/gyrosplat/results/rows/gkw_rows.jsonl) that this package regenerates: the recipe,
# the plain splats, the Fourier pair, four wavelets (Haar, db4, db8, bior4.4), the Haar code
# with the refinement, PCA, ZFP and SZ3 at both budgets (73,728 and 156,669 stored reals) on
# the 39 evaluation snapshots, and the port of the field-only neural field (the stored rows of
# both neural fields and of the autoencoders come from PINC's own code and
# models; they are in neugk/gyrosplat/results/rows/gkw_rows.jsonl only).  The fitting seeds and the codes at
# equal bytes are in scripts/gyrosplat/reproduce_experiments.sh.  One GPU; about twenty minutes per
# snapshot.
# Nine snapshots (iteration_5:20/32, 26:32, 50:32, 204:32, 214:20, 280:20/25/60) exist in two
# preprocessed versions with different content; the rows use the second preprocessing, so
# <data_root> must resolve those nine to it.
#   scripts/gyrosplat/reproduce_gkw.sh <data_root> <results_dir> [<raw_root>]
# CELLS="traj:t ..." restricts the run to some snapshots; GYROSPLAT overrides the command
# (e.g. "python -m neugk.gyrosplat").  The host `neugk` package must be importable.
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
declare -A BUDGET=( [302x]=73728 [142x]=156669 )

run () {  # run <output> <command...>: skip finished rows, keep going after a failure
  local out=$1; shift
  [ -f "$out" ] && return 0
  "$@" --out "$out" || echo "FAILED $out"
}

for cell in $CELLS; do
  traj=${cell%%:*}; t=${cell##*:}; tag=${traj}_t$t
  DATA="data.root=$ROOT data.trajectory=$traj data.timestep=$t $EXTRA"
  for tier in 302x 142x; do
    CFG="--config configs/gyrosplat/recipe/gkw_$tier.yaml"
    B="--budget ${BUDGET[$tier]}"
    run "$OUT/gyrosplat_${tier}_$tag.json" $GG fit --dataset gkw $CFG $DATA
    run "$OUT/splats_${tier}_$tag.json" $GG baseline --dataset gkw --method splats $B $CFG $DATA
    run "$OUT/fourier_gn_${tier}_$tag.json" $GG baseline --dataset gkw --method fourier_gn $B $CFG $DATA
    run "$OUT/wavelet_gn_${tier}_$tag.json" $GG baseline --dataset gkw --method wavelet_gn $B $CFG $DATA baselines.wavelet=haar
    for w in db4 db8 bior4.4; do
      run "$OUT/wavelet_${w}_${tier}_$tag.json" $GG baseline --dataset gkw --method wavelet $B $CFG $DATA baselines.wavelet=$w
    done
    for m in pca zfp sz3 neural_field; do
      run "$OUT/${m}_${tier}_$tag.json" $GG baseline --dataset gkw --method "$m" $B $CFG $DATA
    done
  done
done
echo "sweep done $(date -u)"
