#!/usr/bin/env bash
# The appendix experiments that this package regenerates, on top of scripts/gyrosplat/reproduce_gkw.sh
# (which writes the seed-0 rows).  Stages, chosen with STAGES="..." (default: all four):
#   seeds    GyroSplat with seeds 1-4 and the plain splats with seeds 1-2, both budgets, 39
#            snapshots (the neural fields' seeds come from PINC's code);
#   bytes    the Fourier and wavelet codes at the payload bytes of GyroSplat's two recipes
#            (294,928 and 626,692 bytes): an int32 position per kept coefficient, so the
#            Fourier codes keep bytes/6 values (two per complex coefficient) and the wavelet
#            codes bytes/8;
#   sweep    the rate-distortion sweep: GyroSplat and the plain splats (seeds 0-2) at six
#            more sizes, the Fourier pair at the same bytes, Haar and bior4.4, PCA, ZFP and
#            SZ3 at the same number of stored values, and the port of the field-only neural
#            field widened to each of the eight budgets (seeds 0-2);
#   damping  the damping and step count of the final Gauss-Newton refinement on 12 snapshots
#            (each point a full fit; the table re-ran only the last stage from saved states,
#            so the joint-fit state differs here by floating-point noise).
# The neural fields of Table 1 (field objective and physics stage, all seeds), the neural field
# + physics of the sweep, the AE/VAE/VQ-VAE latent refinement and the solver restarts need the
# code, the trained models and the solver of other code bases; their rows are in
# neugk/gyrosplat/results/experiments/ only.  Output: one JSON per run under <results_dir>/<stage>/.
#   scripts/gyrosplat/reproduce_experiments.sh <data_root> <results_dir>
# CELLS="traj:t ..." restricts the snapshots; GYROSPLAT overrides the command.
set -uo pipefail
ROOT=${1:?data root (the preprocessed snapshots)}
OUT=${2:?results directory}
GG=${GYROSPLAT:-python -m neugk.gyrosplat}
STAGES=${STAGES:-seeds bytes sweep damping}
ALL="iteration_13:100 iteration_13:150 iteration_13:200 iteration_100:100 iteration_100:150 iteration_100:200
iteration_200:100 iteration_200:150 iteration_200:200 iteration_115:100 iteration_131:150 iteration_235:100
iteration_454:10 iteration_5:20 iteration_280:25 iteration_280:60 iteration_280:20 iteration_520:5 iteration_520:15
iteration_711:20 iteration_711:45 iteration_877:15 iteration_877:20 iteration_214:20
extra_iteration_0:150 extra_iteration_15:150 extra_iteration_19:150 extra_iteration_23:150 extra_iteration_26:150
extra_iteration_31:150 extra_iteration_36:150 extra_iteration_5:150
iteration_204:32 iteration_26:32 iteration_432:32 iteration_50:32 iteration_5:32 iteration_681:32 iteration_893:32"
DAMPING_CELLS="extra_iteration_0:150 extra_iteration_15:150 iteration_100:150 iteration_13:100 iteration_204:32
iteration_280:20 iteration_280:25 iteration_280:60 iteration_50:32 iteration_5:20 iteration_877:20 iteration_893:32"
CELLS=${CELLS:-$ALL}
# GyroSplat's stored values (4 bytes each) at the sweep sizes: n_equiv -> values
declare -A VALUES=( [1312]=22297 [1638]=27845 [2185]=37142 [2913]=49521 [4337]=73732 [6554]=111418 [9216]=156673 [13108]=222833 )
declare -A NOMINAL=( [1312]=1000 [1638]=800 [2185]=600 [2913]=450 [4337]=302 [6554]=200 [9216]=142 [13108]=100 )
declare -A NEQ=( [302x]=4337 [142x]=9216 )
declare -A BUDGET=( [302x]=73728 [142x]=156669 )   # the stored-value budgets of reproduce_gkw.sh

run () {  # run <output> <command...>: skip finished rows, keep going after a failure
  local out=$1; shift
  [ -f "$out" ] && return 0
  mkdir -p "$(dirname "$out")"
  "$@" --out "$out" || echo "FAILED $out"
}

for cell in $CELLS; do
  traj=${cell%%:*}; t=${cell##*:}; tag=${traj}_t$t
  DATA="data.root=$ROOT data.trajectory=$traj data.timestep=$t"
  for tier in 302x 142x; do
    CFG="--config configs/gyrosplat/recipe/gkw_$tier.yaml"
    bytes=$(( 4 * ${VALUES[${NEQ[$tier]}]} ))
    if [[ " $STAGES " == *" seeds "* ]]; then
      for s in 1 2 3 4; do
        run "$OUT/seeds/gyrosplat_${tier}_s${s}_$tag.json" $GG fit --dataset gkw $CFG $DATA seed=$s
      done
      for s in 1 2; do
        run "$OUT/seeds/splats_${tier}_s${s}_$tag.json" $GG baseline --dataset gkw --method splats \
          --budget ${BUDGET[$tier]} $CFG $DATA seed=$s
      done
    fi
    if [[ " $STAGES " == *" bytes "* ]]; then
      for m in fourier fourier_gn; do
        run "$OUT/bytes/${m}_${tier}_$tag.json" $GG baseline --dataset gkw --method $m --budget $(( bytes / 6 )) $CFG $DATA
      done
      for w in haar bior4.4 db4; do
        run "$OUT/bytes/wavelet_${w}_${tier}_$tag.json" $GG baseline --dataset gkw --method wavelet \
          --budget $(( bytes / 8 )) $CFG $DATA baselines.wavelet=$w
      done
      run "$OUT/bytes/wavelet_gn_${tier}_$tag.json" $GG baseline --dataset gkw --method wavelet_gn \
        --budget $(( bytes / 8 )) $CFG $DATA baselines.wavelet=haar
    fi
  done
  if [[ " $STAGES " == *" sweep "* ]]; then
    CFG="--config configs/gyrosplat/recipe/gkw_302x.yaml"
    for n in 1312 1638 2185 2913 4337 6554 9216 13108; do   # the port of the field-only neural field, widened
      for s in 0 1 2; do                                     # to the budget, at every size
        run "$OUT/sweep/neural_field_${NOMINAL[$n]}x_s${s}_$tag.json" $GG baseline --dataset gkw --method neural_field \
          --budget ${VALUES[$n]} $CFG $DATA seed=$s baselines.nf.width=null
      done
    done
    for n in 1312 1638 2185 2913 6554 13108; do              # 302x and 142x: the other stages and reproduce_gkw.sh
      v=${VALUES[$n]}; c=${NOMINAL[$n]}x
      for s in 0 1 2; do
        run "$OUT/sweep/gyrosplat_${c}_s${s}_$tag.json" $GG fit --dataset gkw $CFG $DATA n_equiv=$n seed=$s
        run "$OUT/sweep/splats_${c}_s${s}_$tag.json" $GG baseline --dataset gkw --method splats --budget $v $CFG $DATA seed=$s
      done
      for m in fourier fourier_gn; do
        run "$OUT/sweep/${m}_${c}_$tag.json" $GG baseline --dataset gkw --method $m --budget $(( 4 * v / 6 )) $CFG $DATA
      done
      for w in haar bior4.4; do
        run "$OUT/sweep/wavelet_${w}_${c}_$tag.json" $GG baseline --dataset gkw --method wavelet --budget $v $CFG $DATA \
          baselines.wavelet=$w
      done
      for m in pca zfp sz3; do
        run "$OUT/sweep/${m}_${c}_$tag.json" $GG baseline --dataset gkw --method $m --budget $v $CFG $DATA
      done
    done
  fi
  if [[ " $STAGES " == *" damping "* && " $(echo $DAMPING_CELLS) " == *" $cell "* ]]; then
    CFG="--config configs/gyrosplat/recipe/gkw_302x.yaml"
    for mu in 1e-8 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1; do
      run "$OUT/damping/gyrosplat_mu${mu}_steps8_$tag.json" $GG fit --dataset gkw $CFG $DATA gn.mu=$mu gn.steps=8
    done
    for k in 1 2 4 16; do
      run "$OUT/damping/gyrosplat_mu1e-4_steps${k}_$tag.json" $GG fit --dataset gkw $CFG $DATA gn.mu=1e-4 gn.steps=$k
    done
  fi
done
echo "experiments done $(date -u)"
