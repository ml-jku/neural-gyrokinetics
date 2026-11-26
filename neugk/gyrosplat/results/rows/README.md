# The rows behind the paper's tables

`gkw_rows.jsonl`: one JSON object per line, one line per method (or ablation stage), budget
and evaluation snapshot; 2,106 lines over the 39 snapshots.  The fields are documented in
`neugk/gyrosplat/eval/paper/rows.py`, which also converts this package's own result files into the
same format.  The notebooks build the ablation tables from this file and, together with
`neugk/gyrosplat/results/experiments/` (the fitting seeds and the codes at equal bytes), the comparison
tables.

| rows                                   | count | source                                                  |
|----------------------------------------|------:|---------------------------------------------------------|
| methods at both budgets (single fits)  |  1,014 | 13 methods at 302x, 10 at 142x, 39 snapshots each      |
| the refined codes and db4 / db8        |    312 | Fourier and Haar + refinement, db4, db8, both budgets   |
| the ablation ladder at 302x            |    780 | 23 rows (recipe, stage) plus the deployed stages at 142x  |

The rows of GyroSplat, the plain splats, the Fourier and Haar codes, the wavelets, PCA, ZFP
and SZ3 were produced with this package's code on one H100 per snapshot (the configurations
of `scripts/gyrosplat/reproduce_gkw.sh` and `scripts/gyrosplat/reproduce_ablation.sh`).  The neural fields (field
objective and physics stage) and the autoencoder, VAE and VQ-VAE rows come from the code and
the trained models of PINC and were scored with the same functions.  The `source` field is the identifier of the run
that produced a row.
