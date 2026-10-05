"""One TORAX step with gyroflow, dumping latents plus paired decoded 5D snapshots.

Needs a TORAX config module exposing GYROFLOW_TRANSPORT, RHO_MATCH and
config_with_transport; point TORAX_CONFIG_DIR at the directory holding it.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.environ.get("TORAX_CONFIG_DIR")
if not CFG_DIR:
    raise SystemExit("set TORAX_CONFIG_DIR to the directory holding basic_itg_config.py")
sys.path.insert(0, CFG_DIR)
OUT = os.environ.get("LATENT_PACK_DIR", os.path.join(os.getcwd(), "latent_pack"))

import torax  # noqa
f = os.environ.get("XLA_FLAGS", "").replace("--xla_cpu_opt_preset=FAST_COMPILE", "").strip()
os.environ["XLA_FLAGS"] = f
if not f:
    os.environ.pop("XLA_FLAGS", None)
import basic_itg_config as cfg

transport = dict(cfg.GYROFLOW_TRANSPORT)
transport.update({
    "latent_dump_dir": os.path.join(OUT, "npz"),
    "latent_dump_max_calls": int(os.environ.get("MAX_CALLS", "0")),
    "latent_dump_decoded_max_calls": int(os.environ.get("DEC_CALLS", "3")),
    "latent_dump_decoded": int(os.environ.get("N_DECODED", "2")),
    "diagnostics_path": os.path.join(OUT, "diagnostics.jsonl"),
})
config_dict = cfg.config_with_transport(transport, case="iter")
config_dict["numerics"]["t_final"] = float(os.environ.get("T_FINAL", "0.2"))
torax_config = torax.ToraxConfig.from_dict(config_dict)
tree, _ = torax.run_simulation(torax_config)
tree.to_netcdf(os.path.join(OUT, "gyroflow_run.nc"))
meta = {
    "case": "iter",
    "n_samples": transport["n_samples"],
    "sampler_steps": transport["sampler_steps"],
    "latent_dump_decoded": transport["latent_dump_decoded"],
    "rho_match": list(cfg.RHO_MATCH),
    "ae_checkpoint": transport["ae_checkpoint_path"],
    "dit_checkpoint": transport["dit_checkpoint_path"],
    "df_stats": transport["df_stats_path"],
}
with open(os.path.join(OUT, "meta.json"), "w") as fh:
    json.dump(meta, fh, indent=2)
print("pack written to", OUT)
