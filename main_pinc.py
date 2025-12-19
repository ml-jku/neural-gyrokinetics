from datetime import datetime
import gc
import os
import os.path as osp
import sys
import traceback
import torch.multiprocessing as mp
import torch
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

from neugk.utils import (
    set_seed,
    compress_src,
    find_free_port,
    filter_cli_priority,
    filter_config_subset,
)
from neugk.pinc.ae_run import pinc_ae_runner


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig):
    # torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    set_seed(config.seed)

    print("#" * 88, "\nStarting Cyclone with configs:")
    print(OmegaConf.to_yaml(config))
    print("#" * 88, "\n")

    dict_config = OmegaConf.to_container(config)
    date_and_time = datetime.today().strftime("%Y%m%d_%H%M%S")
    if config.output_path is None:
        dict_config["output_path"] = osp.join("outputs", date_and_time)
    else:
        dict_config["output_path"] = osp.join(dict_config["output_path"], date_and_time)

    # dict_config["checkpoint"] = dict_config["output_path"]

    config = OmegaConf.create(dict_config)

    if not os.path.exists(config.output_path):
        os.makedirs(dict_config["output_path"], exist_ok=True)

    compress_src(dict_config["output_path"])

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus * config.ddp.n_nodes
    else:
        world_size = 1

    # TODO(diff) do we keep the multitask setup with the autoencoder?
    if dict_config["autoencoder"]["loss_weights"] is None:
        dict_config["autoencoder"]["loss_weights"] = {}
    if dict_config["autoencoder"]["extra_loss_weights"] is None:
        dict_config["autoencoder"]["extra_loss_weights"] = {}

    if dict_config["stage"] == "peft":
        # for peft load ae from checkpoint and merge with PEFT config
        if dict_config.get("ae_checkpoint"):
            try:
                import yaml

                ae_checkpoint_path = dict_config["ae_checkpoint"]
                if os.path.isfile(ae_checkpoint_path):
                    ae_config_path = os.path.join(
                        os.path.dirname(ae_checkpoint_path), "config.yaml"
                    )
                else:
                    ae_config_path = os.path.join(ae_checkpoint_path, "config.yaml")

                print(f"Loading autoencoder config from: {ae_config_path}")

                if os.path.exists(ae_config_path):
                    with open(ae_config_path, "r") as f:
                        checkpoint_config = yaml.safe_load(f)

                    # get the ae config from checkpoint
                    checkpoint_ae_config = checkpoint_config.get("autoencoder", {})

                    # provided PEFT-specific config
                    current_peft_config = dict_config["autoencoder"].get("peft", {})
                    current_loss_weights = dict_config["autoencoder"].get(
                        "loss_weights", {}
                    )
                    current_extra_loss_weights = dict_config["autoencoder"].get(
                        "extra_loss_weights", {}
                    )
                    current_loss_scheduler = dict_config["autoencoder"].get(
                        "loss_scheduler", {}
                    )

                    # replace ae config with checkpoint version
                    dict_config["autoencoder"] = checkpoint_ae_config.copy()

                    # PEFT-specific overrides
                    dict_config["autoencoder"]["peft"] = current_peft_config
                    if current_loss_weights:
                        dict_config["autoencoder"][
                            "loss_weights"
                        ] = current_loss_weights
                    if current_extra_loss_weights:
                        dict_config["autoencoder"][
                            "extra_loss_weights"
                        ] = current_extra_loss_weights
                    if current_loss_scheduler:
                        dict_config["autoencoder"][
                            "loss_scheduler"
                        ] = current_loss_scheduler

                    print(f"Merged config from checkpoint with PEFT settings")
                    print(f"Type: {dict_config['autoencoder'].get('name', 'unknown')}")
                else:
                    raise FileNotFoundError(f"Config not found: {ae_config_path}")
            except Exception as e:
                print(f"ERROR loading autoencoder config for PEFT: {e}")
                raise
        else:
            print("ERROR: PEFT stage requires ae_checkpoint to be specified")
            raise ValueError("PEFT stage requires ae_checkpoint parameter")

    try:
        if dict_config["logging"]["run_id"] is None:
            if dict_config["stage"] == "autoencoder":
                name = f"{dict_config['stage']}_{dict_config['autoencoder']['name']}"
            elif dict_config["stage"] == "peft":
                ae_type = dict_config.get("autoencoder", {}).get("name", "unknown")
                name = f"peft_{ae_type}"
            elif dict_config["stage"] == "diffusion":
                sched = dict_config["diffusion"]["scheduler"]["name"]
                model = dict_config["diffusion"]["model"]["name"]
                name = f"{dict_config['stage']}_{sched}_{model}"
            print(f"Generated run name: {name}")
            dict_config["logging"]["run_id"] = f"{name}_{date_and_time}"
            config = OmegaConf.create(dict_config)

        if config.ddp.enable and world_size > 1 and config.ddp.n_nodes == 1:
            if "SLURM_NODELIST" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            else:
                # os.system(f'export MASTER_ADDR=$(scontrol show hostname {os.environ["SLURM_NODELIST"]})')
                # only works for single node so far, adapt above for multinode
                os.environ["MASTER_ADDR"] = os.environ["SLURM_NODELIST"]
            os.environ["MASTER_PORT"] = str(find_free_port())
            if "NCCL_SOCKET_IFNAME" in os.environ:
                # unset nccl comm interface
                del os.environ["NCCL_SOCKET_IFNAME"]
            mp.spawn(pinc_ae_runner, args=(config, world_size), nprocs=world_size)
        elif config.ddp.enable and world_size > 1 and config.ddp.n_nodes > 1:
            # script should be launched via torchrun such that env variables have been set
            rank = int(os.environ["RANK"])
            pinc_ae_runner(rank, config, world_size=world_size)
        else:
            rank = 0
            pinc_ae_runner(rank, config, world_size=1)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        gc.collect()


if __name__ == "__main__":
    """
    normally, load hydra config.
    if a checkpoint is passed:
    - cli args have priority (except the model)
    - the checkpoint config overwrites the default config
    - fields missing from the default config (eg set at a later stage) are dropped
    """

    with hydra.initialize(version_base=None, config_path="configs"):
        config = hydra.compose("main", overrides=sys.argv[1:])

        if config.ae_checkpoint is not None:
            checkpoint_path = osp.abspath(config.ae_checkpoint)
            config_path = osp.join(checkpoint_path, "config.yaml")

            if getattr(config.training, "use_latest_checkpoint", False):
                checkpoint_path = osp.join(checkpoint_path, "ckp.pth")
            else:
                checkpoint_path = osp.join(checkpoint_path, "best.pth")

            if os.path.isfile(checkpoint_path) and os.path.isfile(config_path):
                if config.stage == "peft":
                    # for peft we only need the model weights, not the training config (stage, logging, etc.)
                    print(f"PEFT stage: Will load model weights from {checkpoint_path}")
                else:
                    checkpoint_config = OmegaConf.load(config_path)
                    # NOTE: priority to cli (except model and separate_zf)
                    try:
                        aecli = sys.argv.pop(
                            ["autoencoder" in c for c in sys.argv].index(True)
                        )
                        warnings.warn(
                            f"Autoencoder arg '{aecli}' found in cli, but it was ignored."
                            f"A checkpoint path was specified: '{config_path}'."
                        )
                    except ValueError:
                        pass
                    # special dataset settings set explicitly
                    config.dataset.spatial_ifft = checkpoint_config.dataset.spatial_ifft
                    config.dataset.separate_zf = checkpoint_config.dataset.separate_zf
                    config.dataset.real_potens = checkpoint_config.dataset.real_potens
                    # TODO(diff) what else to drop from autoencoder config?
                    del checkpoint_config.autoencoder.loss_scheduler
                    del checkpoint_config.autoencoder.loss_weights
                    del checkpoint_config.autoencoder.extra_loss_weights
                    filter_cli_priority(sys.argv[1:], checkpoint_config)
                    # drop unknown fields
                    filter_config_subset(config, checkpoint_config)
                    config = OmegaConf.merge(config, checkpoint_config)
                    # overwrite checkpoint
                    config.ae_checkpoint = checkpoint_path
                    print(f"Loaded config from checkpoint '{config_path}'")
            else:
                raise ValueError(f"{checkpoint_path} does not exist!")

        main(config)
