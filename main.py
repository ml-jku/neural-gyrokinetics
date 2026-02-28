"""
Hydra entry point for GyroSwin and PINC.
"""

import gc
import os
import os.path as osp
import sys
import traceback
from datetime import datetime
import subprocess
import random

import torch
import yaml

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as mp

from neugk.utils import compress_src, filter_cli_priority, find_free_port

from neugk.gyroswin import GyroSwinRunner
from neugk.pinc import PINCRunner
from neugk.diffusion import get_diffusion_runner


def dispatch_runner(rank, config, world_size):
    workflow = config.get("workflow", "gyroswin")
    # get base workflow name (handle pinc_autoencoder, pinc_peft,...)
    base_workflow = workflow.split("_")[0] if "_" in workflow else workflow

    if base_workflow == "gyroswin":
        GyroSwinRunner(rank, config, world_size=world_size)()
    elif base_workflow == "pinc":
        PINCRunner(rank, config, world_size=world_size)()
    elif workflow == "diffusion":
        get_diffusion_runner(rank, config, world_size=world_size)()
    else:
        raise NotImplementedError


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    rand_suffix = random.randint(0, 999)
    print("#" * 88, "\nStarting Cyclone with configs:")
    print(OmegaConf.to_yaml(config))
    print("#" * 88, "\n")

    workflow = config.get("workflow")
    dict_config = OmegaConf.to_container(config)
    date_and_time = datetime.today().strftime("%Y%m%d_%H%M%S")
    date_and_time = f"{date_and_time}_{rand_suffix:03d}"
    if HydraConfig.initialized():
        dict_config["choices"] = HydraConfig.get().runtime.choices

    if workflow == "gyroswin" and config.get("load_ckpt"):
        assert os.path.exists(
            config.output_path
        ), "Output path does not exist, cannot load ckpt"
        assert os.path.exists(
            f"{config.output_path}/ckp.pth"
        ), "Output path does not contain checkpoint"

        loaded_conf = OmegaConf.load(f"{config.output_path}/config.yaml")
        filter_cli_priority(sys.argv[1:], loaded_conf)
        config = OmegaConf.merge(config, loaded_conf)

        config.logging.run_id = f"{config.model.name}_{date_and_time}"
        dict_config = OmegaConf.to_container(config)
    else:
        if config.output_path is None:
            dict_config["output_path"] = osp.join("outputs", date_and_time)
        else:
            # TODO ignores if there are checkpoints in there
            dict_config["output_path"] = osp.join(
                dict_config["output_path"], date_and_time
            )

        if not os.path.exists(dict_config["output_path"]):
            os.makedirs(dict_config["output_path"], exist_ok=True)

        compress_src(dict_config["output_path"])
        config = OmegaConf.create(dict_config)

    # peft logic (pinc only)
    if workflow == "pinc" and config.get("stage") == "peft":
        if dict_config.get("ae_checkpoint"):
            try:
                ae_checkpoint_path = dict_config["ae_checkpoint"]
                # Determine config path
                ae_config_path = (
                    osp.join(osp.dirname(ae_checkpoint_path), "config.yaml")
                    if os.path.isfile(ae_checkpoint_path)
                    else osp.join(ae_checkpoint_path, "config.yaml")
                )

                print(f"Loading autoencoder config from: {ae_config_path}")

                if os.path.exists(ae_config_path):
                    with open(ae_config_path, "r") as f:
                        checkpoint_config = yaml.safe_load(f)

                    checkpoint_ae_config = checkpoint_config.get("model", {})

                    current_peft = dict_config["model"].get("peft", {})
                    current_loss = dict_config["model"].get("loss_weights", {})
                    current_extra = dict_config["model"].get("extra_loss_weights", {})
                    current_sched = dict_config["model"].get("loss_scheduler", {})

                    dict_config["model"] = checkpoint_ae_config.copy()
                    dict_config["model"]["peft"] = current_peft
                    if current_loss:
                        dict_config["model"]["loss_weights"] = current_loss
                    if current_extra:
                        dict_config["model"]["extra_loss_weights"] = current_extra
                    if current_sched:
                        dict_config["model"]["loss_scheduler"] = current_sched

                    print("Merged config from checkpoint with PEFT settings")
                    config = OmegaConf.create(dict_config)
                else:
                    raise FileNotFoundError(f"Config not found: {ae_config_path}")
            except Exception as e:
                print(f"ERROR loading model config for PEFT: {e}")
                raise
        else:
            raise ValueError("PEFT stage requires ae_checkpoint parameter")

    if workflow == "pinc":
        if config.model.get("loss_weights") is None:
            config.model.loss_weights = {}
        if config.model.get("extra_loss_weights") is None:
            config.model.extra_loss_weights = {}

    # generate id
    if config.logging.run_id is None:
        if workflow == "gyroswin":
            name = config.model.name
        else:
            # set training style for pinc
            workflow = config.get("workflow", "unknown")
            stage = config.get("stage", "autoencoder")
            config.stage = stage
            if stage == "autoencoder":
                name = f"{stage}_{config.model.name}"
            elif stage == "peft":
                ae_type = config.get("model", {}).get("name", "unknown")
                name = f"peft_{ae_type}"
            elif stage == "diffusion":
                model = config.model.model_type
                name = f"{stage}_{model}"
            else:
                name = "experiment"

        config.logging.run_id = f"{name}_{date_and_time}"

    try:
        is_ddp = config.ddp.enable
        is_torchrun = "RANK" in os.environ
        is_slurm = "SLURM_JOB_ID" in os.environ

        if is_ddp:
            world_size = config.ddp.n_nodes * torch.cuda.device_count()
            if not is_torchrun and is_slurm:
                overrides = HydraConfig.get().overrides.task
                overrides = [
                    o for o in overrides if not o.startswith("hydra/launcher=")
                ]
                if config.ddp.n_nodes == 1:
                    # should be run with torchrun
                    cmd = [
                        "torchrun",
                        f"--nproc_per_node={torch.cuda.device_count()}",
                        "main.py",
                    ] + overrides
                else:
                    # multinode setup
                    cmd = [
                        "torchrun",
                        f"--nnodes={config.ddp.n_nodes}",
                        f"--nproc_per_node={torch.cuda.device_count()}",
                        f"--rdzv_backend={os.environ['RDZV_BACKEND']}",
                        f"--rdzv_id={os.environ['RDZV_ID']}",
                        f"--rdzv_endpoint={os.environ['HEAD_NODE_IP']}:29501",
                        "main.py",
                    ] + overrides

                print(f"DDP job with command: {' '.join(cmd)}")
                subprocess.check_call(cmd)
                return
            elif is_torchrun and not is_slurm or is_torchrun and is_slurm:
                rank = int(os.environ["RANK"])
                dispatch_runner(rank, config, world_size=world_size)
            else:
                # here we revert to mp.spawn to allow "normal" DDP startup
                print(f"Local DDP with mp.spawn (nprocs={torch.cuda.device_count()})")
                os.environ["MASTER_ADDR"] = "localhost"
                os.environ["MASTER_PORT"] = str(find_free_port())
                if "NCCL_SOCKET_IFNAME" in os.environ:
                    del os.environ["NCCL_SOCKET_IFNAME"]

                mp.spawn(
                    dispatch_runner,
                    args=(config, world_size),
                    nprocs=torch.cuda.device_count(),
                    join=True,
                )
        else:
            # single gpu
            rank = 0
            dispatch_runner(rank, config, world_size=1)

    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        gc.collect()


if __name__ == "__main__":
    main()
