"""
Hydra entry point for GyroSwin and PINC.
"""

import gc
import os
import os.path as osp
import sys
import traceback
from datetime import datetime

import hydra
import torch
import torch.multiprocessing as mp
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Project Imports
from neugk.utils import set_seed, compress_src, find_free_port, filter_cli_priority
from neugk.pinc.autoencoders.ae_utils import restart_config_autoencoder

from neugk.gyroswin import GyroSwinRunner
from neugk.pinc import PINCRunner
from neugk.diffusion import DDPMRunner


def dispatch_runner(rank, config, world_size):
    workflow = config.get("workflow", "gyroswin")
    # get base workflow name (handle pinc_autoencoder, pinc_peft,...)
    base_workflow = workflow.split("_")[0] if "_" in workflow else workflow
    
    if base_workflow == "gyroswin":
        GyroSwinRunner(rank, config, world_size=world_size)()
    elif base_workflow == "pinc":
        PINCRunner(rank, config, world_size=world_size)()
    elif base_workflow == "diffusion":
        DDPMRunner(rank, config, world_size=world_size)()
    else:
        raise NotImplementedError


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    set_seed(config.seed)

    print("#" * 88, "\nStarting Cyclone with configs:")
    print(OmegaConf.to_yaml(config))
    print("#" * 88, "\n")

    workflow = config.get("workflow")
    dict_config = OmegaConf.to_container(config)
    date_and_time = datetime.today().strftime("%Y%m%d_%H%M%S")
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
            stage = workflow.split("_")[-1] if "_" in workflow else "autoencoder"
            config.stage = stage
            if stage == "autoencoder":
                name = f"{stage}_{config.model.name}"
            elif stage == "peft":
                ae_type = config.get("model", {}).get("name", "unknown")
                name = f"peft_{ae_type}"
            elif stage == "diffusion":
                sched = config.diffusion.scheduler.name
                model = config.diffusion.model.name
                name = f"{stage}_{sched}_{model}"
            else:
                name = "experiment"

        config.logging.run_id = f"{name}_{date_and_time}"

    # setup distributed training
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus * config.ddp.n_nodes
    else:
        world_size = 1

    try:
        if config.ddp.enable and world_size > 1:
            if config.ddp.n_nodes == 1:
                # single node, multi gpu
                os.environ["MASTER_ADDR"] = os.environ.get(
                    "SLURM_NODELIST", "localhost"
                )
                os.environ["MASTER_PORT"] = str(find_free_port())
                if "NCCL_SOCKET_IFNAME" in os.environ:
                    del os.environ["NCCL_SOCKET_IFNAME"]

                mp.spawn(dispatch_runner, args=(config, world_size), nprocs=world_size)
            else:
                # multiple nodes (launch via torchrun)
                rank = int(os.environ["RANK"])
                dispatch_runner(rank, config, world_size=world_size)
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
    use_manual_load = False
    for arg in sys.argv:
        if "ae_checkpoint=" in arg:
            use_manual_load = True
            break

    if use_manual_load:
        with hydra.initialize(version_base=None, config_path="configs"):
            main(restart_config_autoencoder())
    else:
        main()
