"""Hydra entrypoint."""

from datetime import datetime
import gc
import os
import os.path as osp
import sys
import traceback
import torch.multiprocessing as mp
import torch
import yaml

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from neugk.utils import set_seed, compress_src, find_free_port
from neugk.gyroswin.run import gyroswin_runner


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    set_seed(config.seed)

    dict_config = OmegaConf.to_container(config)
    date_and_time = datetime.today().strftime("%Y%m%d_%H%M%S")
    choices = HydraConfig.get().runtime.choices
    dict_config["choices"] = choices
    if not config.load_ckpt:
        if config.output_path is None:
            dict_config["output_path"] = osp.join("outputs", date_and_time)
        else:
            dict_config["output_path"] = osp.join(
                dict_config["output_path"], date_and_time
            )

        if not os.path.exists(dict_config["output_path"]):
            os.makedirs(dict_config["output_path"], exist_ok=True)

        compress_src(dict_config["output_path"])

        if dict_config["logging"]["run_id"] is None:
            run_id = f"{dict_config['model']['name']}_{date_and_time}"
            dict_config["logging"]["run_id"] = run_id
        config = OmegaConf.create(dict_config)
    else:
        # check that output path exists
        assert os.path.exists(
            config.output_path
        ), "Output path does not exist, cannot load ckpt"
        assert os.path.exists(
            f"{config.output_path}/ckp.pth"
        ), "Output path does not contain checkpoint best.pt"
        config = OmegaConf.load(f"{config.output_path}/config.yaml")
        cli_conf = OmegaConf.from_dotlist(
            [str(o) for o in HydraConfig.get().overrides.task]
        )
        config = OmegaConf.merge(config, cli_conf)
        config.logging.run_id = f"{config.model.name}_{date_and_time}"
        print(cli_conf)

    print("#" * 88, "\nStarting Cyclone with configs:")
    print(OmegaConf.to_yaml(config))
    print("#" * 88, "\n")

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus * config.ddp.n_nodes
    else:
        world_size = 1

    try:
        if config.ddp.enable and world_size > 1 and config.ddp.n_nodes == 1:
            if "SLURM_NODELIST" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            else:
                # only works for single node so far, adapt above for multinode
                os.environ["MASTER_ADDR"] = os.environ["SLURM_NODELIST"]
            os.environ["MASTER_PORT"] = str(find_free_port())
            if "NCCL_SOCKET_IFNAME" in os.environ:
                # unset nccl comm interface
                del os.environ["NCCL_SOCKET_IFNAME"]
            mp.spawn(gyroswin_runner, args=(config, world_size), nprocs=world_size)
        elif config.ddp.enable and world_size > 1 and config.ddp.n_nodes > 1:
            # script should be launched via torchrun to set env variables
            rank = int(os.environ["RANK"])
            gyroswin_runner(rank, config, world_size=world_size)
        else:
            rank = 0
            gyroswin_runner(rank, config, world_size=1)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        gc.collect()


if __name__ == "__main__":
    main()
