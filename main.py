from datetime import datetime
import gc
import os
import os.path as osp
import sys
import traceback
import torch.multiprocessing as mp
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

from utils import set_seed, compress_src, find_free_port


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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

    dict_config["ckpt_path"] = dict_config["output_path"]
    config = OmegaConf.create(dict_config)

    if not os.path.exists(config.output_path):
        os.makedirs(dict_config["output_path"], exist_ok=True)

    compress_src(dict_config["output_path"])

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    if config.method == "default":
        from run import runner
    if config.method == "xnet":
        from experimental.run_xnet import runner

    try:
        if dict_config["logging"]["run_id"] is None:
            print(dict_config["model"]["name"])
            dict_config["logging"][
                "run_id"
            ] = f"{dict_config['model']['name']}_{date_and_time}"
            config = OmegaConf.create(dict_config)

        if config.use_ddp and world_size > 1:
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
            mp.spawn(runner, args=(config, world_size), nprocs=world_size)
        else:
            rank = 0
            runner(rank, config, world_size=1)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        gc.collect()


if __name__ == "__main__":
    main()
