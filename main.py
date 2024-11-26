import gc
import os
import os.path as osp
import uuid
import sys
import traceback

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from run import runner
from utils import set_seed, compress_src, setup_logging


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    print(config)
    set_seed(config.seed)

    hydra_cfg = HydraConfig.get()
    launcher = hydra_cfg["runtime"]["choices"]["hydra/launcher"]
    dataset_choice = hydra_cfg["runtime"]["choices"]["dataset"]

    dict_config = OmegaConf.to_container(config)
    if config.output_path is None:
        dict_config["output_path"] = osp.join(
            "outputs", str(uuid.uuid4()).split("-")[0]
        )
        config = OmegaConf.create(dict_config)

    if not os.path.exists(config.output_path):
        os.makedirs(dict_config["output_path"], exist_ok=True)

    compress_src(config.output_path)

    try:
        logging_writer = setup_logging(config)
        runner(config, logging_writer)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        gc.collect()


if __name__ == "__main__":
    main()
