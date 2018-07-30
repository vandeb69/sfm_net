import importlib
import tensorflow as tf

# from data_loader.sfm_net import SfmNetLoader_DeepTesla
from models.sfm_net_model import SfmNetModel
from trainers.sfm_net_trainer import SfmNetTrainer

from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import DefinedSummarizer
from utils.utils import import_from

if __name__ == "__main__":
    config_file = "../configs/example_sfm_net.json"
    config = process_config(config_file)

    create_dirs([config.summary_dir, config.checkpoint_dir])
    Loader = import_from("data_loader.sfm_net",  config.data_loader)
    data_loader = Loader(config)


