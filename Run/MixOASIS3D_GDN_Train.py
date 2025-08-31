import argparse, os, sys, datetime
from omegaconf import OmegaConf
from transformers import logging as transf_logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
import torch
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utils import instantiate_from_config
from utils.utils_train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from utils.utils_train import set_logger, init_workspace, load_checkpoints, load_checkpoints_unet



def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument("--seed", "-s", type=int, default=20250101, help="seed for seed_everything")

    parser.add_argument("--name", "-n", type=str, default="", help="experiment name, as saving folder")

    parser.add_argument("--base", "-b", nargs="*", metavar="base_config.yaml", help="paths to base configs. Loaded from left-to-right. "
                            "Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    
    parser.add_argument("--train", "-t", action='store_true', default=False, help='train')
    parser.add_argument("--val", "-v", action='store_true', default=False, help='val')
    parser.add_argument("--test", action='store_true', default=False, help='test')

    parser.add_argument("--logdir", "-l", type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument("--auto_resume", action='store_true', default=False, help="resume from full-info checkpoint")
    parser.add_argument("--auto_resume_weight_only", action='store_true', default=False, help="resume from weight-only checkpoint")
    parser.add_argument("--debug", "-d", action='store_true', default=False, help="enable post-mortem debugging")
    

    return parser
    
def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    default_trainer_args = parser.parse_args([])
    return sorted(k for k in vars(default_trainer_args) if getattr(args, k) != getattr(default_trainer_args, k))


parser = get_parser()
# print(parser)
parser = Trainer.add_argparse_args(parser)


### test ####
name="MixedOasis_Layer1" #layers=1
# name="MixedOasis_Layer2" #layers=2
# name="MixedOasis_Layer3" #layers=3
# name="MixedOasis_Layer4" #layers=4
# name="MixedOasis_Layer5" #layers=5

save_root = "2025_SAVE_MELBA"  # Replace with your save root directory
HOST_GPU_NUM = 1


current_directory = os.getcwd()  # current work directory
parent_directory = os.path.dirname(current_directory)  # parent work directory
target_path = os.path.join(parent_directory, save_root, name)
print(target_path)
os.makedirs(target_path, exist_ok=True)

config_file = f"{parent_directory}/configs/{name}/2025config.yaml"  # Replace with the actual config file path
logdir = f"{parent_directory}/{save_root}"  # Replace with the actual log directory path


args, unknown = parser.parse_known_args([
    '--base', config_file,
    '--train',
    '--name', name,
    '--logdir', logdir,
    '--devices', str(HOST_GPU_NUM),
    'lightning.trainer.num_nodes=1'
])


print(args.base)
print(current_directory)
print(parent_directory)


now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
transf_logging.set_verbosity_error()
seed_everything(args.seed)

configs = [OmegaConf.load(cfg) for cfg in args.base]
print(configs)
cli = OmegaConf.from_dotlist(unknown)
print(cli)
print(len(configs))
print(configs[0].keys())
print(configs[0]['data']['params'])


config = OmegaConf.merge(*configs, cli)
lightning_config = config.pop("lightning", OmegaConf.create())
trainer_config = lightning_config.get("trainer", OmegaConf.create()) 
print(lightning_config)
print(trainer_config)


print(args.logdir)


global_rank=0
print(args.name, args.logdir, config, lightning_config, global_rank)
workdir, ckptdir, cfgdir, loginfo = init_workspace(args.name, args.logdir, config, lightning_config, global_rank)



print(config)
print(type(config))
print(config.model)
print(config.data['params']['test']['params']['templateIDX'])

test_data_params = config.data.get('params', {}).get('test', {}).get('params', {})
if 'templateIDX' in test_data_params:
    templateIDX = test_data_params['templateIDX']
    config.model['params']['templateIDX'] = templateIDX
else:
    print("templateIDX not found")

## setup workspace directories
logger = set_logger(logfile=os.path.join(loginfo, 'log_%d:%s.txt'%(global_rank, now)))
logger.info("@lightning version: %s [>=1.8 required]"%(pl.__version__))  
## MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
logger.info("***** Configing Model *****")

config.model.params.logdir = workdir

model = instantiate_from_config(config.model)

## update trainer config
for k in get_nondefault_trainer_args(args):
    trainer_config[k] = getattr(args, k)
    print(k, getattr(args, k))

num_nodes = trainer_config.num_nodes
ngpu_per_node = trainer_config.devices
num_rank=1
logger.info(f"Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs")


logger.info("***** Configing Data *****")
print(config.data)

data = instantiate_from_config(config.data)


logger.info("***** Configing Trainer *****")
if "accelerator" not in trainer_config:
    trainer_config["accelerator"] = "gpu"

print(trainer_config)


## setup trainer args: pl-logger and callbacks
trainer_kwargs = dict()
trainer_kwargs["num_sanity_val_steps"] = 0
logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug)
trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)


## setup callbacks
callbacks_cfg = get_trainer_callbacks(lightning_config, config, workdir, ckptdir, logger)
trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
strategy_cfg = get_trainer_strategy(lightning_config)
trainer_kwargs["strategy"] = strategy_cfg if type(strategy_cfg) == str else instantiate_from_config(strategy_cfg)
trainer_kwargs['precision'] = lightning_config.get('precision', 32)
trainer_kwargs["sync_batchnorm"] = False

print(trainer_kwargs["callbacks"])
print(len(trainer_kwargs["callbacks"]))
print(callbacks_cfg)

trainer_args = argparse.Namespace(**trainer_config)
trainer = Trainer.from_argparse_args(trainer_args, **trainer_kwargs)
print(trainer_args)

seed_everything(args.seed)
trainer.fit(model, data)


