

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

# %%

parser = get_parser()
# print(parser)
parser = Trainer.add_argparse_args(parser)


name="MELBA_MixedOasis_v1.1_simplenop"
name="MELBA_MixedOasis_v1.1.0_simplenop"
name="MELBA_MixedOasis_v1.1.1_simplenop"
name="MELBA_MixedOasis_v1.1.2_simplenop"


name="MELBA_MixedOasis_v1.1.3_simplenop"
name="MELBA_MixedOasis_LDDMM2"



name="MELBA_MixedOasis_v1.1.3_simplenop" #layers=1
#Ablation study:
name="MELBA_MixedOasis_v2.1.3_simplenop" #layers=2
name="MELBA_MixedOasis_v3.1.3_simplenop" #layers=3
name="MELBA_MixedOasis_v4.1.3_simplenop" #layers=4
# name="MELBA_MixedOasis_v5.1.3_simplenop" #layers=5


name="MELBA_MixedOasis_LDDMM3"   #test the performance of LDDMM  change                     reg weight = 0.1
name="MELBA_MixedOasis_LDDMM4"   #test the performance of LDDMM  change the smoothness1      reg weight = 0.2
name="MELBA_MixedOasis_LDDMM5"   #test the performance of LDDMM  change the smoothness2      reg weight = 0.2


name="MELBA_MixedOasis_SVF2"   #test the performance of LDDMM  change                     reg weight = 0.1
name="MELBA_MixedOasis_SVF2"   #test the performance of LDDMM  change the smoothness1      reg weight = 0.2
name="MELBA_MixedOasis_SVF3"   #test the performance of LDDMM  change the smoothness2      reg weight = 0.2

name="MELBA_MixedOasis_SVF-V2"  #use the same weight as GDN



name="MELBA_MixedOasis_2v1.1.3_simplenop" #layers=1
# name="MELBA_MixedOasis_2v2.1.3_simplenop" #layers=2
# name="MELBA_MixedOasis_2v3.1.3_simplenop" #layers=3
# name="MELBA_MixedOasis_2v4.1.3_simplenop" #layers=4
# name="MELBA_MixedOasis_2v5.1.3_simplenop" #layers=5
# name="MELBA_MixedOasis_2v5.2.3_simplenop" #layers=5


name="MELBA_MixedOasis_LDDMM_0.15_identity"


name="MELBA_MixedOasis_LDDMM_0.5_identity"
# name="MELBA_MixedOasis_LDDMM_1.0_identity"
# name="MELBA_MixedOasis_LDDMM_1.5_identity"
# name="MELBA_MixedOasis_LDDMM_2.0_identity"





# name="MELBA_REG_MixedOasis_SVF_0_0.15"



name="000MELBA_Template_MixedOasis_2v1.1.3_simplenop"
# name="000MELBA_MixedOasis_SVF"
# name="000MELBA_MixedOasis_LDDMM_0.3"
# name="000MELBA_MixedOasis_QS_0.3"




save_root = "2025_SAVE_MELBA"  # Replace with your save root directory
HOST_GPU_NUM = 1

current_directory = os.getcwd()  # 获取当前工作目录                     /home/nellie/code/cvpr/BaseLine/DynamiCrafter/main
parent_directory = os.path.dirname(current_directory)  # 获取父级目录   /home/nellie/code/cvpr/BaseLine/DynamiCrafter
target_path = os.path.join(parent_directory, save_root, name)
print(target_path)
os.makedirs(target_path, exist_ok=True)

config_file = f"{parent_directory}/MELBA_configs/{name}/2025config.yaml"  # Replace with the actual config file path
logdir = f"{parent_directory}/{save_root}"  # Replace with the actual log directory path

args, unknown = parser.parse_known_args([
    '--base', config_file,
    '--train',
    '--name', name,
    '--logdir', logdir,
    '--devices', str(HOST_GPU_NUM),
    'lightning.trainer.num_nodes=1'
])
# print(args)
# print(unknown)

# %%
print(args.base)
print(current_directory)
print(parent_directory)

# %%
print(args)
print(unknown)

# %%
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
## disable transformer warning
transf_logging.set_verbosity_error()
seed_everything(args.seed)

# %%
## yaml configs: "model" | "data" | "lightning"
configs = [OmegaConf.load(cfg) for cfg in args.base]
print(configs)
cli = OmegaConf.from_dotlist(unknown)
print(cli)
print(len(configs))
print(configs[0].keys())
print(configs[0]['data']['params'])

# %%
config = OmegaConf.merge(*configs, cli)
lightning_config = config.pop("lightning", OmegaConf.create())
trainer_config = lightning_config.get("trainer", OmegaConf.create()) 
if "continue_checkpoint" in config.model:
    trainer_config["resume_from_checkpoint"] = config.model.continue_checkpoint
print(lightning_config)
print(trainer_config)

# %%
print(args.logdir)

# %%
global_rank=0
print(args.name, args.logdir, config, lightning_config, global_rank)
workdir, ckptdir, cfgdir, loginfo = init_workspace(args.name, args.logdir, config, lightning_config, global_rank)


# %%
print(config)
print(type(config))
print(config.model)

# %%
## setup workspace directories
logger = set_logger(logfile=os.path.join(loginfo, 'log_%d:%s.txt'%(global_rank, now)))
logger.info("@lightning version: %s [>=1.8 required]"%(pl.__version__))  
## MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
logger.info("***** Configing Model *****")

config.model.params.logdir = workdir

# %%
print(config.model)
model = instantiate_from_config(config.model)
# assert 3>333
## load checkpoints
# model = load_checkpoints(model, config.model)


#compute the number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# assert 3>123



# model = load_checkpoints_unet(model, config.model)
# print(model)
# %%
# print(model)

# %%
print(trainer_config)
print(args)

## update trainer config
for k in get_nondefault_trainer_args(args):
    trainer_config[k] = getattr(args, k)
    print(k, getattr(args, k))

num_nodes = trainer_config.num_nodes
ngpu_per_node = trainer_config.devices
num_rank=1
logger.info(f"Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs")

# %%
## setup learning rate
# base_lr = config.model.base_learning_rate
# bs = config.data.params.batch_size
# if getattr(config.model, 'scale_lr', True):
#     model.learning_rate = num_rank * bs * base_lr
# else:
#     model.learning_rate = base_lr

# %%
## DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
logger.info("***** Configing Data *****")
print(config.data)

data = instantiate_from_config(config.data)

# %%
# TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
logger.info("***** Configing Trainer *****")
if "accelerator" not in trainer_config:
    trainer_config["accelerator"] = "gpu"

print(trainer_config)

# %%
## setup trainer args: pl-logger and callbacks
trainer_kwargs = dict()
trainer_kwargs["num_sanity_val_steps"] = 0
logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug)
trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

# %%
print(logger_cfg)

# %%
print(workdir)
print(ckptdir)

# %%

## setup callbacks
callbacks_cfg = get_trainer_callbacks(lightning_config, config, workdir, ckptdir, logger)
trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
strategy_cfg = get_trainer_strategy(lightning_config)
trainer_kwargs["strategy"] = strategy_cfg if type(strategy_cfg) == str else instantiate_from_config(strategy_cfg)
trainer_kwargs['precision'] = lightning_config.get('precision', 32)
trainer_kwargs["sync_batchnorm"] = False

# %%
print(trainer_kwargs["callbacks"])
print(len(trainer_kwargs["callbacks"]))
print(callbacks_cfg)

# %%

## trainer config: others
trainer_args = argparse.Namespace(**trainer_config)
trainer = Trainer.from_argparse_args(trainer_args, **trainer_kwargs)
print(trainer_args)

trainer.fit(model, data)

