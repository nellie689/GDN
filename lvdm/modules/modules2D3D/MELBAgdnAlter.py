import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import torchvision
from utils.utils import instantiate_from_config
from lvdm.Int import VecInt, SpatialTransformer, jacobian_det_for_all, save_dfm_dfmlabel,\
    Epdiff, Grad, Mgridplot, add_zero_channel_and_norm, plot_dfm_process, plot_norm_list_gpu, dice_coefficient_for_brain, hdorff_coefficient_for_brain
import lagomorph as lm
import numpy as np
import time
import os


hookflag = False
def grad_stats(name):
    def hook(grad):
        if hookflag:
            if torch.is_complex(grad):
                real_grad = grad.real
                imag_grad = grad.imag

                print(f"[{name}] Gradient Stats (Real) | Mean: {real_grad.mean().item():.4e} | Std: {real_grad.std().item():.4e} | "f"Max: {real_grad.max().item():.4e} | Min: {real_grad.min().item():.4e}")
                print(f"[{name}] Gradient Stats (Imag) | Mean: {imag_grad.mean().item():.4e} | Std: {imag_grad.std().item():.4e} | "f"Max: {imag_grad.max().item():.4e} | Min: {imag_grad.min().item():.4e}")
            else:
                print(f"[{name}] Gradient Stats | Mean: {grad.mean().item():.4e} | Std: {grad.std().item():.4e} | "f"Max: {grad.max().item():.4e} | Min: {grad.min().item():.4e}")
        else:
            pass
    return hook

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()

class NN(pl.LightningModule):
    def __init__(self, unet_config, nop_config, reg_config, nop_loss_config, weight_mse, weight_reg, weight_sigma, unet_lr, nop_lr, templateIDX=-1,
                 *args, **kwargs):
        super().__init__()
        self.indexs_src = []
        self.indexs_tar = []
        self.templateIDX = templateIDX

        self.unet = instantiate_from_config(unet_config)
        self.nop = instantiate_from_config(nop_config)
        self.regmetric = instantiate_from_config(reg_config)
        self.nop_loss = instantiate_from_config(nop_loss_config)

        self.mse_metric = torchmetrics.MeanSquaredError()
        self.ndims = unet_config['params']['ndims']
        self.TSteps = nop_config['params']['TSteps']


        self.weight_mse = weight_mse / (weight_sigma**2)
        self.weight_reg = weight_reg
        self.unet_lr = unet_lr
        self.nop_lr = nop_lr
        self.ReparametrizeVectorField = kwargs['ReparametrizeVectorField'] if 'ReparametrizeVectorField' in kwargs else False
        # print(self.ReparametrizeVectorField)
        # assert 3>129
        
        self.PredictMomnetum = kwargs['PredictMomnetum'] if 'PredictMomnetum' in kwargs else False
        
        self.alter_train_epoches = kwargs['alter_train_epoches'] if 'alter_train_epoches' in kwargs else [200,3000] #[100,600]

        self.MEpdiff = Epdiff(**{
            'inshape': kwargs['inshape'],
            'alpha': kwargs['alpha'], #alpha,
            'gamma': kwargs['gamma'], #gamma,
            # 'TSteps': kwargs['TSteps']
            'TSteps': nop_config['params']['TSteps']
        })
        self.grid = self.MEpdiff.identity(kwargs['inshape'])
        self.inshape = kwargs['inshape']

        # Register hooks for self.unet parameters
        cnt=0
        for name, param in self.unet.named_parameters():
            cnt += 1
            if cnt % 2 == 0:
                if param.requires_grad:
                    param.register_hook(grad_stats(f"self.unet.{name}"))

        # Register hooks for self.nop parameters
        cnt = 0
        for name, param in self.nop.named_parameters():
            cnt += 1
            if cnt % 10 == 0:
                if param.requires_grad:
                    param.register_hook(grad_stats(f"self.nop.{name}"))



        # Call configure_optimizers to count optimizers
        optimizers = self.configure_optimizers()
        # Handle cases where a single optimizer or a list of optimizers is returned
        if isinstance(optimizers, (list, tuple)):
            self.num_optimizers = len(optimizers)
        elif isinstance(optimizers, dict) and 'optimizer' in optimizers:
            # Case where a dict is returned with 'optimizer' key
            self.num_optimizers = 1 if isinstance(optimizers['optimizer'], optim.Optimizer) else len(optimizers['optimizer'])
        else:
            self.num_optimizers = 1  # Assume single optimizer


        if self.num_optimizers==1:
            self.training_step = self.training_step1
            self.optimizer_step = self.optimizer_step1
        else:
            self.training_step = self.training_step2
            self.optimizer_step = self.optimizer_step2

    def encoder_step(self, x):
        VecLnt = self.unet.exec_encoder(x)  #V in Latent space   [1, 8, 8, 8, 8]
        return VecLnt
    
    def decoder_step(self, x):
        VecF = self.unet.exec_decoder(x) #V in full space
        return VecF
    
    def decoder_step_parallel(self, x_List):
        VecF_List = self.unet.parallel_decoder(x_List) #V in full space
        return VecF_List
    
    def nop_step(self, x):
        VecLnt_List = self.nop(x)
        return VecLnt_List


    # phiinv_disp_list, phi_disp_list, v_list, m_list = self.MEpdiff.ShootWithV0(velocity_0)
    def common_step(self, batch, batch_idx):
        if self.ndims == 2:
            b,c,w,h = batch['src'].shape
            self.src = batch['src'].view(-1, 1, w,h) #shape [120, 1, 32, 32]
            self.tar = batch['tar'].view(-1, 1, w,h) #shape [120, 1, 32, 32]
        else:
            self.src = batch['src']
            self.tar = batch['tar']
            self.srclabel = batch['srclabel']
            self.tarlabel = batch['tarlabel']

        ipt = torch.cat((self.src, self.tar), dim=1)  #shape [120, 2, 32, 32]



        VecLnt = self.encoder_step(ipt)
        VecLnt_List = self.nop_step(VecLnt)  #len=5

        if self.PredictMomnetum:
            MomF_List_pre = [ self.decoder_step(VecLnt) for VecLnt in VecLnt_List ]
            VecF_List_pre = [self.MEpdiff.metric.sharp(VecF) for VecF in MomF_List_pre]
            # VecF_List_pre__ = [self.MEpdiff.metric.sharp(VecF) for VecF in MomF_List_pre]
            # VecF_List_pre = [self.MEpdiff.metric.flat(VecF) for VecF in VecF_List_pre__]

        else:
            VecF_List_pre = [ self.decoder_step(VecLnt) for VecLnt in VecLnt_List ]
            MomF_List_pre = [self.MEpdiff.metric.flat(VecF) for VecF in VecF_List_pre]

        if self.ReparametrizeVectorField:
            VecF_List = self.regmetric.ReparametrizeVec(VecF_List_pre, MomF_List_pre)
            MomF_List = [self.MEpdiff.metric.flat(VecF) for VecF in VecF_List]
        else:
            VecF_List = VecF_List_pre
            MomF_List = MomF_List_pre

        phiinv_disp_list, phi_disp_list = self.MEpdiff.IntWithVList2(VecF_List)
        dfmSrc = self.MEpdiff.transformer(self.src, phiinv_disp_list[-1])


        self.dfmSrc = dfmSrc

        #Loss Part 1
        mse = self.mse_metric(dfmSrc, self.tar)
        
        #Loss Part 2
        VecF_List_Tensor_0 = VecF_List[0].unsqueeze(0)
        MomF_List_Tensor_0 = MomF_List[0].unsqueeze(0)
        reg = self.regmetric(VecF_List_Tensor_0, MomF_List_Tensor_0)

        #Loss Part 3
        # phiinv_disp_list, phi_disp_list, v_list, m_list = self.MEpdiff.ShootWithV0(VecF_List[0])
        phiinv_disp_list_gt, phi_disp_list_gt, v_list_gt, m_list_gt = self.MEpdiff.ShootWithV0(VecF_List[0])
        

        VecF_List_Tensor = torch.stack(VecF_List[1:], dim=0)
        VecFGT_List_Tensor = torch.stack(v_list_gt[1:], dim=0)
        lossobj = self.nop_loss(VecF_List_Tensor, VecFGT_List_Tensor)
        noploss = lossobj['noploss']
        noprel = lossobj['noprel']
        nopmse = lossobj['nopmse']

        loss = self.weight_mse * mse + self.weight_reg * reg + noploss
        los_regis = self.weight_mse * mse + self.weight_reg * reg


        #predicted from FNO
        return VecF_List, MomF_List, loss, mse, reg, phiinv_disp_list, phi_disp_list, VecF_List_pre, MomF_List_pre, noprel, nopmse, noploss, los_regis
        
    def common_step_(self, batch, batch_idx):
        VecF_List, MomF_List, loss, mse, reg, _, _, _, _, noprel, nopmse, noploss, los_regis = self.common_step(batch, batch_idx)
        
        self.log_dict(
            {
                "train_loss": loss,
                "train_mse": mse,
                "train_reg": reg,
                "train_noprel": noprel,
                "train_nopmse": nopmse,
                "train_noploss": noploss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        print("~~~~~~~~~~~~~~~~~~~~~~~", noploss.item(), los_regis.item())
        # return loss
        # return noploss
        return los_regis
    
    def training_step1(self, batch, batch_idx, optimizer_idx=None):
        return self.common_step_(batch, batch_idx)
    
    def optimizer_step1(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_lbfgs):
        # print("optimizer_step   ", epoch, batch_idx, optimizer_idx)
        optimizer.step(closure=optimizer_closure)
    
    def training_step2(self, batch, batch_idx, optimizer_idx=None): #[optimizer_unet, optimizer_nop, optimizer_all]
        # self.alter_train_epoches = [200,3000]
        ep_per_round = self.alter_train_epoches[0]
        ep_for_alter = self.alter_train_epoches[1]

        # print("~~~~~~~~~ training_step   ", self.current_epoch, batch_idx, optimizer_idx)
        if optimizer_idx == 0:  #optimizer_unet
            if self.current_epoch % int(ep_per_round*2) < ep_per_round and self.current_epoch < ep_for_alter:  #falls within the interval [0, 200) or [400, 600) or [800, 1000) ...")
                return self.common_step_(batch, batch_idx)

        if optimizer_idx == 1: #optimizer_nop
            if self.current_epoch % int(ep_per_round*2) >= ep_per_round and self.current_epoch < ep_for_alter:
                return self.common_step_(batch, batch_idx)
        
        if optimizer_idx == 2: #optimizer_all
            if self.current_epoch >= ep_for_alter:
                return self.common_step_(batch, batch_idx)
    
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
        # print(f"Epoch {self.current_epoch} started at {time.strftime('%H:%M:%S')}")
    
    def on_train_epoch_end(self):
        if self.epoch_start_time is not None:
            elapsed_time = time.time() - self.epoch_start_time
            print(f"\n\n Epoch {self.current_epoch} ended. Time elapsed: {elapsed_time:.2f} seconds. \n\n")

    def optimizer_step2(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_lbfgs):
        # print("!!!!!! optimizer_step   ", epoch, batch_idx, optimizer_idx)
        if optimizer_idx == 0: #optimizer_nop
            if self.current_epoch % 400 < 200 and self.current_epoch < 3000:  #falls within the interval [0, 200) or [400, 600) or [800, 1000) ...")
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()
        
        if optimizer_idx == 1: #optimizer_unet
            if self.current_epoch % 400 >= 200 and self.current_epoch < 3000:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()
        
        if optimizer_idx == 2:
            if self.current_epoch >= 3000:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()
    
    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=self.unet_lr)
        unet_params = self.unet.parameters()
        nop_params = self.nop.parameters()

        optimizer_unet = optim.Adam(
            unet_params,
            lr=self.unet_lr,
        )
        optimizer_nop = optim.Adam(
            nop_params,
            lr=self.nop_lr,
            # weight_decay=1e-5  # 可选的权重衰减
        )
        optimizer_all = optim.Adam(self.parameters(), lr=self.unet_lr)


        return optimizer_all
    

    def validation_step(self, batch, batch_idx):
        VecF_List, MomF_List, loss, mse, reg, _, _ , _, _, noprel, nopmse, noploss, los_regis = self.common_step(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
                "val_mse": mse,
                "val_reg": reg,
                "val_noprel": noprel,
                "val_nopmse": nopmse,
                "val_noploss": noploss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss


    ## Normal test_step
    def test_step(self, batch, batch_idx):
        VecF_List, MomF_List, loss, mse, reg, phiinv_disp_list, phi_disp_list, VecF_List_pre, MomF_List_pre, noprel, nopmse, noploss, los_regis = self.common_step(batch, batch_idx)

        dfmSrc_list = [self.MEpdiff.transformer(self.src, phiinv_disp) for phiinv_disp in phiinv_disp_list]
        if self.ndims == 3:
            dfmSrclabel = self.MEpdiff.transformer(self.srclabel, phiinv_disp_list[-1], mode='nearest')

        self.log_dict(
            {
                "test_loss": loss,
                "test_mse": mse,
                "test_reg": reg,
                "test_noprel": noprel,
                "test_nopmse": nopmse,
                "test_noploss": noploss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # torch.cuda.empty_cache()
       

        return {
            "dfmSrclabel": dfmSrclabel if self.ndims == 3 else None,
            "tarlabel": self.tarlabel if self.ndims == 3 else None,
            "srclabel": self.srclabel if self.ndims == 3 else None,
            
            # "phiinv": phiinv_disp_list[-1],
            "phiinv": phi_disp_list[-1],

            "pred": self.dfmSrc,
            "src": self.src,
            "tar": self.tar,
            "loss": loss,
            "mse": mse,
            "reg": reg,
            "VecF_List": VecF_List,
            "phiinv_disp_list": phiinv_disp_list,
            # "phiinv_disp_list": phi_disp_list,
            # # "phi_disp_list": phi_disp_list,
            "dfmSrc_list": dfmSrc_list,

            # "MomF_List": MomF_List,

            # "VecF_List_pre": VecF_List_pre,
            # "MomF_List_pre": MomF_List_pre,
            
        }
        return loss
    
    def test_epoch_end1(self, outputs):
        plot_norm_list_gpu(outputs, self.regmetric, self.mse_metric)
    
    def test_epoch_end(self, outputs):
        plot_dfm_process(outputs, ongpuGrad=False, mse_metric=self.mse_metric, TSteps=self.TSteps, inshape=self.inshape, showTSteps=[0,1,2,3,4], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES",saveRes=False)
        
        
    def on_load_checkpoint(self, checkpoint):
        print(self.current_epoch, self.global_step) #3995 59940
        print(checkpoint.keys())
        print(checkpoint['epoch'], checkpoint['global_step'])
        print("Checkpoint is being loaded!")
        # assert 2>128







