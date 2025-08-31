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
from lvdm.Int import VecInt, SpatialTransformer, jacobian_det_for_all, save_dfm_dfmlabel, get_jacobian_det_for_all, jacobian_determinant, \
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
                print(f"Gradient stats for {name} (real part): mean={real_grad.mean().item()}, std={real_grad.std().item()}, max={real_grad.max().item()}, min={real_grad.min().item()}")
                print(f"Gradient stats for {name} (imaginary part): mean={imag_grad.mean().item()}, std={imag_grad.std().item()}, max={imag_grad.max().item()}, min={imag_grad.min().item()}")
            else:
                print(f"Gradient stats for {name}: mean={grad.mean().item()}, std={grad.std().item()}, max={grad.max().item()}, min={grad.min().item()}")
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
    def __init__(self, unet_config, nop_config, reg_config, weight_mse, weight_reg, weight_sigma, unet_lr, nop_lr,
                 *args, **kwargs):
        super().__init__()
        # print(self.current_epoch, self.global_step) #3995 59940
        # self.current_epoch = 3995
        # self.global_step = 59940
        # self.start_epoch = 3995
        # self.start_global_step = 59940

        self.indexs_src = []
        self.indexs_tar = []

        self.unet = instantiate_from_config(unet_config)
        self.nop = instantiate_from_config(nop_config)
        self.regmetric = instantiate_from_config(reg_config)
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
            if cnt % 7 == 0:
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
        VecLnt = self.unet.exec_encoder(x)  #V in Latent space
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


    def common_step_parallel(self, ipt, batch_idx):
        ## test time efficienty of ShootWithV0   
        VecLnt = self.encoder_step(ipt)     
        time_start = time.time()
        for i in range(100):
            VecLnt_List = self.nop_step(VecLnt)
            if self.PredictMomnetum:
                MomF_List_pre = self.decoder_step_parallel(VecLnt_List)
                VecF_List_pre = [self.MEpdiff.metric.sharp(VecF) for VecF in MomF_List_pre]
            else:
                VecF_List_pre = self.decoder_step_parallel(VecLnt_List)
                MomF_List_pre = [self.MEpdiff.metric.flat(VecF) for VecF in VecF_List_pre]
            # if self.ReparametrizeVectorField:
            #     VecF_List = self.regmetric.ReparametrizeVec(VecF_List_pre, MomF_List_pre)
            #     MomF_List = [self.MEpdiff.metric.flat(VecF) for VecF in VecF_List]
            # else:
            #     VecF_List = VecF_List_pre
            #     MomF_List = MomF_List_pre
        time_end = time.time()
        print("Time for ShootWithV0: ", (time_end - time_start)/100)  
        # Time for ShootWithV0:  0.020390443801879883
        assert 3>333



        VecLnt = self.encoder_step(ipt)
        VecLnt_List = self.nop_step(VecLnt)
        if self.PredictMomnetum:
            MomF_List_pre = self.decoder_step_parallel(VecLnt_List)
            VecF_List_pre = [self.MEpdiff.metric.sharp(VecF) for VecF in MomF_List_pre]
        else:
            VecF_List_pre = self.decoder_step_parallel(VecLnt_List)
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

        mse = self.mse_metric(dfmSrc, self.tar)

        VecF_List_Tensor = torch.stack(VecF_List, dim=0)
        MomF_List_Tensor = torch.stack(MomF_List, dim=0)

        reg = self.regmetric(VecF_List_Tensor, MomF_List_Tensor)
        loss = self.weight_mse * mse + self.weight_reg * reg

        return VecF_List, MomF_List, loss, mse, reg, phiinv_disp_list, phi_disp_list
    
        
    def common_step_test_time(self, batch, batch_idx):
        TN =100
        time_start = time.time()
        for i in range(TN):
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
            TN =100
            time_start = time.time()
            for i in range(TN):
                VecLnt_List = self.nop_step(VecLnt)
                VecF_List_pre = [ self.decoder_step(VecLnt) for VecLnt in VecLnt_List ]
            
            time_end = time.time()
            print("Time for Regsitration: ", (time_end - time_start)/TN)  
            #Time for Regsitration:  0.010874528884887696  2D
            assert 3>333


            phiinv_disp_list, phi_disp_list = self.MEpdiff.IntWithVList2(VecF_List_pre)



            

            dfmSrc = self.MEpdiff.transformer(self.src, phiinv_disp_list[-1])
            self.dfmSrc = dfmSrc




        time_end = time.time()
        print("Time for Regsitration: ", (time_end - time_start)/TN)  
        #Time for Regsitration:  0.010874528884887696  2D
        assert 3>333



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

        print("src: ", self.src.shape, self.src.max(), self.src.min(), self.src.mean())
        assert 3>128


        ipt = torch.cat((self.src, self.tar), dim=1)  #shape [120, 2, 32, 32]

        
        VecLnt = self.encoder_step(ipt)
        VecLnt_List = self.nop_step(VecLnt)



        # time_start = time.time()
        # for i in range(100):
        #     VecLnt_List = self.nop_step(VecLnt)
        # time_end = time.time()
        # print("Time for Regsitration: ", (time_end - time_start)/100)  
        # #Time for Regsitration:  0.0036987829208374023
        # assert 3>333





        # time_start = time.time()
        # for i in range(100):
        #     VecLnt_List = self.nop_step(VecLnt)
        #     if self.PredictMomnetum:
        #         MomF_List_pre = [ self.decoder_step(VecLnt) for VecLnt in VecLnt_List ]
        #         VecF_List_pre = [self.MEpdiff.metric.sharp(VecF) for VecF in MomF_List_pre]
        #     else:
        #         VecF_List_pre = [ self.decoder_step(VecLnt) for VecLnt in VecLnt_List ]
        #         MomF_List_pre = [self.MEpdiff.metric.flat(VecF) for VecF in VecF_List_pre]
        #     if self.ReparametrizeVectorField:
        #         VecF_List = self.regmetric.ReparametrizeVec(VecF_List_pre, MomF_List_pre)
        #         MomF_List = [self.MEpdiff.metric.flat(VecF) for VecF in VecF_List]
        #     else:
        #         VecF_List = VecF_List_pre
        #         MomF_List = MomF_List_pre
        # time_end = time.time()
        # print("Time for Regsitration: ", (time_end - time_start)/100)  
        # #Time for Regsitration:  0.0036987829208374023
        # assert 3>333



        # time_start = time.time()
        # for i in range(100):
        #     VecLnt_List = self.nop_step(VecLnt)
        #     if self.PredictMomnetum:
        #         MomF_List_pre = [ self.decoder_step(VecLnt) for VecLnt in VecLnt_List ]
        #         VecF_List_pre = [self.MEpdiff.metric.sharp(VecF) for VecF in MomF_List_pre]
        #     else:
        #         VecF_List_pre = [ self.decoder_step(VecLnt) for VecLnt in VecLnt_List ]
        #         MomF_List_pre = [self.MEpdiff.metric.flat(VecF) for VecF in VecF_List_pre]
        #     if self.ReparametrizeVectorField:
        #         VecF_List = self.regmetric.ReparametrizeVec(VecF_List_pre, MomF_List_pre)
        #         MomF_List = [self.MEpdiff.metric.flat(VecF) for VecF in VecF_List]
        #     else:
        #         VecF_List = VecF_List_pre
        #         MomF_List = MomF_List_pre
        #     phiinv_disp_list, phi_disp_list = self.MEpdiff.IntWithVList2(VecF_List)
        #     dfmSrc = self.MEpdiff.transformer(self.src, phiinv_disp_list[-1])
        # time_end = time.time()
        # print("Time for Regsitration: ", (time_end - time_start)/100)  
        # #Time for Regsitration:  0.0036987829208374023
        # assert 3>333






        
        if self.PredictMomnetum:
            MomF_List_pre = [ self.decoder_step(VecLnt) for VecLnt in VecLnt_List ]
            VecF_List_pre = [self.MEpdiff.metric.sharp(VecF) for VecF in MomF_List_pre]
        else:
            VecF_List_pre = [ self.decoder_step(VecLnt) for VecLnt in VecLnt_List ]
            MomF_List_pre = [self.MEpdiff.metric.flat(VecF) for VecF in VecF_List_pre]
        # print("VecF_List: ", VecF_List[-1].shape) #[120, 2, 32, 32]
        

        if self.ReparametrizeVectorField:
            VecF_List = self.regmetric.ReparametrizeVec(VecF_List_pre, MomF_List_pre)
            MomF_List = [self.MEpdiff.metric.flat(VecF) for VecF in VecF_List]
        else:
            VecF_List = VecF_List_pre
            MomF_List = MomF_List_pre

        # VecF_List = VecF_List_pre
        # MomF_List = MomF_List_pre

        # for i in range(len(VecF_List)):
        #     print(f"VecF_List_{i}: ", VecF_List[i].shape, VecF_List[i].max(), VecF_List[i].min(), VecF_List[i].mean())
        # assert 2>239
        phiinv_disp_list, phi_disp_list = self.MEpdiff.IntWithVList2(VecF_List)


        # time_start = time.time()
        # for i in range(100):
        #     self.MEpdiff.IntWithVList2(VecF_List)
        # time_end = time.time()
        # print("Time for IntWithVList2: ", (time_end - time_start)/100)  
        # #Time for Regsitration:  0.0039
        # assert 3>333

        

        dfmSrc = self.MEpdiff.transformer(self.src, phiinv_disp_list[-1])
        self.dfmSrc = dfmSrc

        # loss = F.mse_loss(dfmSrc, self.tar)
        # self.log("train_loss", loss)
        mse = self.mse_metric(dfmSrc, self.tar)

        VecF_List_Tensor = torch.stack(VecF_List, dim=0)
        MomF_List_Tensor = torch.stack(MomF_List, dim=0)

        reg = self.regmetric(VecF_List_Tensor, MomF_List_Tensor)
        loss = self.weight_mse * mse + self.weight_reg * reg

        return VecF_List, MomF_List, loss, mse, reg, phiinv_disp_list, phi_disp_list, VecF_List_pre, MomF_List_pre

    def common_step_(self, batch, batch_idx):
        # print(self.current_epoch)
        # print(self.global_step)
        # # self.current_epoch = 3995
        # # self.global_step = 59940
        # assert 2>129
        


        VecF_List, MomF_List, loss, mse, reg, _, _, _, _ = self.common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
                "train_mse": mse,
                "train_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    
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
        # return optimizer_unet
        # return optimizer_all
        # return [optimizer_unet, optimizer_nop]
        return [optimizer_unet, optimizer_nop, optimizer_all]


        # return optimizer_nop
        # return optimizer_unet


        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer_unet,  
        #     step_size=10,  # 每10个epoch调整一次
        #     gamma=0.1  # 学习率衰减因子
        # )
        
        # 返回多个优化器和调度器
        # return {
        #     "optimizer": [optimizer_unet, optimizer_nop],
        #     # "lr_scheduler": {
        #     #     "monitor": "val_loss",  # 监控验证集损失
        #     #     "frequency": 1  # 每个epoch更新一次
        #     # }
        # }

    def validation_step(self, batch, batch_idx):
        VecF_List, MomF_List, loss, mse, reg, _, _ , _, _ = self.common_step(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
                "val_mse": mse,
                "val_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    #test_step_LDDMMShoot
    def test_stepqqqq(self, batch, batch_idx):
        # common_step(self, batch, ReparametrizeVectorField=None):
        VecF_List_, _, _, _, _, _, _, _, _ = self.common_step(batch, batch_idx)
        phiinv_disp_list, phi_disp_list, VecF_List, MomF_List = self.MEpdiff.ShootWithV0(VecF_List_[0])

        # ## test time efficienty of ShootWithV0
        # time_start = time.time()
        # for i in range(100):
        #     self.MEpdiff.ShootWithV0(VecF_List_[0])
        # time_end = time.time()
        # print("Time for ShootWithV0: ", (time_end - time_start)/100)  
        # # Time for ShootWithV0:  0.03448579549789429
        # assert 3>333



        ## test time efficienty of ShootWithV0
        time_start = time.time()
        for i in range(100):
            phiinv_disp_list, phi_disp_list, VecF_List, MomF_List = self.MEpdiff.ShootWithV0(VecF_List_[0])
            lm.interp(self.src, phiinv_disp_list[-1])
        time_end = time.time()
        print("Time for ShootWithV0: ", (time_end - time_start)/100)  
        # Time for ShootWithV0:  0.03448579549789429
        assert 3>333




        dfmSrc_list = [lm.interp(self.src, phiinv_disp) for phiinv_disp in phiinv_disp_list]
        # dfmSrc_list = [self.MEpdiff.transformer(self.src, phiinv_disp) for phiinv_disp in phiinv_disp_list]
        if self.ndims == 3:
            dfmSrclabel = self.MEpdiff.transformer(self.srclabel, phiinv_disp_list[-1], mode='nearest')

        

        dfmSrc = self.MEpdiff.transformer(self.src, phiinv_disp_list[-1])
        self.dfmSrc = dfmSrc
        
        mse = self.mse_metric(dfmSrc, self.tar)

        VecF_List_Tensor = torch.stack(VecF_List, dim=0)
        MomF_List_Tensor = torch.stack(MomF_List, dim=0)

        reg = self.regmetric(VecF_List_Tensor, MomF_List_Tensor)
        loss = self.weight_mse * mse + self.weight_reg * reg

        self.log_dict(
            {
                "test_loss": loss,
                "test_mse": mse,
                "test_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {
            "dfmSrclabel": dfmSrclabel if self.ndims == 3 else None,
            "tarlabel": self.tarlabel if self.ndims == 3 else None,
            "srclabel": self.srclabel if self.ndims == 3 else None,

            # "phiinv": phiinv_disp_list[-1],

            "pred": self.dfmSrc,
            "src": self.src,
            "tar": self.tar,
            "loss": loss,
            "mse": mse,
            "reg": reg,
            "VecF_List": VecF_List,
            # "phiinv_disp_list": phiinv_disp_list,
            "phi_disp_list": phi_disp_list,
            "dfmSrc_list": dfmSrc_list,

            # "MomF_List": MomF_List,

            # "VecF_List_pre": VecF_List_pre,
            # "MomF_List_pre": MomF_List_pre,
        }
    

    ## Normal test_step
    def test_step(self, batch, batch_idx):
        # self.common_step_test_time(batch, batch_idx)
        # time_start = time.time()
        # for i in range(100):
        #     self.common_step(batch, batch_idx)
        #     # print(VecF_List_[0].shape)
        # time_end = time.time()
        # print("Time for Regsitration: ", (time_end - time_start)/100)  
        # # Time for Regsitration:  0.03296561241149902
        # assert 3>333

        

        # print(self.logger.log_dir) #/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_ROOT_DIR/2025training_mnist_v1.0/tensorboard/version_14
        VecF_List, MomF_List, loss, mse, reg, phiinv_disp_list, phi_disp_list, VecF_List_pre, MomF_List_pre = self.common_step(batch, batch_idx)
        dfmSrc_list = [self.MEpdiff.transformer(self.src, phiinv_disp) for phiinv_disp in phiinv_disp_list]

        if self.ndims == 3:
            dfmSrclabel = self.MEpdiff.transformer(self.srclabel, phiinv_disp_list[-1], mode='nearest')
        
        # assert 3>111
        self.log_dict(
            {
                "test_loss": loss,
                "test_mse": mse,
                "test_reg": reg,
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

            "pred": self.dfmSrc,
            "src": self.src,
            "tar": self.tar,
            "loss": loss,
            "mse": mse,
            "reg": reg,
            "VecF_List": VecF_List,
            "phiinv_disp_list": phiinv_disp_list,
            # "phiinv_disp_list": phi_disp_list,
            # "phi_disp_list": phi_disp_list,
            "dfmSrc_list": dfmSrc_list,

            "MomF_List": MomF_List,

            # "VecF_List_pre": VecF_List_pre,
            # "MomF_List_pre": MomF_List_pre,
            
        }
        return loss
    

    #test parallel
    def test_step22(self, batch, batch_idx):
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


        ## test time efficienty of ShootWithV0
        # time_start = time.time()
        # for i in range(100):
        #     self.common_step_parallel(ipt, batch_idx)
        #     # print(VecF_List_[0].shape)
        # time_end = time.time()
        # print("Time for ShootWithV0: ", (time_end - time_start)/100)  
        # # Time for ShootWithV0:  0.03448579549789429
        # assert 3>333



        VecF_List, MomF_List, loss, mse, reg, phiinv_disp_list, phi_disp_list = self.common_step_parallel(ipt, batch_idx)
        dfmSrc_list = [self.MEpdiff.transformer(self.src, phiinv_disp) for phiinv_disp in phiinv_disp_list]

        if self.ndims == 3:
            dfmSrclabel = self.MEpdiff.transformer(self.srclabel, phiinv_disp_list[-1], mode='nearest')

        self.log_dict(
            {
                "test_loss": loss,
                "test_mse": mse,
                "test_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # torch.cuda.empty_cache()
        return {
            "dfmSrclabel": dfmSrclabel if self.ndims == 3 else None,
            # "tarlabel": self.tarlabel if self.ndims == 3 else None,
            # "srclabel": self.srclabel if self.ndims == 3 else None,

            # "phiinv": phiinv_disp_list[-1],

            "pred": self.dfmSrc,
            "src": self.src,
            "tar": self.tar,
            "loss": loss,
            "mse": mse,
            "reg": reg,
            "VecF_List": VecF_List,
            "phiinv_disp_list": phiinv_disp_list,
            # "phi_disp_list": phi_disp_list,
            "dfmSrc_list": dfmSrc_list,

            # "MomF_List": MomF_List,

            # "VecF_List_pre": VecF_List_pre,
            # "MomF_List_pre": MomF_List_pre,
            
        }
        return loss



    def test_epoch_end1(self, outputs):
        plot_norm_list_gpu(outputs, self.regmetric, self.mse_metric)
   
    def test_epoch_end(self, outputs):
        jacobian_det_for_all(outputs, ongpuGrad=False, TSteps=self.TSteps, dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), inshape=self.inshape, regW=self.weight_reg)
        # hdorff_coefficient_for_brain(outputs, ongpuGrad=False, TSteps=self.TSteps, dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), regW=self.weight_reg, hdorff=True)
        dice_coefficient_for_brain(outputs, ongpuGrad=False, TSteps=self.TSteps, dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), regW=self.weight_reg, hdorff=False)
        
        # plot_dfm_process(outputs, ongpuGrad=False, mse_metric=self.mse_metric, TSteps=self.TSteps, showTSteps=[0,1,5,7,9], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES")
        # plot_dfm_process(outputs, ongpuGrad=False, mse_metric=self.mse_metric, TSteps=self.TSteps, showTSteps=[0,1,2,3,4], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES")
        plot_dfm_process(outputs, ongpuGrad=False, mse_metric=self.mse_metric, TSteps=self.TSteps, showTSteps=[4], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES",saveRes=False)

        # plot_norm_list_gpu(outputs, self.regmetric, self.mse_metric, TSteps=self.TSteps, dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES")

    def on_load_checkpoint(self, checkpoint):
        print(self.current_epoch, self.global_step) #3995 59940
        print(checkpoint.keys())
        print(checkpoint['epoch'], checkpoint['global_step'])
        print("Checkpoint is being loaded!")
        # assert 2>128





class SVF(pl.LightningModule):
    def __init__(self, unet_config, reg_config, weight_reg, unet_lr, templateIDX=-1,
                *args, **kwargs):
        super().__init__()
        self.unet = instantiate_from_config(unet_config)
        self.regmetric = instantiate_from_config(reg_config)
        self.mse_metric = torchmetrics.MeanSquaredError()
        self.weight_reg = weight_reg
        self.unet_lr = unet_lr
        self.ndims = unet_config['params']['ndims']
        self.inshape = kwargs['inshape']
        self.TSteps = kwargs['TSteps']
        self.templateIDX = templateIDX
        
        # self.weight_reg = kwargs['weight_reg']
        assert unet_config['params']['ndims'] == reg_config['params']['ndims'], "set of ndims should be the same in unet and regmetric"

        """ 
        take: velocity_0  
        return: phiinv_disp and list_of_phiinv_disp
        """
        self.integrate = VecInt(**{
            'inshape': kwargs['inshape'],
            'TSteps': kwargs['TSteps']
        })

        """ 
        take: src and phiinv_disp
        return: defromed_src and phiinv
        """
        self.transformer = SpatialTransformer(**{
            'inshape': kwargs['inshape'],
            'mode': 'bilinear'
        })
    
    def encoder_step(self, x):
        VecLnt = self.unet.exec_encoder(x)  #V in Latent space
        return VecLnt
    
    def decoder_step(self, x):
        VecF = self.unet.exec_decoder(x) #V in full space
        return VecF
    
    def common_step_stage1_train(self, batch, batch_idx):
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
        VecF = self.decoder_step(VecLnt)

        velocity_0 = VecF

        phiinv_disp_list = self.integrate(velocity_0)
        dfmSrc, phiinv =  self.transformer(self.src, phiinv_disp_list[-1])
        print(self.src.shape, self.tar.shape, dfmSrc.shape, phiinv.shape)
        assert 2>222

        return velocity_0, dfmSrc, phiinv, None
    
    def common_step_test_time(self, batch, batch_idx):
        TN =100
        time_start = time.time()
        for i in range(TN):

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
            VecF = self.decoder_step(VecLnt)

            velocity_0 = VecF

            phiinv_disp_list = self.integrate(velocity_0)
            dfmSrc, phiinv =  self.transformer(self.src, phiinv_disp_list[-1])

        time_end = time.time()
        print("Time for Registration LagoM: ", (time_end - time_start)/TN)  
        assert 3>222
       
    def common_step_stage1_test(self, batch, batch_idx):
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
        VecF = self.decoder_step(VecLnt)

        velocity_0 = VecF
        velocity_Neg = -velocity_0

        ## test time efficienty of IntWithVList2
        # time_start = time.time()
        # for i in range(100):
        #     phiinv_disp_list = self.integrate(velocity_0)
        # time_end = time.time()
        # print("Time for nop_step: ", (time_end - time_start)/100)  
        # #Time for nop_step:  0.004525275230407715
        # #Time for nop_step:  0.003625168800354004
        # assert 2>239

        # print(self.src.shape)
        # assert 2>222

        phi_disp_list = self.integrate(velocity_Neg) 
        _, phi =  self.transformer(self.src, phi_disp_list[-1])

       

        phiinv_disp_list = self.integrate(velocity_0)
        dfmSrc, phiinv =  self.transformer(self.src, phiinv_disp_list[-1])

        if self.ndims == 3:
            dfmSrclabel, _ = self.transformer(self.srclabel, phiinv_disp_list[-1], mode='nearest')
            return velocity_0, dfmSrc, phiinv, dfmSrclabel, phi
        else:
            return velocity_0, dfmSrc, phiinv, None, phi
            # return velocity_0, dfmSrc, phiinv, None, torch.zeros_like(phi)

    def common_step(self, batch, batch_idx):
        #print current batch index and epoch index
        
        # self.src = x[:,0:1,...]
        # self.tar = x[:,1:2,...]
        # velocity_0 = self.unet(x)
        # return velocity_0
        velocity_0, dfmSrc, phiinv, _ = self.common_step_stage1_train(batch, batch_idx)
        mse = self.mse_metric(dfmSrc, self.tar)

        VecF_List_Tensor = velocity_0.unsqueeze(0)
        # print(VecF_List_Tensor.shape) #torch.Size([1, 12, 3, 128, 128, 128])
        # assert 1>123
        reg = self.regmetric(VecF_List_Tensor, VecF_List_Tensor)


        
        loss = mse + self.weight_reg * reg

        #lddmm
        # self.weight_mse = weight_mse / (weight_sigma**2)
        # self.weight_reg = weight_reg
        # los_regis = self.weight_mse * mse + self.weight_reg * reg

        # loss = 0.5/(0.03**2) * mse + 0.15 * reg
        
        
        return loss, mse, reg
    
    def training_step(self, batch, batch_idx):
        loss, mse, reg = self.common_step(batch, batch_idx)
        # print("optimizer_idx ",optimizer_idx, loss, mse, reg)
        self.log_dict(
            {
                "train_loss": loss,
                "train_mse": mse,
                "train_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # print("success")
        return loss
        
        # return {
        #         "loss": loss,
        #         "train_mse": mse,
        #         "train_reg": reg,
        #     }

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
        # print(f"Epoch {self.current_epoch} started at {time.strftime('%H:%M:%S')}")

    def on_train_epoch_end(self):
        if self.epoch_start_time is not None:
            elapsed_time = time.time() - self.epoch_start_time
            print(f"\n\n Epoch {self.current_epoch} ended. Time elapsed: {elapsed_time:.2f} seconds. \n\n")

    def to_numpy(self,arr):
        if isinstance(arr, np.ndarray):
            return arr
        try:
            from pycuda import gpuarray
            if isinstance(arr, gpuarray.GPUArray):
                return arr.get()
        except ImportError:
            pass
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                return arr.cpu().numpy()
        except ImportError:
            pass

        raise Exception(f"Cannot convert type {type(arr)} to numpy.ndarray.")
    
    def Mgridplot(self,u, ax, Nx=64, Ny=64, displacement=True, color='red', IMG=None, **kwargs):
        """Given a displacement field, plot a displaced grid"""
        u = self.to_numpy(u)
        """Given a displacement field, plot a displaced grid"""
        

        assert u.shape[0] == 1, "Only send one deformation at a time"
    
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.axis('off')  # Remove axis
        ax.set_adjustable('box')  # Ensure the aspect ratio is consistent for each plot
        

        if IMG is not None:
            # ax.imshow(IMG, cmap='gray', alpha=0.5)
            ax.imshow(IMG, cmap='gray')


        if Nx is None:
            Nx = u.shape[2]
        if Ny is None:
            Ny = u.shape[3]
        # downsample displacements
        h = np.copy(u[0,:,::u.shape[2]//Nx, ::u.shape[3]//Ny])

        # now reset to actual Nx Ny that we achieved
        Nx = h.shape[1]
        Ny = h.shape[2]
        # adjust displacements for downsampling
        h[0,...] /= float(u.shape[2])/Nx
        h[1,...] /= float(u.shape[3])/Ny

        if displacement: # add identity
            '''
                h[0]: 
            '''
            h[0,...] += np.arange(Nx).reshape((Nx,1))  #h[0]:  (118, 109)  add element: 118*1
            h[1,...] += np.arange(Ny).reshape((1,Ny))

        # put back into original index space
        h[0,...] *= float(u.shape[2])/Nx
        h[1,...] *= float(u.shape[3])/Ny

        from scipy.ndimage import zoom
        upscale = 4
        h_smooth = zoom(h, (1, upscale, upscale), order=3)  # cubic interpolation
        h = np.copy(h_smooth)

        # create a meshgrid of locations
        for i in range(h.shape[1]):
            # ax.plot( h[0,i,:], h[1,i,:], color=color, linewidth=1.0, **kwargs)
            ax.plot(h[1, i, :], h[0, i, :], color=color, linewidth=1.0, **kwargs)
        for i in range(h.shape[2]):
            # ax.plot(h[0,:,i], h[1,:,i],  color=color, linewidth=1.0, **kwargs)
            ax.plot(h[1, :, i],h[0, :, i] , color=color, linewidth=1.0, **kwargs)
            
        # Adjust axis limits to remove unnecessary whitespace
        ax.set_xlim(h[1].min(), h[1].max())  
        ax.set_ylim(h[0].min(), h[0].max())  
        ax.margins(0)  
        ax.set_adjustable('datalim')
        # ax.set_xlim(ax.get_ylim())  # Swap x and y limits
        # ax.set_ylim(ax.get_xlim())  # Swap x and y limits
        ax.set_aspect('equal')
        ax.invert_yaxis()

    def validation_step(self, batch, batch_idx):
        loss, mse, reg = self.common_step(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
                "val_mse": mse,
                "val_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        # self.common_step_test_time(batch, batch_idx)
        
        # time_start = time.time()
        # for i in range(100):
        #     velocity_0, dfmSrc, phiinv, dfmSrclabel, phi = self.common_step_stage1_test(batch, batch_idx)
        # end_time = time.time()
        # print("Time for Regsitration: ", (end_time - time_start)/100)  #Time for Regsitration:  0.010874528884887696
        # assert 3>333
        velocity_0, dfmSrc, phiinv, dfmSrclabel, phi = self.common_step_stage1_test(batch, batch_idx)

        mse = self.mse_metric(dfmSrc, self.tar)
        VecF_List_Tensor = velocity_0.unsqueeze(0)
        reg = self.regmetric(VecF_List_Tensor, VecF_List_Tensor)

        loss = mse + self.weight_reg * reg

        self.log_dict(
            {
                "test_loss": loss,
                "test_mse": mse,
                "test_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {
            "dfmSrclabel": dfmSrclabel if self.ndims == 3 else None,
            "tarlabel": self.tarlabel if self.ndims == 3 else None,
            "srclabel": self.srclabel if self.ndims == 3 else None,

            "phiinv": phi,
            "phi": phi,


            "pred": dfmSrc,
            "src": self.src,
            "tar": self.tar,
            "mse": mse,
            "velocity_0": velocity_0,
            "reg": reg
        }
    
    def test_epoch_end111(self, outputs):
        jacobian_det_for_all(outputs, ongpuGrad=False, TSteps=self.TSteps, templateIDX=self.templateIDX, phiinv=False,\
                             dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), inshape=self.inshape, regW=self.weight_reg)
        hdorff_coefficient_for_brain(outputs, ongpuGrad=False, TSteps=self.TSteps, templateIDX=self.templateIDX,\
                                      dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), regW=self.weight_reg)
        

        # jacobian_det_for_all(outputs, ongpuGrad=False, TSteps=self.TSteps,  templateIDX=self.templateIDX, \
        #                      dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), inshape=self.inshape, regW=self.weight_reg)
        # dice_coefficient_for_brain(outputs, ongpuGrad=False, TSteps=self.TSteps,  templateIDX=self.templateIDX, \
        #                            dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), regW=self.weight_reg, hdorff=False)
         
    def clear_labels(self, labels, type="oasis1"):
        masks = {
            "oasis1": {
                "maskL": [1, 2, 3, 5, 6, 7, 8, 9, 14, 13],
                "maskR": [20, 21, 22, 24, 25, 26, 27, 28, 30, 13],
                "maskT": [1, 2, 3, 5, 6, 7, 8, 9, 14, 13, 20, 21, 22, 24, 25, 26, 27, 28, 30],
            },

            "oasis3": {
                "maskL": [2, 3, 4, 7, 8, 10, 11, 12, 17, 16],
                "maskR": [41, 42, 43, 46, 47, 49, 50, 51, 53, 16],
                "maskT": [2, 3, 4, 7, 8, 10, 11, 12, 17, 16, 41, 42, 43, 46, 47, 49, 50, 51, 53],
            }
        }


        maskL = masks[type]["maskL"]
        maskR = masks[type]["maskR"]
        maskT = masks[type]["maskT"]
        transf_label = labels.copy()
        # 处理 maskT
        mask = np.isin(transf_label, np.array(maskT))
        transf_label[~mask] = 0
        # 遍历 maskL 和 maskR
        for idx in range(len(maskL)):
            specific_values = np.array([maskL[idx], maskR[idx]])
            # 处理 transf_label
            mask = np.isin(transf_label, specific_values)
            transf_label[mask] = idx+1
        return transf_label

    def test_epoch_end(self, outputs):
        # dir=os.path.dirname(os.path.dirname(self.logger.log_dir))
        import matplotlib.pyplot as plt
        # # Collect data from all batches
        # # test_losses = [output["test_loss"] for output in outputs]
        srcs = torch.cat([output["src"] for output in outputs]).cpu().numpy()
        tars = torch.cat([output["tar"] for output in outputs]).cpu().numpy()
        preds = torch.cat([output["pred"] for output in outputs]).cpu().numpy()
        phiinvs = torch.cat([output["phiinv"] for output in outputs]).cpu().numpy()
        # phis = torch.cat([output["phi"] for output in outputs]).cpu().numpy()
        # dfmSrclabels = torch.cat([output["dfmSrclabel"] for output in outputs]).cpu().numpy()  #(100, 1, 128, 128, 128)

        # #save phiinvs as npy
        # np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_MELBA/000MELBA_SEG_S0438_T0441/phi_disp_VM_S0438_T0441.npy", phiinvs)
        # assert 1>999



        # #save dfmSrc as .nii.gz
        # import SimpleITK as sitk
        # preds = torch.cat([output["pred"] for output in outputs])
        # preds = F.interpolate(preds, scale_factor=2, mode='trilinear')
        # dfmSrcimg = preds[0,0].cpu().detach().numpy(); dfmSrcimg = sitk.GetImageFromArray(dfmSrcimg)
        # sitk.WriteImage(dfmSrcimg, "/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_MELBA/000MELBA_SEG_S0438_T0441/VM-dfm438.nii.gz")
        # print(preds[0,0].shape)
        # assert 1>222



        # #save phiinvs as npy
        # np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_MELBA/000MELBA_SEG_S0438_T0441/phiinv_disp_VM_S0438_T0441.npy", phiinvs)
        # assert 1>908

        # import SimpleITK as sitk
        # predimg = sitk.GetImageFromArray(preds[0, 0])
        # dfmSrclabels = self.clear_labels(dfmSrclabels)
        # dfmSrclabelimg = sitk.GetImageFromArray(dfmSrclabels[0, 0])
        # sitk.WriteImage(predimg, f"{dir}/RES/000pred.nii.gz")
        # sitk.WriteImage(dfmSrclabelimg, f"{dir}/RES/000dfmSrclabel.nii.gz")
        # # assert 1>908

        # print(phiinvs.shape, np.max(phiinvs), np.min(phiinvs), srcs.shape)  #(1, 3, 128, 128, 128) 5.0862494 -4.0275164
        # assert 1>908


        velocity_0s = torch.cat([output["velocity_0"] for output in outputs]).cpu().numpy()
        mses = [output["mse"] for output in outputs]
        regs = [output["reg"] for output in outputs]
        # print(mses, regs)

        # preds = np.load("/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_MELBA/000MELBA_MixedOasis_TM_V3/RES/DfmSrc.npy")
        # phiinvs = np.load("/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_MELBA/000MELBA_MixedOasis_TM_V3/RES/phi_disp.npy")
        # print(srcs.shape,        tars.shape,             preds.shape,            velocity_0s.shape,        phiinvs.shape)
        # #(100, 1, 32, 32)        (100, 1, 32, 32)        (100, 1, 32, 32)
        # (100, 1, 128, 128, 128) (100, 1, 128, 128, 128) (100, 1, 128, 128, 128) (100, 3, 128, 128, 128)
       
    
        disp = torch.cat([output["phiinv"] for output in outputs]).cpu().numpy()
        if len(disp.shape) == 5: #3D   1, 3, 128, 128, 128
            disp = np.transpose(disp, (0, 2, 3, 4, 1))  #100, 128, 128, 128, 3
        elif len(disp.shape) == 4: #2D
            disp = np.transpose(disp, (0, 2, 3, 1))  #100, 128, 128, 3
        
        dets = []
        for idx in range(disp.shape[0]):
            det = jacobian_determinant(disp[idx], inshape=self.inshape)
            print(idx, "   ", det.shape)
            dets.append(det)
        
        #expend a dimension
        dets = np.expand_dims(dets, axis=1)  #100, 128, 128, 128
        dets = np.stack(dets)  #100, 1, 128, 128, 128
        
        
        
        
        imagesize = srcs.shape[-1] #128
        select_idx=int(imagesize/2-3*(imagesize/64))
        select_idx = 59




        if len(srcs.shape) == 5: #3D
            vie = 0
            srcs = srcs[:, :, select_idx]
            tars = tars[:, :, select_idx]
            preds = preds[:, :, select_idx]
            JabDetSlice = dets[:, :, select_idx]
            phiinvs = phiinvs[:, 1:, select_idx] #5,2,128,128

            """ vie = 1
            srcs = srcs[:, :, :, select_idx]
            tars = tars[:, :, :, select_idx]
            preds = preds[:, :, :, select_idx]
            JabDetSlice = dets[:, :, :, select_idx] 
            phiinvs = phiinvs[:, ::2, :, select_idx] """ 

            """ vie = 2
            srcs = srcs[:, :, :, :, select_idx]
            tars = tars[:, :, :, :, select_idx]
            preds = preds[:, :, :, :, select_idx]
            JabDetSlice = dets[:, :, :, :, select_idx]
            phiinvs = phiinvs[:, 0:2, :, :, select_idx] """

            print("phiinvs",phiinvs.shape)
            np.save(f"{dir}/RES/0-JabDetSlice_{vie}.npy", JabDetSlice)


        batch_size = min(20, srcs.shape[0])
        batch_size_2 = max(2, batch_size)
        # 创建图形
        
        basesize = 5
        nrows_list=["src", "tar",'dfmSrc','phiinv']
        fig, axes = plt.subplots(batch_size_2, len(nrows_list), figsize=(len(nrows_list)*basesize, batch_size*basesize ))  # 每行 len(nrows_list) 个图，batch_size 行
       
        imgsize = int(srcs.shape[-1]/3); imgsize = max(8, imgsize)
        for i in range(batch_size):
            axes_y = axes[i]

            # 绘制源图像
            axes_y[0].imshow(srcs[i, 0], cmap="gray", vmin=0, vmax=1)  # 选择第 0 通道
            axes_y[0].axis("off")
            axes_y[0].set_title(f"{i+1} Source")

            # 绘制目标图像
            axes_y[1].imshow(tars[i, 0], cmap="gray", vmin=0, vmax=1)  # 选择第 0 通道
            axes_y[1].axis("off")
            axes_y[1].set_title("Target")

            # 绘制预测图像
            axes_y[2].imshow(preds[i, 0], cmap="gray", vmin=0, vmax=1)  # 选择第 0 通道
            axes_y[2].axis("off")
            axes_y[2].set_title("Prediction")
            
            print(phiinvs[i:i+1].shape)
            # self.Mgridplot(phiinvs[i:i+1], axes_y[3],  imgsize, imgsize, displacement = True, IMG=srcs[i, 0])
            self.Mgridplot(phiinvs[i:i+1], axes_y[3],  imgsize, imgsize, displacement = True)

        # 调整子图布局
        # plt.tight_layout()
        # savep = f"{dir}/RES/0-visualization_{vie}.png"
        # print(f"save the plot in {savep}")
        # plt.savefig(savep,bbox_inches='tight',dpi=200)

        # np.save(f"{dir}/RES/0-JabDetSlice_{vie}.npy", JabDetSlice)
        plt.show()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.unet_lr)






class SVF2(pl.LightningModule):
    def __init__(self, unet_config, reg_config, weight_reg, weight_mse, weight_sigma, unet_lr,
                *args, **kwargs):
        super().__init__()
        self.unet = instantiate_from_config(unet_config)
        self.regmetric = instantiate_from_config(reg_config)
        self.mse_metric = torchmetrics.MeanSquaredError()
        self.unet_lr = unet_lr
        self.ndims = unet_config['params']['ndims']
        self.inshape = kwargs['inshape']
        self.TSteps = kwargs['TSteps']
        # self.weight_reg = kwargs['weight_reg']
        assert unet_config['params']['ndims'] == reg_config['params']['ndims'], "set of ndims should be the same in unet and regmetric"


        self.weight_mse = weight_mse / (weight_sigma**2)
        self.weight_reg = weight_reg


        """ 
        take: velocity_0  
        return: phiinv_disp and list_of_phiinv_disp
        """
        self.integrate = VecInt(**{
            'inshape': kwargs['inshape'],
            'TSteps': kwargs['TSteps']
        })

        """ 
        take: src and phiinv_disp
        return: defromed_src and phiinv
        """
        self.transformer = SpatialTransformer(**{
            'inshape': kwargs['inshape'],
            'mode': 'bilinear'
        })
    
    def encoder_step(self, x):
        VecLnt = self.unet.exec_encoder(x)  #V in Latent space
        return VecLnt
    
    def decoder_step(self, x):
        VecF = self.unet.exec_decoder(x) #V in full space
        return VecF
    
    def common_step_stage1_train(self, batch, batch_idx):
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
        VecF = self.decoder_step(VecLnt)

        velocity_0 = VecF

        phiinv_disp_list = self.integrate(velocity_0)
        dfmSrc, phiinv =  self.transformer(self.src, phiinv_disp_list[-1])

        return velocity_0, dfmSrc, phiinv, None
    
    def common_step_test_time(self, batch, batch_idx):
        TN =100
        time_start = time.time()
        for i in range(TN):

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
            VecF = self.decoder_step(VecLnt)

            velocity_0 = VecF

            phiinv_disp_list = self.integrate(velocity_0)
            dfmSrc, phiinv =  self.transformer(self.src, phiinv_disp_list[-1])

        time_end = time.time()
        print("Time for Registration LagoM: ", (time_end - time_start)/TN)  
        assert 3>222
       
    def common_step_stage1_test(self, batch, batch_idx):
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
        VecF = self.decoder_step(VecLnt)

        velocity_0 = VecF

        ## test time efficienty of IntWithVList2
        # time_start = time.time()
        # for i in range(100):
        #     phiinv_disp_list = self.integrate(velocity_0)
        # time_end = time.time()
        # print("Time for nop_step: ", (time_end - time_start)/100)  
        # #Time for nop_step:  0.004525275230407715
        # #Time for nop_step:  0.003625168800354004
        # assert 2>239

        phiinv_disp_list = self.integrate(velocity_0)
        dfmSrc, phiinv =  self.transformer(self.src, phiinv_disp_list[-1])

        if self.ndims == 3:
            dfmSrclabel, _ = self.transformer(self.srclabel, phiinv_disp_list[-1], mode='nearest')
            return velocity_0, dfmSrc, phiinv, dfmSrclabel
        else:
            return velocity_0, dfmSrc, phiinv, None

    def common_step(self, batch, batch_idx):
        #print current batch index and epoch index
        
        # self.src = x[:,0:1,...]
        # self.tar = x[:,1:2,...]
        # velocity_0 = self.unet(x)
        # return velocity_0
        velocity_0, dfmSrc, phiinv, _ = self.common_step_stage1_train(batch, batch_idx)
        mse = self.mse_metric(dfmSrc, self.tar)

        VecF_List_Tensor = velocity_0.unsqueeze(0)
        # print(VecF_List_Tensor.shape) #torch.Size([1, 12, 3, 128, 128, 128])
        # assert 1>123
        reg = self.regmetric(VecF_List_Tensor, VecF_List_Tensor)


        
      
        loss = self.weight_mse * mse + self.weight_reg * reg

        return loss, mse, reg
    
    def training_step(self, batch, batch_idx):
        loss, mse, reg = self.common_step(batch, batch_idx)
        # print("optimizer_idx ",optimizer_idx, loss, mse, reg)
        self.log_dict(
            {
                "train_loss": loss,
                "train_mse": mse,
                "train_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # print("success")
        return loss
        
        # return {
        #         "loss": loss,
        #         "train_mse": mse,
        #         "train_reg": reg,
        #     }

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
        # print(f"Epoch {self.current_epoch} started at {time.strftime('%H:%M:%S')}")
    def on_train_epoch_end(self):
        if self.epoch_start_time is not None:
            elapsed_time = time.time() - self.epoch_start_time
            print(f"\n\n Epoch {self.current_epoch} ended. Time elapsed: {elapsed_time:.2f} seconds. \n\n")

    def to_numpy(self,arr):
        if isinstance(arr, np.ndarray):
            return arr
        try:
            from pycuda import gpuarray
            if isinstance(arr, gpuarray.GPUArray):
                return arr.get()
        except ImportError:
            pass
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                return arr.cpu().numpy()
        except ImportError:
            pass

        raise Exception(f"Cannot convert type {type(arr)} to numpy.ndarray.")
    

    
    def Mgridplot(self,u, ax, Nx=64, Ny=64, displacement=True, color='red', **kwargs):
        """Given a displacement field, plot a displaced grid"""
        u = self.to_numpy(u)
        assert u.shape[0] == 1, "Only send one deformation at a time"
    
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.axis('off')  # Remove axis
        ax.set_adjustable('box')  # Ensure the aspect ratio is consistent for each plot
        

        if Nx is None:
            Nx = u.shape[2]
        if Ny is None:
            Ny = u.shape[3]
        # downsample displacements
        h = np.copy(u[0,:,::u.shape[2]//Nx, ::u.shape[3]//Ny])

        # now reset to actual Nx Ny that we achieved
        Nx = h.shape[1]
        Ny = h.shape[2]
        
        # adjust displacements for downsampling
        h[0,...] /= float(u.shape[2])/Nx
        h[1,...] /= float(u.shape[3])/Ny

        if displacement: # add identity
            '''
                h[0]: 
            '''
            h[0,...] += np.arange(Nx).reshape((Nx,1))  #h[0]:  (118, 109)  add element: 118*1
            h[1,...] += np.arange(Ny).reshape((1,Ny))

        # put back into original index space
        h[0,...] *= float(u.shape[2])/Nx
        h[1,...] *= float(u.shape[3])/Ny
        # create a meshgrid of locations
        for i in range(h.shape[1]):
            # ax.plot( h[0,i,:], h[1,i,:], color=color, linewidth=0.1, **kwargs)
            ax.plot(h[1, i, :], h[0, i, :], color=color, linewidth=1.0, **kwargs)
        for i in range(h.shape[2]):
            # ax.plot(h[0,:,i], h[1,:,i],  color=color, linewidth=0.1, **kwargs)
            ax.plot(h[1, :, i],h[0, :, i] , color=color, linewidth=1.0, **kwargs)
            
        ax.set_aspect('equal')
        ax.invert_yaxis()
        # ax.set_xlim(ax.get_ylim())  # Swap x and y limits
        # ax.set_ylim(ax.get_xlim())  # Swap x and y limits

    def validation_step(self, batch, batch_idx):
        loss, mse, reg = self.common_step(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
                "val_mse": mse,
                "val_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        # self.common_step_test_time(batch, batch_idx)
        velocity_0, dfmSrc, phiinv, dfmSrclabel = self.common_step_stage1_test(batch, batch_idx)

        mse = self.mse_metric(dfmSrc, self.tar)
        VecF_List_Tensor = velocity_0.unsqueeze(0)
        reg = self.regmetric(VecF_List_Tensor, VecF_List_Tensor)

        loss = mse + self.weight_reg * reg

        self.log_dict(
            {
                "test_loss": loss,
                "test_mse": mse,
                "test_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {
            "dfmSrclabel": dfmSrclabel if self.ndims == 3 else None,
            "tarlabel": self.tarlabel if self.ndims == 3 else None,
            "srclabel": self.srclabel if self.ndims == 3 else None,

            "phiinv": phiinv,

            "pred": dfmSrc,
            "src": self.src,
            "tar": self.tar,
            "mse": mse,
            "velocity_0": velocity_0,
            "reg": reg
        }
    
    def test_epoch_end(self, outputs):
        jacobian_det_for_all(outputs, ongpuGrad=False, TSteps=self.TSteps, dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), inshape=self.inshape, regW=self.weight_reg)
        dice_coefficient_for_brain(outputs, ongpuGrad=False, TSteps=self.TSteps, dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), regW=self.weight_reg, hdorff=False)
        

    def test_epoch_end222(self, outputs):
        import matplotlib.pyplot as plt
        # Collect data from all batches
        # test_losses = [output["test_loss"] for output in outputs]
        srcs = torch.cat([output["src"] for output in outputs]).cpu().numpy()
        tars = torch.cat([output["tar"] for output in outputs]).cpu().numpy()
        preds = torch.cat([output["pred"] for output in outputs]).cpu().numpy()
        phiinvs = torch.cat([output["phiinv"] for output in outputs]).cpu().numpy()
        velocity_0s = torch.cat([output["velocity_0"] for output in outputs]).cpu().numpy()
        mses = [output["mse"] for output in outputs]
        regs = [output["reg"] for output in outputs]
        # print(mses, regs)
        print(srcs.shape,        tars.shape,             preds.shape,            velocity_0s.shape,        phiinvs.shape)
        # #(100, 1, 32, 32)        (100, 1, 32, 32)        (100, 1, 32, 32)
        # (100, 1, 128, 128, 128) (100, 1, 128, 128, 128) (100, 1, 128, 128, 128) (100, 3, 128, 128, 128)
        if len(srcs.shape) == 5: #3D
            select_idx = 64
            srcs = srcs[:, :, select_idx]
            tars = tars[:, :, select_idx]
            preds = preds[:, :, select_idx]
            phiinvs = phiinvs[:, 1:, select_idx] #5,2,128,128
            print("phiinvs",phiinvs.shape)

        batch_size = min(20, srcs.shape[0])
        imgsize = int(srcs.shape[-1]/3); imgsize = max(32, imgsize)
        # 创建图形
        
        basesize = 5
        nrows_list=["src", "tar",'dfmSrc','phiinv']
        fig, axes = plt.subplots(batch_size, len(nrows_list), figsize=(len(nrows_list)*basesize, batch_size*basesize ))  # 每行 len(nrows_list) 个图，batch_size 行
        # print("axes: ", axes.shape)  #axes:  (5, 4)
        # assert 2>239
        
        for i in range(batch_size):
            axes_y = axes[i]

            # 绘制源图像
            axes_y[0].imshow(srcs[i, 0], cmap="gray", vmin=0, vmax=1)  # 选择第 0 通道
            axes_y[0].axis("off")
            axes_y[0].set_title("Source")

            # 绘制目标图像
            axes_y[1].imshow(tars[i, 0], cmap="gray", vmin=0, vmax=1)  # 选择第 0 通道
            axes_y[1].axis("off")
            axes_y[1].set_title("Target")

            # 绘制预测图像
            axes_y[2].imshow(preds[i, 0], cmap="gray", vmin=0, vmax=1)  # 选择第 0 通道
            axes_y[2].axis("off")
            axes_y[2].set_title("Prediction")
            
            print(phiinvs[i:i+1].shape)
            self.Mgridplot(phiinvs[i:i+1], axes_y[3],  imgsize, imgsize, displacement = True)

        # 调整子图布局
        plt.tight_layout()
        plt.show()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.unet_lr)






class LDDMM(pl.LightningModule):
    def __init__(self, unet_config, reg_config, weight_mse, weight_reg, weight_sigma, unet_lr, templateIDX=-1,
                 *args, **kwargs):
        super().__init__()
        self.unet = instantiate_from_config(unet_config)
        self.regmetric = instantiate_from_config(reg_config)
        self.mse_metric = torchmetrics.MeanSquaredError()
        self.weight_reg = weight_reg
        self.unet_lr = unet_lr
        self.ndims = unet_config['params']['ndims']
        self.img_size = kwargs['inshape'][0]
        self.TSteps = kwargs['TSteps']
        self.inshape = kwargs['inshape']

        self.weight_mse = weight_mse / (weight_sigma**2)
        self.unet_lr = unet_lr

        self.templateIDX = templateIDX

        self.PredictMomnetum = kwargs['PredictMomnetum'] if 'PredictMomnetum' in kwargs else False
        
        
        # self.fluid_params = [alpha, 0, gamma]
        # self.metric = lm.FluidMetric(self.fluid_params)

        self.MEpdiff = Epdiff(**{
            'inshape': kwargs['inshape'],
            'alpha': kwargs['alpha'], #alpha,
            'gamma': kwargs['gamma'], #gamma,
            'TSteps': kwargs['TSteps']
        })
        self.grid = self.MEpdiff.identity(kwargs['inshape'])

    def encoder_step(self, x):
        VecLnt = self.unet.exec_encoder(x)  #V in Latent space
        return VecLnt
    
    def decoder_step(self, x):
        VecF = self.unet.exec_decoder(x) #V in full space
        return VecF
    
    def common_step_stage1(self, batch, batch_idx):
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
        # print("ipt: ", ipt.shape)  #ipt:  torch.Size([120, 2, 128, 128, 128])


        VecLnt = self.encoder_step(ipt)
        VecF = self.decoder_step(VecLnt)
        velocity_0 = VecF

        return velocity_0
    
    def common_step(self, batch, batch_idx):
        # print(self.PredictMomnetum)
        # assert 3>123

        if self.PredictMomnetum:
            momentum_0 = self.common_step_stage1(batch, batch_idx)
            velocity_0 = self.MEpdiff.metric.sharp(momentum_0)
        else:
            velocity_0 = self.common_step_stage1(batch, batch_idx)

        phiinv_disp_list, phi_disp_list, v_list, m_list = self.MEpdiff.ShootWithV0(velocity_0)
        dfmSrc = lm.interp(self.src, phiinv_disp_list[-1])

    

        mse = self.mse_metric(dfmSrc, self.tar)

        VecF_List_Tensor = velocity_0.unsqueeze(0)
        MomF_List_Tensor = m_list[0].unsqueeze(0)
        reg = self.regmetric(VecF_List_Tensor, MomF_List_Tensor)

        loss = self.weight_mse * mse + self.weight_reg * reg
        return loss, mse, reg
    
    def training_step(self, batch, batch_idx):
        loss, mse, reg = self.common_step(batch, batch_idx)
        # print("optimizer_idx ",optimizer_idx, loss, mse, reg)
        self.log_dict(
            {
                "train_loss": loss,
                "train_mse": mse,
                "train_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # print("success")
        return loss
    
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if self.epoch_start_time is not None:
            elapsed_time = time.time() - self.epoch_start_time
            print(f"\n\n Epoch {self.current_epoch} ended. Time elapsed: {elapsed_time:.2f} seconds. \n\n")


    def validation_step(self, batch, batch_idx):
        loss, mse, reg = self.common_step(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
                "val_mse": mse,
                "val_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss


    #LgMo test
    def test_step(self, batch, batch_idx):
        # time_start = time.time()
        # for i in range(100):
        #     velocity_0 = self.common_step_stage1(batch, batch_idx)
        #     phiinv_disp_list, phi_disp_list, v_list, m_list = self.MEpdiff.ShootWithV0(velocity_0)
        #     dfmSrc = lm.interp(self.src, phiinv_disp_list[-1])
        # time_end = time.time()
        # print("Time for Regsitration: ", (time_end - time_start)/100)  
        # # Time for Regsitration:  0.03296561241149902
        # assert 3>333
        ## test time efficienty of Reference
        
        # time_start = time.time()
        # for i in range(100):
        #     if self.PredictMomnetum:
        #         momentum_0 = self.common_step_stage1(batch, batch_idx)
        #         velocity_0 = self.MEpdiff.metric.sharp(momentum_0)
        #     else:
        #         velocity_0 = self.common_step_stage1(batch, batch_idx)
        #     phiinv_disp_list, phi_disp_list, v_list, m_list = self.MEpdiff.ShootWithV0(velocity_0)
        #     dfmSrc = lm.interp(self.src, phiinv_disp_list[-1])
        # time_end = time.time()
        # print("Time for Reference: ", (time_end - time_start)/100)
        # assert 3>987

        


        if self.PredictMomnetum:
            momentum_0 = self.common_step_stage1(batch, batch_idx)
            velocity_0 = self.MEpdiff.metric.sharp(momentum_0)
        else:
            velocity_0 = self.common_step_stage1(batch, batch_idx)


        phiinv_disp_list, phi_disp_list, v_list, m_list = self.MEpdiff.ShootWithV0(velocity_0)
        phiinv_disp_list2, phi_disp_list2 = self.MEpdiff.IntWithVList2(v_list)

        dfmSrc = lm.interp(self.src, phiinv_disp_list[-1])
        dfmSrc2 = lm.interp(self.src, phiinv_disp_list2[-1])


        # phis = phi_disp_list[-1].cpu().detach().numpy()
        # np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_MELBA/000MELBA_SEG_S0438_T0441/phi_disp_LM_S0438_T0441.npy", phis)
        # assert 1>999

        

        # #save dfmSrc as .nii.gz
        # import SimpleITK as sitk
        # dfmSrc = F.interpolate(dfmSrc, scale_factor=2, mode='trilinear')
        # dfmSrcimg = dfmSrc[0,0].cpu().detach().numpy(); dfmSrcimg = sitk.GetImageFromArray(dfmSrcimg)
        # sitk.WriteImage(dfmSrcimg, "/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_MELBA/000MELBA_SEG_S0438_T0441/LM-dfm438.nii.gz")
        # print(self.src.shape)
        # assert 1>222


        # phiinvs = phiinv_disp_list[-1].cpu().detach().numpy()
        # np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_MELBA/000MELBA_SEG_S0438_T0441/phiinv_disp_LM_S0438_T0441.npy", phiinvs)
        # assert 1>908


        dfmSrc_list = [lm.interp(self.src, phiinv_disp) for phiinv_disp in phiinv_disp_list]
        if self.ndims == 3:
            dfmSrclabel = self.MEpdiff.transformer(self.srclabel, phiinv_disp_list[-1], mode='nearest')

       

        mse = self.mse_metric(dfmSrc, self.tar)
        VecF_List_Tensor = velocity_0.unsqueeze(0)
        MomF_List_Tensor = m_list[0].unsqueeze(0)
        reg = self.regmetric(VecF_List_Tensor, MomF_List_Tensor)

        loss = self.weight_mse * mse + self.weight_reg * reg

        # return {
        #     "pred": dfmSrc,
        #     "pred2": dfmSrc2,
        #     "dfmSrclabel": dfmSrclabel if self.ndims == 3 else None,
        #     "src": self.src,
        #     "tar": self.tar,
        #     "mse": mse,
        #     "velocity_0": velocity_0,
        #     "reg": reg,
        #     "phiinv_disp_list": phiinv_disp_list,
        #     "phi_disp_list": phi_disp_list,
        #     "VecF_List": v_list,
        #     "MomF_List": m_list,
        #     "dfmSrc_list": dfmSrc_list,
        #     "phiinv": phiinv_disp_list[-1],


            
        # }
        self.log_dict(
            {
                "test_loss": loss,
                "test_mse": mse,
                "test_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {
            "dfmSrclabel": dfmSrclabel if self.ndims == 3 else None,
            "tarlabel": self.tarlabel if self.ndims == 3 else None,
            "srclabel": self.srclabel if self.ndims == 3 else None,

            # "phiinv": phiinv_disp_list[-1],
            "phiinv": phi_disp_list[-1],


            "pred": dfmSrc,
            "src": self.src,
            "tar": self.tar,
            "loss": loss,
            "mse": mse,
            "reg": reg,
            "VecF_List": v_list,
            "phiinv_disp_list": phiinv_disp_list,
            "dfmSrc_list": dfmSrc_list,


            # "MomF_List": m_list,
            #### "phi_disp_list": phi_disp_list,
        }
    

    # save training data for QS training
    def test_step111(self, batch, batch_idx):
        velocity_0 = self.common_step_stage1(batch, batch_idx)
        phiinv_disp_list, phi_disp_list, v_list, m_list = self.MEpdiff.ShootWithV0(velocity_0)
        # phiinv_disp_list2, phi_disp_list2 = self.MEpdiff.IntWithVList2(v_list)
        
        dfmSrc = lm.interp(self.src, phiinv_disp_list[-1])
        # dfmSrc2 = lm.interp(self.src, phiinv_disp_list2[-1])       

        mse = self.mse_metric(dfmSrc, self.tar)
        VecF_List_Tensor = velocity_0.unsqueeze(0)
        MomF_List_Tensor = m_list[0].unsqueeze(0)
        reg = self.regmetric(VecF_List_Tensor, MomF_List_Tensor)

        loss = self.weight_mse * mse + self.weight_reg * reg

        self.log_dict(
            {
                "test_loss": loss,
                "test_mse": mse,
                "test_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {
            "loss": loss,
            "mse": mse,
            "reg": reg,
        }


    def test_epoch_end(self, outputs):
        """ jacobian_det_for_all(outputs, ongpuGrad=False, TSteps=self.TSteps, templateIDX=self.templateIDX, phiinv=False,\
                             dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), inshape=self.inshape, regW=self.weight_reg)
        hdorff_coefficient_for_brain(outputs, ongpuGrad=False, TSteps=self.TSteps, templateIDX=self.templateIDX,\
                                      dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), regW=self.weight_reg) """
        
        plot_dfm_process(outputs, ongpuGrad=False, mse_metric=self.mse_metric, TSteps=self.TSteps, inshape=self.inshape, showTSteps=[0,-1], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES",saveRes=False)
        


        # jacobian_det_for_all(outputs, ongpuGrad=False, TSteps=self.TSteps,  templateIDX=self.templateIDX, \
        #                      dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), inshape=self.inshape, regW=self.weight_reg)
        # dice_coefficient_for_brain(outputs, ongpuGrad=False, TSteps=self.TSteps,  templateIDX=self.templateIDX, \
        #                            dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), regW=self.weight_reg, hdorff=False)


        
        # plot_dfm_process(outputs, ongpuGrad=False, TSteps=self.TSteps, showTSteps=[-1], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES")
        # plot_norm_list_gpu(outputs, self.regmetric, self.mse_metric, TSteps=self.TSteps dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES")
        # plot_dfm_process(outputs, ongpuGrad=False, mse_metric=self.mse_metric, TSteps=self.TSteps, showTSteps=[0,1,2,3,4], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES",saveRes=False)
        # plot_dfm_process(outputs, ongpuGrad=False, mse_metric=self.mse_metric, TSteps=self.TSteps, showTSteps=[4], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES",saveRes=False)


    def test_epoch_end_wj(self, outputs):
        srcs = torch.cat([output["src"] for output in outputs]).cpu().numpy()
        tars = torch.cat([output["tar"] for output in outputs]).cpu().numpy()
        preds = torch.cat([output["pred"] for output in outputs]).cpu().numpy()
        preds_list = [output["dfmSrc_list"] for output in outputs]
        zeros_channel = np.zeros((100, 1, 32, 32))

        #get the VecF_List
        VecF_0 = self.add_zero_channel_and_norm(torch.cat([output["VecF_List"][0] for output in outputs]).cpu().numpy())
        VecF_1 = self.add_zero_channel_and_norm(torch.cat([output["VecF_List"][1] for output in outputs]).cpu().numpy())
        VecF_2 = self.add_zero_channel_and_norm(torch.cat([output["VecF_List"][2] for output in outputs]).cpu().numpy())
        VecF_3 = self.add_zero_channel_and_norm(torch.cat([output["VecF_List"][3] for output in outputs]).cpu().numpy())
        VecF_4 = self.add_zero_channel_and_norm(torch.cat([output["VecF_List"][4] for output in outputs]).cpu().numpy())
        VecF_5 = self.add_zero_channel_and_norm(torch.cat([output["VecF_List"][5] for output in outputs]).cpu().numpy())
        VecF_6 = self.add_zero_channel_and_norm(torch.cat([output["VecF_List"][6] for output in outputs]).cpu().numpy())
        VecF_7 = self.add_zero_channel_and_norm(torch.cat([output["VecF_List"][7] for output in outputs]).cpu().numpy())
        VecF_8 = self.add_zero_channel_and_norm(torch.cat([output["VecF_List"][8] for output in outputs]).cpu().numpy())
        VecF_9 = self.add_zero_channel_and_norm(torch.cat([output["VecF_List"][9] for output in outputs]).cpu().numpy())

        #phiinv_disp_list
        phiinv_disp_list_0 = torch.cat([output["phiinv_disp_list"][0] for output in outputs]).cpu().numpy()
        phiinv_disp_list_1 = torch.cat([output["phiinv_disp_list"][1] for output in outputs]).cpu().numpy()
        phiinv_disp_list_3 = torch.cat([output["phiinv_disp_list"][3] for output in outputs]).cpu().numpy()
        phiinv_disp_list_5 = torch.cat([output["phiinv_disp_list"][5] for output in outputs]).cpu().numpy()
        phiinv_disp_list_7 = torch.cat([output["phiinv_disp_list"][7] for output in outputs]).cpu().numpy()
        phiinv_disp_list_9 = torch.cat([output["phiinv_disp_list"][9] for output in outputs]).cpu().numpy()
    
        #dfm list
        dfmSrc_list_0 = torch.cat([output["dfmSrc_list"][0] for output in outputs]).cpu().numpy()
        dfmSrc_list_1 = torch.cat([output["dfmSrc_list"][1] for output in outputs]).cpu().numpy()
        dfmSrc_list_2 = torch.cat([output["dfmSrc_list"][2] for output in outputs]).cpu().numpy()
        dfmSrc_list_3 = torch.cat([output["dfmSrc_list"][3] for output in outputs]).cpu().numpy()
        dfmSrc_list_4 = torch.cat([output["dfmSrc_list"][4] for output in outputs]).cpu().numpy()
        dfmSrc_list_5 = torch.cat([output["dfmSrc_list"][5] for output in outputs]).cpu().numpy()
        dfmSrc_list_6 = torch.cat([output["dfmSrc_list"][6] for output in outputs]).cpu().numpy()
        dfmSrc_list_7 = torch.cat([output["dfmSrc_list"][7] for output in outputs]).cpu().numpy()
        dfmSrc_list_8 = torch.cat([output["dfmSrc_list"][8] for output in outputs]).cpu().numpy()
        dfmSrc_list_9 = torch.cat([output["dfmSrc_list"][9] for output in outputs]).cpu().numpy()

        


        print(srcs.shape, tars.shape, preds.shape, VecF_0.shape, phiinv_disp_list_9.shape, dfmSrc_list_9.shape) #(100, 1, 32, 32) (100, 1, 32, 32) (100, 1, 32, 32) (100, 2, 32, 32) (100, 2, 32, 32)
        # assert 2>333

        batch_size = 20
        num_per_row = 6+3
        # 创建图形
        import matplotlib.pyplot as plt
        import os
        grids_dir = os.path.join(self.logger.log_dir, "grids")
        os.makedirs(grids_dir, exist_ok=True)
        fig, axes = plt.subplots(batch_size*2, num_per_row, figsize=(num_per_row*2.5, batch_size*2*2.5))  # 每行 3  batch_size 行

        for i in range(batch_size):
            if i != 15:
                continue
            # 如果只有一行，需要特殊处理 axes 的索引
            """ if batch_size == 1:
                ax_src, ax_tar, ax_pred, ax_pred2, ax_pred3, ax_pred4 = axes[0], axes[1], axes[2], axes[3], axes[4], axes[5]
            else:
                ax_src, ax_tar, ax_pred, ax_pred2, ax_pred3, ax_pred4 = axes[i] """
            # ax_src, ax_tar, ax_pred, ax_pred2, ax_pred3, ax_pred4 = axes[i]

            # 绘制源图像
            ax_src = axes[2*i][0]
            ax_src.imshow(srcs[i, 0], cmap="gray")  # 选择第 0 通道
            ax_src.axis("off")
            ax_src.set_title("Source")
            axes[2*i+1][0].cla(); axes[2*i+1][0].axis("off")

            # 绘制目标图像
            ax_tar = axes[2*i][1]
            ax_tar.imshow(tars[i, 0], cmap="gray")  # 选择第 0 通道
            ax_tar.axis("off")
            ax_tar.set_title("Target")
            axes[2*i+1][1].cla(); axes[2*i+1][1].axis("off")


            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/src.npy", srcs[i, 0])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/tar.npy", tars[i, 0])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/V_0.npy", VecF_0[i])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/dfmSrc_0.npy", dfmSrc_list_0[i, 0])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/V_1.npy", VecF_1[i])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/dfmSrc_1.npy", dfmSrc_list_1[i, 0])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/V_2.npy", VecF_2[i])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/dfmSrc_2.npy", dfmSrc_list_2[i, 0])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/V_3.npy", VecF_3[i])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/dfmSrc_3.npy", dfmSrc_list_3[i, 0])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/V_4.npy", VecF_4[i])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/dfmSrc_4.npy", dfmSrc_list_4[i, 0])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/V_5.npy", VecF_5[i])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/dfmSrc_5.npy", dfmSrc_list_5[i, 0])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/V_6.npy", VecF_6[i])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/dfmSrc_6.npy", dfmSrc_list_6[i, 0])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/V_7.npy", VecF_7[i])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/dfmSrc_7.npy", dfmSrc_list_7[i, 0])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/V_8.npy", VecF_8[i])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/dfmSrc_8.npy", dfmSrc_list_8[i, 0])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/V_9.npy", VecF_9[i])
            np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/dfmSrc_9.npy", dfmSrc_list_9[i, 0])

            # assert 3>333












            # 绘制预测图像
            ax_pred = axes[2*i][2]
            ax_pred.imshow(preds[i, 0], cmap="gray")  # 选择第 0 通道
            ax_pred.axis("off")
            ax_pred.set_title("DfmSrc")
            axes[2*i+1][2].cla(); axes[2*i+1][2].axis("off")

            # 绘制预测图像
            ax_pred2 = axes[2*i][3]
            ax_pred2.imshow(VecF_0[i])  # 选择第 0 通道
            # ax_pred2.imshow(dfmSrc_list_0[i, 0], cmap="gray")  # 选择第 0 通道
            ax_pred2.axis("off")
            ax_pred2.set_title("VecF_0")
            Mgridplot(phiinv_disp_list_0[i:i+1], axes[2*i+1][3],  32, 32, displacement = True)

            

            # 绘制预测图像
            ax_pred3 = axes[2*i][4]
            ax_pred3.imshow(VecF_1[i])  # 选择第 0 通道
            # ax_pred3.imshow(dfmSrc_list_1[i, 0], cmap="gray")  # 选择第 0 通道
            ax_pred3.axis("off")
            ax_pred3.set_title("VecF_1")
            Mgridplot(phiinv_disp_list_1[i:i+1], axes[2*i+1][4],  32, 32, displacement = True)

            # 绘制预测图像
            ax_pred4 = axes[2*i][5]
            ax_pred4.imshow(VecF_3[i])  # 选择第 0 通道
            # ax_pred4.imshow(dfmSrc_list_3[i, 0], cmap="gray")
            ax_pred4.axis("off")
            ax_pred4.set_title("VecF_3")
            Mgridplot(phiinv_disp_list_3[i:i+1], axes[2*i+1][5],  32, 32, displacement = True)

            # 绘制预测图像
            ax_pred5 = axes[2*i][6]
            ax_pred5.imshow(VecF_5[i])  # 选择第 0 通道
            # ax_pred5.imshow(dfmSrc_list_5[i, 0], cmap="gray")
            ax_pred5.axis("off")
            ax_pred5.set_title("VecF_5")
            Mgridplot(phiinv_disp_list_5[i:i+1], axes[2*i+1][6],  32, 32, displacement = True)
            
            # 绘制预测图像
            ax_pred6 = axes[2*i][7]
            ax_pred6.imshow(VecF_7[i])  # 选择第 0 通道
            # ax_pred6.imshow(dfmSrc_list_7[i, 0], cmap="gray")
            ax_pred6.axis("off")
            ax_pred6.set_title("VecF_7")
            Mgridplot(phiinv_disp_list_7[i:i+1], axes[2*i+1][7],  32, 32, displacement = True)

            # 绘制预测图像
            ax_pred7 = axes[2*i][8]
            ax_pred7.imshow(VecF_9[i])  # 选择第 0 通道
            # ax_pred7.imshow(dfmSrc_list_9[i, 0], cmap="gray")
            ax_pred7.axis("off")
            ax_pred7.set_title("VecF_9")
            Mgridplot(phiinv_disp_list_9[i:i+1], axes[2*i+1][8],  32, 32, displacement = True)


        # 调整子图布局
        plt.tight_layout()
        #save
        # plt.savefig(os.path.join(grids_dir, f"/home/nellie/code/cvpr/Project/DynamiCrafter/main/wj_2.png"))
        plt.show()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.unet_lr)

    def defmorSrc(self, velocity_0):
        phiinv_disp_list, phi_disp_list, v_list, m_list = self.MEpdiff.ShootWithV0(velocity_0)
        dfmSrc = lm.interp(self.src, phiinv_disp_list[-1])
        phiinv = phiinv_disp_list[-1] + self.grid
        phi = phi_disp_list[-1] + self.grid
        return dfmSrc, phiinv, phi, v_list[0], m_list[0]

    def deformSrcList(self, velocity_0):
        phiinv_disp_list, phi_disp_list, v_list, m_list = self.MEpdiff.ShootWithV0(velocity_0)
        dfmSrc_list = [lm.interp(self.src, phiinv_disp) for phiinv_disp in phiinv_disp_list]
        phiinv_list = [phiinv_disp + self.grid for phiinv_disp in phiinv_disp_list]
        phi_list = [phi_disp + self.grid for phi_disp in phi_disp_list]
        return dfmSrc_list, phiinv_list, phi_list, v_list, m_list





class QS(pl.LightningModule):
    def __init__(self, unet_config, reg_config, weight_mse, weight_reg, weight_sigma, unet_lr, templateIDX=-1,
                 *args, **kwargs):
        super().__init__()
        self.unet = instantiate_from_config(unet_config)
        self.regmetric = instantiate_from_config(reg_config)
        self.unetqs = instantiate_from_config(unet_config)

        self.mse_metric = torchmetrics.MeanSquaredError()
        self.weight_reg = weight_reg
        self.unet_lr = unet_lr
        self.ndims = unet_config['params']['ndims']
        self.img_size = kwargs['inshape'][0]
        self.TSteps = kwargs['TSteps']
        self.inshape = kwargs['inshape']
        self.templateIDX = templateIDX

        self.weight_mse = weight_mse / (weight_sigma**2)
        self.unet_lr = unet_lr

        self.PredictMomnetum = kwargs['PredictMomnetum'] if 'PredictMomnetum' in kwargs else False
       
        
        # self.fluid_params = [alpha, 0, gamma]
        # self.metric = lm.FluidMetric(self.fluid_params)

        self.MEpdiff = Epdiff(**{
            'inshape': kwargs['inshape'],
            'alpha': kwargs['alpha'], #alpha,
            'gamma': kwargs['gamma'], #gamma,
            'TSteps': kwargs['TSteps']
        })
        self.grid = self.MEpdiff.identity(kwargs['inshape'])


        # Register hooks for self.unet parameters
        cnt=0
        for name, param in self.unetqs.named_parameters():
            cnt += 1
            if cnt % 7 == 0:
                if param.requires_grad:
                    param.register_hook(grad_stats(f"self.unet.{name}"))



    def encoder_step(self, x):
        VecLnt = self.unet.exec_encoder(x)  #V in Latent space
        return VecLnt
    
    def decoder_step(self, x):
        VecF = self.unet.exec_decoder(x) #V in full space
        return VecF
    
    def common_step_stage1(self, batch, batch_idx):
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
        VecF = self.decoder_step(VecLnt)
        velocity_0 = VecF

        return velocity_0

    def encoder_stepqs(self, x):
        VecLnt = self.unetqs.exec_encoder(x)  #V in Latent space
        return VecLnt
    
    def decoder_stepqs(self, x):
        VecF = self.unetqs.exec_decoder(x) #V in full space
        return VecF 
    
    def common_step_stageqs(self, batch, batch_idx):
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

        VecLnt = self.encoder_stepqs(ipt)
        VecF = self.decoder_stepqs(VecLnt)
        velocity_0 = VecF

        return velocity_0
    
    def common_step_train(self, batch, batch_idx):
        if self.PredictMomnetum:
            momentum_0 = self.common_step_stage1(batch, batch_idx)
            velocity_0 = self.MEpdiff.metric.sharp(momentum_0)
        else:
            velocity_0 = self.common_step_stage1(batch, batch_idx)
            momentum_0 = self.MEpdiff.metric.flat(velocity_0)


        pred_momentum_0 = self.common_step_stageqs(batch, batch_idx)
        mse = self.mse_metric(momentum_0, pred_momentum_0)

        
        return mse
    
    def training_step(self, batch, batch_idx):
        loss = self.common_step_train(batch, batch_idx)
        # print("optimizer_idx ",optimizer_idx, loss, mse, reg)
        self.log_dict(
            {
                "train_mse": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if self.epoch_start_time is not None:
            elapsed_time = time.time() - self.epoch_start_time
            print(f"\n\n Epoch {self.current_epoch} ended. Time elapsed: {elapsed_time:.2f} seconds. \n\n")


    def validation_step(self, batch, batch_idx):
        loss = self.common_step_train(batch, batch_idx)
        self.log_dict(
            {
                "val_mse": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss


    #LgMo test
    def common_step_test(self, batch, batch_idx):
        if self.PredictMomnetum:
            momentum_0 = self.common_step_stage1(batch, batch_idx)
            velocity_0 = self.MEpdiff.metric.sharp(momentum_0)
        else:
            velocity_0 = self.common_step_stage1(batch, batch_idx)
            momentum_0 = self.MEpdiff.metric.flat(velocity_0)


        pred_momentum_0 = self.common_step_stageqs(batch, batch_idx)
        velocity_0 = self.MEpdiff.metric.sharp(pred_momentum_0)
        mse = self.mse_metric(momentum_0, pred_momentum_0)

        phiinv_disp_list, phi_disp_list, v_list, m_list = self.MEpdiff.ShootWithV0(velocity_0)
        dfmSrc = lm.interp(self.src, phiinv_disp_list[-1])

        reg_mse = self.mse_metric(dfmSrc, self.tar)
        VecF_List_Tensor = velocity_0.unsqueeze(0)
        MomF_List_Tensor = m_list[0].unsqueeze(0)
        reg = self.regmetric(VecF_List_Tensor, MomF_List_Tensor)

        loss = self.weight_mse * reg_mse + self.weight_reg * reg
        return reg, loss, mse, reg_mse
    
    def test_step(self, batch, batch_idx):
        pred_momentum_0 = self.common_step_stageqs(batch, batch_idx)
        velocity_0 = self.MEpdiff.metric.sharp(pred_momentum_0)

        phiinv_disp_list, phi_disp_list, v_list, m_list = self.MEpdiff.ShootWithV0(velocity_0)
        phiinv_disp_list2, phi_disp_list2 = self.MEpdiff.IntWithVList2(v_list)

        dfmSrc = lm.interp(self.src, phiinv_disp_list[-1])
        dfmSrc2 = lm.interp(self.src, phiinv_disp_list2[-1])

        phis = phi_disp_list[-1].cpu().detach().numpy()
        np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_MELBA/000MELBA_SEG_S0438_T0441/phi_disp_QS_S0438_T0441.npy", phis)
        assert 1>999


        #save dfmSrc as .nii.gz
        import SimpleITK as sitk
        dfmSrc = F.interpolate(dfmSrc, scale_factor=2, mode='trilinear')
        dfmSrcimg = dfmSrc[0,0].cpu().detach().numpy(); dfmSrcimg = sitk.GetImageFromArray(dfmSrcimg)
        sitk.WriteImage(dfmSrcimg, "/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_MELBA/000MELBA_SEG_S0438_T0441/QS-dfm438.nii.gz")
        print(self.src.shape)
        assert 1>222

        phiinvs = phiinv_disp_list[-1].cpu().detach().numpy()
        np.save(f"/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_MELBA/000MELBA_SEG_S0438_T0441/phiinv_disp_QS_S0438_T0441.npy", phiinvs)
        assert 1>908




        dfmSrc_list = [lm.interp(self.src, phiinv_disp) for phiinv_disp in phiinv_disp_list]
        if self.ndims == 3:
            dfmSrclabel = self.MEpdiff.transformer(self.srclabel, phiinv_disp_list[-1], mode='nearest')

        # assert 3>112

        mse = self.mse_metric(dfmSrc, self.tar)
        VecF_List_Tensor = velocity_0.unsqueeze(0)
        MomF_List_Tensor = m_list[0].unsqueeze(0)
        reg = self.regmetric(VecF_List_Tensor, MomF_List_Tensor)

        loss = self.weight_mse * mse + self.weight_reg * reg

        self.log_dict(
            {
                "test_loss": loss,
                "test_mse": mse,
                "test_reg": reg,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {
            "dfmSrclabel": dfmSrclabel if self.ndims == 3 else None,
            "tarlabel": self.tarlabel if self.ndims == 3 else None,
            "srclabel": self.srclabel if self.ndims == 3 else None,

            # "phiinv": phiinv_disp_list[-1],
            "phiinv": phi_disp_list[-1],


            "pred": dfmSrc,
            "src": self.src,
            "tar": self.tar,
            "loss": loss,
            "mse": mse,
            "reg": reg,
            "VecF_List": v_list,
            "phiinv_disp_list": phiinv_disp_list,
            "dfmSrc_list": dfmSrc_list,


            # "MomF_List": m_list,
            #### "phi_disp_list": phi_disp_list,
        }
    



    def test_epoch_end(self, outputs):
        """ jacobian_det_for_all(outputs, ongpuGrad=False, TSteps=self.TSteps, templateIDX=self.templateIDX, phiinv=False,\
                             dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), inshape=self.inshape, regW=self.weight_reg)
        hdorff_coefficient_for_brain(outputs, ongpuGrad=False, TSteps=self.TSteps, templateIDX=self.templateIDX,\
                                      dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), regW=self.weight_reg) """


        plot_dfm_process(outputs, ongpuGrad=False, mse_metric=self.mse_metric, TSteps=self.TSteps, inshape=self.inshape, showTSteps=[0,-1], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES",saveRes=False)
        


        # jacobian_det_for_all(outputs, ongpuGrad=False, TSteps=self.TSteps, templateIDX=self.templateIDX,\
        #                      dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), inshape=self.inshape, regW=self.weight_reg)
        # dice_coefficient_for_brain(outputs, ongpuGrad=False, TSteps=self.TSteps, templateIDX=self.templateIDX,\
        #                            dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), regW=self.weight_reg, hdorff=False)
        
        # plot_dfm_process(outputs, ongpuGrad=False, mse_metric=self.mse_metric, TSteps=self.TSteps, showTSteps=[0,1,2,3,4], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES")
        # plot_dfm_process(outputs, ongpuGrad=False, TSteps=self.TSteps, showTSteps=[-1], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES")
        # plot_norm_list_gpu(outputs, self.regmetric, self.mse_metric, TSteps=self.TSteps)
        

    def configure_optimizers(self):
        return optim.Adam(self.unetqs.parameters(), lr=self.unet_lr)











class LDDMM_Optimize(pl.LightningModule):
    def __init__(self, reg_config, weight_mse, weight_reg, weight_sigma, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.regmetric = instantiate_from_config(reg_config)
        self.mse_metric = torchmetrics.MeanSquaredError()
        self.ndims = reg_config['params']['ndims']
        self.img_size = kwargs['inshape'][0]
        self.TSteps = kwargs['TSteps']
        self.inshape = kwargs['inshape']


        self.weight_mse = weight_mse / (weight_sigma**2)
        self.weight_reg = weight_reg

        self.numerical_solver = kwargs['numerical_solver'] if 'numerical_solver' in kwargs else False
        self.numerical_lr = kwargs['numerical_lr'] if 'numerical_lr' in kwargs else None
        self.numerical_steps = kwargs['numerical_steps'] if 'numerical_steps' in kwargs else None


        self.MEpdiff = Epdiff(**{
            'inshape': kwargs['inshape'],
            'alpha': kwargs['alpha'], #alpha,
            'gamma': kwargs['gamma'], #gamma,
            'TSteps': kwargs['TSteps']
        })
        self.grid = self.MEpdiff.identity(kwargs['inshape'])

    def common_step_numerical_solver(self, velocity_0):
        phiinv_disp_list, phi_disp_list, v_list, m_list = self.MEpdiff.ShootWithV0(velocity_0)
        #deform
        dfmSrc = lm.interp(self.src, phiinv_disp_list[-1])
        #compute loss
        
        mse = self.mse_metric(dfmSrc, self.tar)
        VecF_List_Tensor = velocity_0.unsqueeze(0)
        MomF_List_Tensor = m_list[0].unsqueeze(0)
        reg = self.regmetric(VecF_List_Tensor, MomF_List_Tensor)
        loss = self.weight_mse * mse + self.weight_reg * reg

        # print(loss.item(), self.weight_mse, self.weight_reg, mse.item(), reg.item())
        # assert 3>123

        return loss, mse, reg, dfmSrc, phiinv_disp_list, phi_disp_list, v_list, m_list
    
    def training_step(self, batch, batch_idx):
        # print(self.numerical_solver = kwargs['numerical_solver'] if 'numerical_solver' in kwargs else False
        # self.numerical_lr = kwargs['numerical_lr'] if 'numerical_lr' in kwargs else None
        # self.numerical_steps)

        # print("numerical_solver: ", self.numerical_solver)
        # print("numerical_lr: ", self.numerical_lr)
        # print("numerical_steps: ", self.numerical_steps)
        # assert 1>123
       
        if self.ndims == 2:
            b,c,w,h = batch['src'].shape
            self.src = batch['src'].view(-1, 1, w,h) #shape [120, 1, 32, 32]
            self.tar = batch['tar'].view(-1, 1, w,h) #shape [120, 1, 32, 32]
            inshape = (b,self.ndims,self.img_size,self.img_size)
        else:
            b,c,w,h,d = batch['src'].shape
            self.src = batch['src'] #shape [120, 1, 128, 128, 128]
            self.tar = batch['tar'] #shape [120, 1, 128, 128, 128]

            self.srclabel = batch['srclabel'] #shape [120, 1, 128, 128, 128]
            self.tarlabel = batch['tarlabel'] #shape [120, 1, 128, 128, 128]

            inshape = (b,self.ndims,self.img_size,self.img_size,self.img_size)

        
        # print(self.src.shape, self.tar.shape) #[1, 1, 32, 32]
        momutunm_0 = torch.zeros(inshape, device=self.src.device, requires_grad=True)
        optimizer = torch.optim.SGD([{'params': [momutunm_0], 'lr': self.numerical_lr}])
        
        #shooting
        #start time
        start_time = time.time()
        for ep in range(self.numerical_steps):
            velocity_0 = self.MEpdiff.metric.sharp(momutunm_0)
            loss, mse, reg, dfmSrc, phiinv_disp_list, phi_disp_list, v_list, m_list = self.common_step_numerical_solver(velocity_0)
            
            optimizer.zero_grad()
            loss.backward()
            # has_nan = torch.isnan(momutunm_0.grad).any()
            # print("Contains NaN:", has_nan.item())  # True
            # assert 3>111
            # if (ep % 1 == 0):
            print(f"{batch_idx}-{ep}: mse: {mse.item()}  reg: {reg.item()} momutunm_0: max={momutunm_0.max().item()}, min={momutunm_0.min().item()}  Gradient momutunm_0: max={momutunm_0.grad.max().item()}, min={momutunm_0.grad.min().item()}")
            optimizer.step()
            # assert 3>123


        #end time
        elapsed_time = time.time() - start_time
        loss, mse, reg, dfmSrc, phiinv_disp_list, phi_disp_list, v_list, m_list = self.common_step_numerical_solver(velocity_0)
        # print(len(phiinv_disp_list))  #10
        dfmSrc_list = [lm.interp(self.src, phiinv_disp) for phiinv_disp in phiinv_disp_list]
        if self.ndims == 3:
            dfmSrclabel = self.MEpdiff.transformer(self.srclabel, phiinv_disp_list[-1], mode='nearest')
        

        # #save velocity_0 as .npy
        # np.save("/home/nellie/code/cvpr/Project/DynamiCrafter/lvdm/cardiacV0.npy", velocity_0.detach().cpu().numpy())
        # assert 2>111

        print(f"batch_idx:{batch_idx} loss: {loss}  mse: {mse}  reg: {reg} \n\n Time elapsed: {elapsed_time:.2f} seconds. \n\n")
        
        return {
            "dfmSrclabel": dfmSrclabel if self.ndims == 3 else None,
            "tarlabel": self.tarlabel if self.ndims == 3 else None,
            "phiinv": phiinv_disp_list[-1],
            
            "loss": loss,
            "pred": dfmSrc,
            "pred2": dfmSrc,
            "src": self.src,
            "tar": self.tar,
            "mse": mse,
            "reg": reg,
            # "phiinv_disp_list": phiinv_disp_list,
            "phiinv_disp_list": phi_disp_list,
            # "phi_disp_list": phi_disp_list,
            "VecF_List": v_list,
            "MomF_List": m_list,
            "dfmSrc_list": dfmSrc_list
        }

    def training_epoch_end1(self, outputs):
        plot_norm_list_gpu(outputs, self.regmetric , self.mse_metric)

    def training_epoch_end(self, outputs):
        # jacobian_det_for_all(outputs, ongpuGrad=True, TSteps=self.TSteps, dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), inshape=self.inshape, regW=self.weight_reg)
        # dice_coefficient_for_brain(outputs, ongpuGrad=True, TSteps=self.TSteps, dir=os.path.dirname(os.path.dirname(self.logger.log_dir)), regW=self.weight_reg, hdorff=False)
        
        
        # plot_dfm_process(outputs, ongpuGrad = True, mse_metric=self.mse_metric, TSteps=self.TSteps, showTSteps=[0,3,5,7,9], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES")
        plot_dfm_process(outputs, ongpuGrad = True, mse_metric=self.mse_metric, TSteps=self.TSteps, inshape=self.inshape, \
                         showTSteps=[0,1,2,3,4,5,6,7,8,9], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES",saveRes=False)
        # plot_dfm_process(outputs, ongpuGrad = True, mse_metric=self.mse_metric, TSteps=self.TSteps, showTSteps=[5,6,7,8,9], dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES")
        # plot_norm_list_gpu(outputs, self.regmetric , self.mse_metric, ongpuGrad = True, TSteps=self.TSteps, dir=f"{os.path.dirname(os.path.dirname(self.logger.log_dir))}/RES")
        # print("outputs['loss']:", outputs['loss'], "outputs['mse']:", outputs['mse'], "outputs['reg']:", outputs['reg'])


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



    #collect indexs_src and indexs_tar
    def test_step(self, batch, batch_idx):
        indexs_src = batch['indexs_src'].squeeze()
        indexs_tar = batch['indexs_tar'].squeeze()
        print(batch_idx, indexs_src, indexs_tar, indexs_tar.shape)
        return {
            "indexs_src": indexs_src,
            "indexs_tar": indexs_tar
        }
    def test_epoch_end(self, outputs):
        indexs_src = []
        indexs_tar = []
        for output in outputs:
            indexs_src.append(output['indexs_src'].item())
            indexs_tar.append(output['indexs_tar'].item())
        
        print("\n",  indexs_src)
        print("\n",  indexs_tar)
        assert 2>129




