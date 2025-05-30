import torch
import numpy as np
import torch.nn as nn
import lagomorph as lm
import pytorch_lightning as pl
from lvdm.Int import VecInt, SpatialTransformer, Epdiff, Grad
import torch.nn.functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    def __init__(self, ndims, in_channels, out_channels, kernal=3, stride=1, padding=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernal, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class unet(nn.Module):
    def __init__(self, inshape, nb_features, infeats, ndims, *args, **kwargs):
        super().__init__()
        nb_levels=None
        max_pool=2
        feat_mult=1
        nb_conv_per_level=1
        half_res=False



        self.inshape = inshape
        self.nb_features=nb_features
        self.infeats=infeats
        self.ndims = ndims
        self.half_res = half_res

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf


        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.final_nf, ndims, kernel_size=3, padding=1)
        
        
        """ for idex in range(len(self.encoder)):
            self.encoder[idex][0].main.name = f'unet-encoder-{idex}-0'
            self.decoder[idex][0].main.name = f'unet-decoder-{idex}-0'
        for idex in range(len(self.remaining)):
            self.remaining[idex].main.name = f'unet-remaining-{idex}'
        self.flow.name = 'unet-flow' """

    def exec_encoder(self, x):
        if(x.shape[-1]==2 and self.ndims==2):
            x = x.permute(0,3,1,2)
        elif(x.shape[-1]==3 and self.ndims==3):
            x = x.permute(0,4,1,2,3)

        # encoder forward pass
        self.x_history = [x]   #torch.Size([3, 2, 100, 100])

        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            self.x_history.append(x)
            x = self.pooling[level](x)  #100x100 -> 50x50 -> 25x25  ->12x12  ->6x6
                                        #128x128   64x64   32x32   16x16
        return x

    def forward(self, x): #x:[25, 2, 32, 32] 
        low_dim_features = self.exec_encoder(x)  #[25, 32, 8, 8]
        fnow_full_dim = self.exec_decoder(low_dim_features)
        return fnow_full_dim

    def UPS_ABI(self, x):
        output_tensor_true = F.interpolate(x, size=self.inshape, mode='bilinear', align_corners=True)
        return output_tensor_true

    def parallel_decoder(self, x_list):
        steps = len(x_list)
        x_parallel = torch.cat(x_list, dim=0)

        # Optimize x_history_parallel construction
        if self.ndims == 3:
            self.x_history_parallel = [h.expand(steps, -1, -1, -1, -1) for h in self.x_history]
        elif self.ndims == 2:
            self.x_history_parallel = [h.expand(steps, -1, -1, -1) for h in self.x_history]

        skip_index = -1
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x_parallel = conv(x_parallel)

            if level < (len(self.upsampling) - 1):  # Avoid redundant checks
                x_parallel = self.upsampling[level](x_parallel)
                x_parallel = torch.cat([x_parallel, self.x_history_parallel[skip_index]], dim=1)
                skip_index -= 1

        for conv in self.remaining:
            x_parallel = conv(x_parallel)
        x_parallel = self.flow(x_parallel)  #[20, 16, 128, 128] -> [20, 2, 128, 128]

        # x_list = torch.split(x_parallel, 1, dim=0)
        # print(x_parallel.shape)
        x_list = [x.unsqueeze(0) for x in torch.unbind(x_parallel, dim=0)]

        # print(x_list[0].shape)  #[3, 128, 128, 128]
        # assert 3>333



        return x_list




    def parallel_decoder333(self, x_list):
        steps = len(x_list)
        self.x_history_parallel = []
        x_parallel = torch.cat(x_list, dim=0)

        if self.ndims==3:
            for tt in range(len(self.x_history)):
                self.x_history_parallel.append(self.x_history[tt].repeat(steps,1,1,1,1))
                # print(tt, " x_history: ", self.x_history_parallel[-1].shape)
        elif self.ndims==2:
            for tt in range(len(self.x_history)):
                self.x_history_parallel.append(self.x_history[tt].repeat(steps,1,1,1))
                # print(tt, " x_history: ", self.x_history_parallel[-1].shape)

        # print("input: ", x_parallel.shape)

        skip_index = -1
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x_parallel = conv(x_parallel)
                # print("conv: ", x_parallel.shape)

            if not self.half_res or level < (self.nb_levels - 2):
                x_parallel = self.upsampling[level](x_parallel)
                x_parallel = torch.cat([x_parallel, self.x_history_parallel[skip_index]], dim=1)
                skip_index -= 1

        for conv in self.remaining:
            x_parallel = conv(x_parallel)
        # print("remaining: ", x_parallel.shape)

        x_parallel = self.flow(x_parallel)  #[20, 16, 128, 128] -> [20, 2, 128, 128]

        # print("flow: ", x_parallel.shape)

        # 1  x_history:  torch.Size([5, 8, 128, 128, 128])
        # 2  x_history:  torch.Size([5, 16, 64, 64, 64])
        # 3  x_history:  torch.Size([5, 16, 32, 32, 32])
        # 4  x_history:  torch.Size([5, 8, 16, 16, 16])
        # input:  torch.Size([5, 8, 8, 8, 8])
        # conv:  torch.Size([5, 8, 8, 8, 8])
        # conv:  torch.Size([5, 16, 16, 16, 16])
        # conv:  torch.Size([5, 16, 32, 32, 32])
        # conv:  torch.Size([5, 16, 64, 64, 64])
        # remaining:  torch.Size([5, 8, 128, 128, 128])
        # flow:  torch.Size([5, 3, 128, 128, 128])

        #split [5, 3, 128, 128, 128] into [1, 3, 128, 128, 128] x 5
        x_list = torch.split(x_parallel, 1, dim=0)
        # x_list = torch.unbind(x_parallel, dim=0)

        

        return x_list

    def exec_decoder(self, x):
        # print("input: ", x.shape)
        skip_index = -1
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
                # print("conv: ", x.shape)

            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, self.x_history[skip_index]], dim=1)
                skip_index -= 1

        for conv in self.remaining:
            x = conv(x)
        # print("remaining: ", x.shape)

        x = self.flow(x)  #[20, 16, 128, 128] -> [20, 2, 128, 128]

        # print("flow: ", x.shape)

        # assert 3>333

        # input:  torch.Size([1, 8, 8, 8, 8])
        # conv:  torch.Size([1, 8, 8, 8, 8])
        # conv:  torch.Size([1, 16, 16, 16, 16])
        # conv:  torch.Size([1, 16, 32, 32, 32])
        # conv:  torch.Size([1, 16, 64, 64, 64])
        # remaining:  torch.Size([1, 8, 128, 128, 128])
        # flow:  torch.Size([1, 3, 128, 128, 128])



        # return x.permute(0,2,3,1)   #-> [20, 128, 128, 2]
        # return x.permute(0,2,3,4,1)   #-> [20, 128, 128, 128,3]
        return x
    
    
        if(len(x.shape)==4):
            return x.permute(0,2,3,1)   #-> [20, 128, 128, 2]
        return x.permute(0,2,3,4,1)   #-> [20, 128, 128, 128,3]



class SVF(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.unet = unet(*args, **kwargs)

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
    
    def forward(self, x):
        self.src = x[:,0:1,...]
        self.tar = x[:,1:2,...]
        velocity_0 = self.unet(x)
        return velocity_0
    
    def defmorSrc(self, velocity_0):
        phiinv_disp_list = self.integrate(velocity_0)
        dfmSrc, phiinv =  self.transformer(self.src, phiinv_disp_list[-1])
        return dfmSrc, phiinv
    
    def defmorSrcList(self, velocity_0):
        phiinv_disp_list = self.integrate(velocity_0)

        dfmSrc_phiinv_list = [self.transformer(self.src, phiinv_disp) for phiinv_disp in phiinv_disp_list]
        dfmSrc_list = [dfmSrc for dfmSrc, _ in dfmSrc_phiinv_list]
        phiinv_list = [phiinv for _, phiinv in dfmSrc_phiinv_list]

        return dfmSrc_list, phiinv_list

class LDDMM(pl.LightningModule):
    def __init__(self, gamma, alpha, *args, **kwargs):
        super().__init__()
        self.unet = unet(*args, **kwargs)
        
        # self.fluid_params = [alpha, 0, gamma]
        # self.metric = lm.FluidMetric(self.fluid_params)

        self.MEpdiff = Epdiff(**{
            'inshape': kwargs['inshape'],
            'alpha': kwargs['alpha'], #alpha,
            'gamma': kwargs['gamma'], #gamma,
            'TSteps': kwargs['TSteps']
        })
        self.grid = self.MEpdiff.identity(kwargs['inshape'])

    def forward(self, x):
        self.src = x[:,0:1,...]
        self.tar = x[:,1:2,...]
        velocity_0 = self.unet(x)
        return velocity_0
    
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
    


   

    

