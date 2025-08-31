import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Global Convolutional Kernel
class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, *args, **kwargs):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights): #input:[b, 20, 12, 12]   weights:[20, 20, 12, 12]
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        # print(input.shape, weights.shape)
        '''1*'''
        # method1
        # result =  torch.einsum("bixy,ioxy->boxy", input, weights)  #[20, 20, 12, 12]

        '''2*'''
        ##  more effcient
        # x1 = torch.einsum("bixy,ioxy->boxy", input, weights[:,:4,...])  #[1, 20, 16, 16, 14]
        # x2 = torch.einsum("bixy,ioxy->boxy", input, weights[:,4:8,...])  #[1, 20, 16, 16, 14]
        # x3 = torch.einsum("bixy,ioxy->boxy", input, weights[:,8:12,...])  #[1, 20, 16, 16, 14]
        # x4 = torch.einsum("bixy,ioxy->boxy", input, weights[:,12:16,...])  #[1, 20, 16, 16, 14]
        # x5 = torch.einsum("bixy,ioxy->boxy", input, weights[:,16:,...])  #[1, 20, 16, 16, 14]
        # result = torch.concat((x1,x2,x3,x4,x5), dim=1)
    

        """ # '''3*''' bixy,ioxy->boxy  bixy*o1ixy->obixy
        weights_permuted = weights.permute(1,0,2,3).unsqueeze(1)
        # k = (input*(weights.permute(1,0,2,3).unsqueeze(1))).permute(1,0,2,3,4)
        k_raw = (input*weights_permuted)
        # input_real = input.real
        # weights_permuted_real = weights_permuted.real
        # k_raw_real = (input_real*weights_permuted_real)        
        k = k_raw.permute(1,0,2,3,4)
        result = torch.sum(k,dim=2)   #[10, 20, 8, 8]
        # # print(torch.allclose(result,result1)) """


        weights_permuted = weights.permute(1,0,2,3).unsqueeze(0)  #1, o, i, x, y
        ipt = input.unsqueeze(1) #b, 1, i, x, y
        elmwise_product = ipt*weights_permuted #bixy * b, 20, i, x, y
        result = torch.sum(elmwise_product, dim=2)  #b, 20, x, y
        



        return result
    
    def forward(self, x):   #Size([25, 20, 2, 2])
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x) #[1, 20, 128, 65]
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)  #[20, 20, 64, 33]
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)        
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))) #out_ft:[20, 20, 128, 65]    x:[20, 20, 128, 128]
        return x

class FNO2dsimple(nn.Module):
    def __init__(self, modes1=2, modes2=2, width=20):
        super().__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(32, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        # Global Convolutional Kernel
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        # self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        # self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        # self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)

        # Local linear tranformation
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        # self.w1 = nn.Conv2d(self.width, self.width, 1)
        # self.w2 = nn.Conv2d(self.width, self.width, 1)
        # self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 32)

    def forward(self, x):  #([25, 32, 2, 2]
        x = x.permute(0,2,3,1)   #([25, 32, 2, 2]   ->[25, 2, 2, 32]

        # grid = self.get_grid(x.shape, x.device) #[20, 64, 64, 2]
        # x = torch.cat((x, grid), dim=-1)        #[20, 64, 64, 3]
        x = self.fc0(x)                         #[20, 64, 64, 20]   Linear(in_features=12, out_features=20, bias=True)
        x = x.permute(0, 3, 1, 2)               #Size([25, 32, 2, 2])
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        # t1 = default_timer()
        x1 = self.conv0(x)                      #[20, 20, 64, 64]   2D Fourier layer. It does FFT, linear transform, and Inverse FFT.  
        # t2 = default_timer()
        x2 = self.w0(x)                         #[20, 20, 64, 64]   Conv2d(20, 20, kernel_size=(1, 1), stride=(1, 1))
        # t3 = default_timer()
        # print(f"!!!!!!!!!!! conv0: {t2-t1}, w0: {t3-t2}")
        x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv1(x)                      #[20, 20, 64, 64]   2D Fourier layer. It does FFT, linear transform, and Inverse FFT.  
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv2(x)                      #[20, 20, 64, 64]   2D Fourier layer. It does FFT, linear transform, and Inverse FFT.  
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv3(x)                      #[20, 20, 64, 64]   2D Fourier layer. It does FFT, linear transform, and Inverse FFT.  
        # x2 = self.w3(x)
        # x = x1 + x2



        """ x2 = self.w0(x)
        x = F.gelu(x2)
        x2 = self.w1(x)
        x = F.gelu(x2)
        x2 = self.w2(x)
        x = F.gelu(x2)
        x2 = self.w3(x)
        x = F.gelu(x2) """
        
        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)               #[20, 64, 64, 20]
        x = self.fc1(x)                         #[20, 64, 64, 128]          Linear(in_features=20, out_features=128, bias=True)
        x = F.gelu(x)
        x = self.fc2(x)                         #[20, 64, 64, 2]            Linear(in_features=128, out_features=1, bias=True)
        return x.permute(0, 3, 1, 2)

    def get_grid(self, shape, device):    #shape:[20, 64, 64, 10]
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)  #[64]
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])  #[20, 64, 64, 1]
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
# Geodesic Learning in latent space, iteratively learns the sequence of low-dimensional features 
class nop2dsimple(nn.Module):
    def __init__(self, ModesFno, WidthFno, TSteps):
        super().__init__()
        self.nop = FNO2dsimple(modes1=ModesFno, modes2=ModesFno, width=WidthFno)
        self.TSteps = TSteps
        
    def forward(self, low_dim_features): #x:[25, 2, 32, 32] 
        v_seq_low_dim = [low_dim_features]
        fnov_low_dim = low_dim_features
        for t in range(1,self.TSteps):  #T:9  step:1
            fnov_low_dim= self.nop(fnov_low_dim)
            v_seq_low_dim.append(fnov_low_dim)
        return v_seq_low_dim
    



# Global Convolutional Kernel
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        
        self.scale = (1 / (in_channels * out_channels))

        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.tst = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        # return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

        weights_permuted = weights.permute(1,0,2,3,4).unsqueeze(0)  #1, o, i, x, y, t
        ipt = input.unsqueeze(1) #b, 1, i, x, y, t
        elmwise_product = ipt*weights_permuted #bixyt * b, 20, i, x, y, t
        result = torch.sum(elmwise_product, dim=2)  #b, 20, x, y, t
        return result

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])  #x:[b, 12, 64, 64, 64]   x_ft:[1, 12, 64, 64, 33]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3dsimple(nn.Module):
    def __init__(self, modes1=4, modes2=4, modes3=4, width=8, WidthIn = 8, layers=1):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """
        
        # print(modes1, modes2, modes3, width, WidthIn)  #8 8 8 8 8      16 16 16 8 16   16 16 16 20 16
        # assert 3>333

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.width_in = WidthIn
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(self.width_in, self.width)
        self.layers = layers
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        # print(self.layers)
        # assert 3>333



        if self.layers >= 1:
            # Global Convolutional Kernel
            self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)

            # Local linear tranformation
            self.w0 = nn.Conv3d(self.width, self.width, 1)
        
        if self.layers > 1:
            self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
            self.w1 = nn.Conv3d(self.width, self.width, 1)
        
        if self.layers > 2:
            self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
            self.w2 = nn.Conv3d(self.width, self.width, 1)

        if self.layers > 3:
            self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
            self.w3 = nn.Conv3d(self.width, self.width, 1)
        
        if self.layers > 4:
            self.conv4 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
            self.w4 = nn.Conv3d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.width_in)



    def forward(self, x):  #x:[20, 64, 64, 64, 3]
        x = x.permute(0,2,3,4,1)
        
        # grid = self.get_grid(x.shape, x.device)  #[20, 64, 64, 64, 3]
        # x = torch.cat((x, grid), dim=-1)  #[20, 64, 64, 64, 6]
        x = self.fc0(x)  #[10, 64, 64, 64, 20]
        x = x.permute(0, 4, 1, 2, 3)  #[10, 20, 64, 64, 64]
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        if self.layers >= 1:
            x1 = self.conv0(x)
            x2 = self.w0(x)
            x = x1 + x2

        if self.layers > 1:
            x = F.gelu(x)
            x1 = self.conv1(x)
            x2 = self.w1(x)
            x = x1 + x2

        if self.layers > 2:
            x = F.gelu(x)
            x1 = self.conv2(x)
            x2 = self.w2(x)
            x = x1 + x2

        if self.layers > 3:
            x = F.gelu(x)
            x1 = self.conv3(x)
            x2 = self.w3(x)
            x = x1 + x2

        if self.layers > 4:
            x = F.gelu(x)
            x1 = self.conv4(x)
            x2 = self.w4(x)
            x = x1 + x2

        # x = x[..., :-self.padding]#[10, 1, 64, 64, 40]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)   #[10, 64, 64, 40, 3]               [1, 128, 128, 128, 3]
        return x.permute(0, 4, 1, 2, 3) 


# Geodesic Learning in latent space, iteratively learns the sequence of low-dimensional features 
class nop3dsimple(nn.Module):
    def __init__(self, ModesFno, WidthFno, TSteps, WidthIn=8, layers=1):
        super().__init__()
        self.nop = FNO3dsimple(modes1=ModesFno, modes2=ModesFno, modes3=ModesFno, width=WidthFno, WidthIn=WidthIn, layers=layers)
        self.TSteps = TSteps
        
    def forward(self, low_dim_features): #x:[25, 2, 32, 32] 
        v_seq_low_dim = [low_dim_features]
        fnov_low_dim = low_dim_features
        for t in range(1,self.TSteps):  #T:9  step:1
            fnov_low_dim= self.nop(fnov_low_dim)
            v_seq_low_dim.append(fnov_low_dim)
        return v_seq_low_dim




