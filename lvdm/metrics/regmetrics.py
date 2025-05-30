import torch
from torchmetrics import Metric

class LpLossMetricList(Metric):
    def __init__(self, W_Mse=5, W_Rel=5, d=2, p=2):
        super().__init__()
        self.d = d
        self.p = p
        self.W_Mse = W_Mse
        self.W_Rel = W_Rel

        self.add_state("total_combined_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_mse_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_rel_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    
    def update(self, xList: torch.Tensor, yList: torch.Tensor):
        # print(xList.shape, yList.shape)  #torch.Size([4, 1, 3, 128, 128, 128]) torch.Size([4, 1, 3, 128, 128, 128])
        # assert 3>222
        """
        xList, yList: shape (B, T, C, H, W) — list of T tensors per sample
        """
        assert xList.shape == yList.shape, "Shape mismatch between xList and yList"
        B, T = xList.shape[0], xList.shape[1]
        total_pairs = B * T

        x_flat = xList.view(total_pairs, -1)
        y_flat = yList.view(total_pairs, -1)

        # ----- Relative Lp Loss -----
        diff_norms = torch.norm(x_flat - y_flat, self.p, dim=1)         # shape: (B*T,)
        y_norms = torch.norm(y_flat, self.p, dim=1)                     # avoid div by zero
        rel_errors = diff_norms / y_norms                               # shape: (B*T,)
        # print(rel_errors.shape, diff_norms.shape, y_norms.shape)  #torch.Size([4]) torch.Size([4]) torch.Size([4])
        # print(rel_errors)

        # assert 2>198



        # print(rel_errors.shape, diff_norms.shape, y_norms.shape)  #torch.Size([4]) torch.Size([4]) torch.Size([4])
        # assert 3>222
        # ----- MSE Loss -----
        mse_errors = torch.mean((x_flat - y_flat) ** 2, dim=1)          # shape: (B*T,)

        # ----- Combine Both -----
        combined_loss = self.W_Rel * rel_errors + self.W_Mse * mse_errors
        
        # ----- Accumulate All -----
        self.total_combined_loss += combined_loss.sum()
        self.total_rel_loss += rel_errors.sum()
        self.total_mse_loss += mse_errors.sum()
        self.total_samples += total_pairs

        # print(self.W_Rel, self.W_Mse, rel_errors.sum(), mse_errors.sum(), total_pairs)
        # assert 3>222


    def compute(self):
        return {
            "noploss": self.total_combined_loss / self.total_samples,
            "noprel": self.total_rel_loss / self.total_samples,
            "nopmse": self.total_mse_loss / self.total_samples
        }

class MVMList(Metric): #Mean of List of Velocity * Momentum
    def __init__(self, ndims, *args, **kwargs):
        super().__init__()
        self.add_state("VMTotal", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("Len", default=torch.tensor(0), dist_reduce_fx="sum")
        self.ndims = ndims
    
    def comput_loss_list(self, VList_Tensor, MList_Tensor): #for visualization
        lenV = VList_Tensor.shape[0]
        nums = VList_Tensor[0].numel()/self.ndims
        VMList = [((VList_Tensor[i]*MList_Tensor[i]).sum()).item() / nums for i in range(lenV)]
        return VMList
    
    def update(self, VList_Tensor, MList_Tensor):
        # print(VList.shape, MList.shape)
        lenV = VList_Tensor.shape[0]
        nums = VList_Tensor[0].numel()/self.ndims
        VMList = [(VList_Tensor[i]*MList_Tensor[i]).sum() / nums for i in range(lenV)]
        # print(VMList)
        self.VMTotal += sum(VMList)
        # self.Len += torch.tensor(len(VMList))
        self.Len += torch.tensor(len(VMList), dtype=self.Len.dtype, device=self.Len.device)

    def compute(self):
        # return self.VMTotal / self.Len
        return self.VMTotal.float() / self.Len.float()
    



    
    def comp_l2_norm(self, v, m): #velocity:[120, 2, 32, 32] return torch.Size([120])
        if self.ndims == 2:
            l2_norm = torch.sum(v*m, dim=(1, 2, 3))
        elif self.ndims == 3:
            l2_norm = torch.sum(v*m, dim=(1, 2, 3, 4))

        return l2_norm / (v[0].numel()/self.ndims)
    
    def ReparametrizeVec(self, VList, MList): #[(120,2,32,32),,,,,,(120,2,32,32)]
        batch_size = VList[0].shape[0]
        T = len(VList)
        #[(120,),(120,),,,,,(120,)]
        VMList_NORM = [self.comp_l2_norm(VList[i], MList[i]) for i in range(T)]
        len_of_geodesic = sum(VMList_NORM) / T
        if self.ndims == 2:
            scales_list = [torch.sqrt(((VMList_NORM[i]) / len_of_geodesic)).view(batch_size, 1, 1, 1) for i in range(T)]
        elif self.ndims == 3:
            scales_list = [torch.sqrt(((VMList_NORM[i]) / len_of_geodesic)).view(batch_size, 1, 1, 1, 1) for i in range(T)]
       
        """ scales_list_mean = [torch.mean(t).item() for t in scales_list]
        print("scales_list_mean:  ", scales_list_mean) """

        VList_new = [VList[i] / scales_list[i]  for i in range(T)]
        # print(VList_new[0].shape, scales_list[0].shape)
        return VList_new

class MVGradList(Metric): #Mean of List of Velocity * Gradient
    def __init__(self, ndims, penalty = 'l2', *args, **kwargs):
        super().__init__()
        self.add_state("VGTotal", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("Len", default=torch.tensor(0), dist_reduce_fx="sum")
        self.ndims = ndims
        self.penalty = penalty

    def update(self, VList_Tensor, _):
        lenV = VList_Tensor.shape[0]
        VGradList = [self.loss(VList_Tensor[i])  for i in range(lenV)]
        self.VGTotal += sum(VGradList)
        # self.Len = torch.tensor(len(VGradList))
        self.Len += torch.tensor(len(VGradList), dtype=self.Len.dtype, device=self.Len.device)

    def compute(self):
        return self.VGTotal / self.Len


    def loss(self, y_pred):
        if(self.ndims == 3):
            dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
            dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
            dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz

            d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
            grad = d / 3.0
        elif(self.ndims == 2):
            # print("y_pred   ",y_pred.shape)
            dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
            dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
          
            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
            d = torch.mean(dx) + torch.mean(dy)
            grad = d / 2.0

        return grad
