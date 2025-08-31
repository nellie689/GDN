import numpy as np
import torch
from torch.utils.data import Dataset
import scipy
import torch.nn.functional as F
import SimpleITK as sitk
import json
import torchvision.transforms as transforms
import random
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import numpy as np



class Dense2d_LDDMM_Optimize(Dataset):
    def __init__(self, src_path, tar_path, mask_src_path, mask_tar_path, *args, **kwargs):
        """
        Args:
            src_path (str): Path to the src .npy file.
            tar_path (str): Path to the tar .npy file.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., augmentations, normalization, etc.).
        """
        super().__init__()
        self.srcs = torch.from_numpy(np.load(src_path))  # shape: (100, 1, 32, 32) (100, 1, 32, 32)
        self.tars = torch.from_numpy(np.load(tar_path))  # shape: (100, 1, 32, 32) (100, 1, 32, 32)
        self.pairs_per_epoch = kwargs['pairs_per_epoch'] if 'pairs_per_epoch' in kwargs else self.srcs.shape[0]
        self.mask_srcs = torch.from_numpy(np.load(mask_src_path))  # shape: (100, 1, 32, 32) (100, 1, 32, 32)
        self.mask_tars = torch.from_numpy(np.load(mask_tar_path))  # shape: (100, 1, 32, 32) (100, 1, 32, 32)

        self.datadeb = np.load("/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_ROOT_DIR/deb_data_one.npy")
        self.datadeb = torch.from_numpy(self.datadeb)

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, iix):
        if hasattr(self, 'paper_res_list'):
            idx = self.paper_res_list[iix]
        else:
            idx = iix

        src = self.srcs[idx]
        tar = self.tars[idx] 
        mask_src = self.mask_srcs[idx]
        mask_tar = self.mask_tars[idx]
        # print(src.shape, tar.shape, mask_src.shape, mask_tar.shape)
        # torch.Size([1, 64, 64]) torch.Size([1, 64, 64]) torch.Size([1, 64, 64]) torch.Size([1, 64, 64])
        # assert 3>333
        
    

        sample = {'src': src, 'tar': tar, 'srclabel': mask_src, 'tarlabel': mask_tar}   



        
        start=6
        f0 = self.datadeb[0].unsqueeze(0)
        fr = self.datadeb[start:start+1]
        return {'src': f0, 'tar': fr, 'srclabel': f0, 'tarlabel': fr}



        
        return sample

class Mnist2d_LDDMM_Optimize(Dataset):
    def __init__(self, src_path, tar_path, *args, **kwargs):
        """
        Args:
            src_path (str): Path to the src .npy file.
            tar_path (str): Path to the tar .npy file.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., augmentations, normalization, etc.).
        """
        super().__init__()
        self.srcs = torch.from_numpy(np.load(src_path))  # shape: (100, 1, 32, 32) (100, 1, 32, 32)
        self.tars = torch.from_numpy(np.load(tar_path))  # shape: (100, 1, 32, 32) (100, 1, 32, 32)
        self.pairs_per_epoch = kwargs['pairs_per_epoch'] if 'pairs_per_epoch' in kwargs else self.srcs.shape[0]
        # self.paper_res_list = [3,11,15,30,33,44,47,58,60,64,70,78,80,83,84,87]  #16
        # self.paper_res_list = [58,60,83,84]  #8
        # self.paper_res_list = [60,58,83,84]  #0
        # self.paper_res_list = [83,60,58,84]  #3
        # # self.paper_res_list = [84,60,58,83]  #4



        # self.paper_res_list = [40+13, 60, 60+4, 60+10, 60+7]
        # self.paper_res_list = [60, 60+4, 60+10, 60+7]
        # self.paper_res_list = [60+4, 60+10, 60+7]
        # self.paper_res_list = [60+10, 60+7]



    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, iix):
        if hasattr(self, 'paper_res_list'):
            idx = self.paper_res_list[iix]
        else:
            idx = iix

        idx = iix+20
        idx = iix+40
        idx = iix+54
        src = self.srcs[idx]
        tar = self.tars[idx]  



        src = np.load("/home/nellie/code/cvpr/ComplexNet/My2D/0000Metric/ForSlidesIPMI2025/Brain/Src.npy")
        tar = np.load("/home/nellie/code/cvpr/ComplexNet/My2D/0000Metric/ForSlidesIPMI2025/Brain/Tar.npy")
        src = torch.from_numpy(src).unsqueeze(0)
        tar = torch.from_numpy(tar).unsqueeze(0)
        # src = src[:,::2,::2]
        # tar = tar[:,::2,::2]



        # print(src.shape, tar.shape)
        # assert 3>333

        
        # print("inshape: ", inshape, "src shape: ", self.src.dtype, "tar shape: ", self.tar.shape)

        # self.src = np.load("/home/nellie/code/cvpr/ComplexNet/My2D/0000Metric/ForSlidesIPMI2025/Brain/Src.npy")
        # self.tar = np.load("/home/nellie/code/cvpr/ComplexNet/My2D/0000Metric/ForSlidesIPMI2025/Brain/Tar.npy")
        # self.src = torch.from_numpy(self.src).unsqueeze(0).unsqueeze(0).cuda()
        # self.tar = torch.from_numpy(self.tar).unsqueeze(0).unsqueeze(0).cuda()
        # self.src = self.src[:,:,::2,::2]
        # self.tar = self.tar[:,:,::2,::2]

        # print("inshape: ", inshape, "src shape: ", self.src.dtype, "tar shape: ", self.tar.shape)
        # assert 2>678


        sample = {'src': src, 'tar': tar}
        
        return sample

class Mnist2d(Dataset):
    def __init__(self, path, split="train", cur_img_size=64, img_size=64, nums=[0,1,2,3,4], pairs_per_epoch=1000, *args, **kwargs):  
        self.split = split
        self.cur_img_size = cur_img_size
        self.img_size = img_size
        self.nums = nums
        self.len_of_num = len(nums)
        self.pairs_per_epoch = pairs_per_epoch
        
        # path = "/home/nellie/data/DataReAffine_64/Mnist_affined_64.mat"
        self.dict = scipy.io.loadmat(path)

        factor = img_size/cur_img_size

       
        for i in range(10):
            #scale
            img = torch.from_numpy(self.dict[f"tag{i}"].astype(np.float32)) #[100, 64, 64]
            img = F.interpolate(img.unsqueeze(1), scale_factor=(factor), mode='bilinear', align_corners=False).squeeze(1)
            # print(img.shape)
            self.dict[f"tag{i}"] = img

    def __len__(self):
        return int(self.pairs_per_epoch/self.len_of_num)

    def __getitem__(self, idx):
        idx_size = 2 * self.len_of_num

        if self.split == "train":
            indices = np.random.randint(80, size=idx_size)
            max_idx = 80
        elif self.split == "valid":
            indices = np.random.randint(10, size=idx_size) + 80
            max_idx = 90
        elif self.split == "test":
            indices = np.random.randint(10, size=idx_size) + 90
            max_idx = 100
        
        if np.random.rand() > 0.5:
            srcidx = indices[0:self.len_of_num]; taridx = indices[self.len_of_num: 2*self.len_of_num]
        else:
            srcidx = indices[self.len_of_num: 2*self.len_of_num]; taridx = indices[0:self.len_of_num]

        #replace the same src and tar idx
        srcidx = np.array(srcidx)
        taridx = np.array(taridx)
        taridx = np.where(srcidx == taridx, (taridx + 1) % max_idx, taridx)

        src = [self.dict[f"tag{tag}"][srcidx[tag]] for tag in self.nums]
        tar = [self.dict[f"tag{tag}"][taridx[tag]] for tag in self.nums]
        
        src = torch.stack(src, dim=0)
        tar = torch.stack(tar, dim=0)

        sample = {'src': src, 'tar': tar}

        return sample


class Oasis3d_LDDMM_Optimize(Dataset):
    def __init__(self, path, idxpath, split, cur_img_size=64, img_size=64, pairs_per_epoch=100, templateIDX=-1, *args, **kwargs):
        """
        Args:
            src_path (str): Path to the src .npy file.
            tar_path (str): Path to the tar .npy file.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., augmentations, normalization, etc.).
        """
        super().__init__()

        #path = "/home/nellie/data/OASIS3-128/testlist_dw.txt"
        self.pairs_per_epoch = pairs_per_epoch
        self.split = split
        self.data_dir = ("/").join(path.split("/")[:-1]) #"/home/nellie/data/OASIS3-128"
        self.templateIDX = templateIDX
        self.factor = img_size/cur_img_size

        with open(path, "r") as file:
            self.file_content = file.read()
        self.file_content = self.file_content.split("\n")
        
        self.img_list = [f"{self.data_dir}/affined_images_128/affined_{item}.nii.gz" for item in self.file_content]
        self.label_list = [f"{self.data_dir}/affined_labels_128/affined_{item}.nii.gz"  for item in self.file_content]
        self.len_of_files=len(self.img_list)

        with open(idxpath, "r", encoding="utf-8") as file:
            idxs = json.load(file)
            self.srcidxs = idxs['srcidx']
            self.taridxs = idxs['taridx']
            

        self.paper_res_list = [4,6,7,10,15]

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, iix):
        # idx = idx #3D_1
        # idx = idx + 10 #3D_2
        # idx = idx + 20 #3D_3
        # idx = idx + 30 #3D_4
        # idx = idx + 40 #3D_5
        # idx = idx + 50 #3D_6
        # idx = idx + 60 #3D_7
        # idx = idx + 70 #3D_8
        # idx = idx + 80 #3D_9
        # idx = idx + 90 #3D_10
        # idx = idx + 20
        # assert 3>111
        # bad_list = [8 ,13 ,17 ,32 ,36 ,45 ,50 ,58 ,67 ,69 ,71 ,84 ,88 ,93 ,94 ,96 ]
        # bad_list = [8 ,13 ,17 ,32 ,36 ,45 ,50 ]
        # good_list = [6 ,21 ,22 ,30 ,33 ,44, 92 ]
        # comp_list = [1 ,2 ,3 ,9 ,10 ,11 ,12 ]
        # idx = bad_list[idx]
        # idx = self.paper_res_list[iix]


        idx = iix
        srcidx = self.srcidxs[idx]
        taridx = self.taridxs[idx]

        src = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[srcidx]))          #(256, 256, 256)
        srclabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[srcidx]))   #(256, 256, 256)
        tar = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[taridx]))          #(256, 256, 256)
        tarlabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[taridx]))   #(256, 256, 256)

        #convert to tensor
        src = torch.from_numpy(src).unsqueeze(0) #torch.Size([1, 256, 256, 256])
        srclabel = torch.from_numpy(srclabel).unsqueeze(0)
        tar = torch.from_numpy(tar).unsqueeze(0)
        tarlabel = torch.from_numpy(tarlabel).unsqueeze(0)

        
        
        if self.factor != 1:
            src = F.interpolate(src.unsqueeze(0), scale_factor=(self.factor), mode='trilinear', align_corners=False).squeeze(0)
            tar = F.interpolate(tar.unsqueeze(0), scale_factor=(self.factor), mode='trilinear', align_corners=False).squeeze(0)
            srclabel = F.interpolate(srclabel.unsqueeze(0), scale_factor=(self.factor), mode='nearest', align_corners=False).squeeze(0)
            tarlabel = F.interpolate(tarlabel.unsqueeze(0), scale_factor=(self.factor), mode='nearest', align_corners=False).squeeze(0)


        return {'src': src, 'tar': tar, 'srclabel': srclabel, 'tarlabel': tarlabel, 'indexs_src': torch.tensor([srcidx]), 'indexs_tar': torch.tensor([taridx])}

class Oasis1d_LDDMM_Optimize(Dataset):
    def __init__(self, path, idxpath, split, cur_img_size=64, img_size=64, pairs_per_epoch=100, templateIDX=-1, *args, **kwargs):
        super().__init__()
        self.pairs_per_epoch = pairs_per_epoch
        self.split = split
        self.data_dir = ("/").join(path.split("/")[:-1]) #"/home/nellie/data/OASIS3-128"
        self.templateIDX = templateIDX
        self.factor = img_size/cur_img_size

        # print(path)  #/home/nellie/data/OASIS1_128/test_valid_list_dw2.txt

        with open(path, "r") as file:
            self.file_content = file.read()
        self.file_content = self.file_content.split("\n")
        
        self.img_list = [f"{self.data_dir}/{item}/pad256_DT_128_nearest_aligned_norm.nii.gz" for item in self.file_content]
        self.label_list = [f"{self.data_dir}/{item}/pad256_DT_128_nearest_aligned_seg35.nii.gz"  for item in self.file_content]
        self.len_of_files=len(self.img_list)

        with open(idxpath, "r", encoding="utf-8") as file:
            idxs = json.load(file)
            self.srcidxs = idxs['srcidx']
            self.taridxs = idxs['taridx']

            if len(idxs['srcidx']) == 0:
                assert self.templateIDX > -1, "templateIDX must be assigned when srcidxs are empty"
                self.srcidxs = [self.templateIDX] * (len(self.taridxs) - 1)
                self.taridxs.pop(self.templateIDX)

        
        self.paper_res_list = [4,6,7,10,15]
        self.paper_res_list = [6]
        self.paper_res_list = [6, 4+16, 9+16, 12+16, 12+16, 14+16]
        #show this
        # self.paper_res_list = [4+16]
        # self.paper_res_list = [14+16]
        self.paper_res_list = [4+16, 14+16]


    def __len__(self):
        return self.pairs_per_epoch


    def __getitem__(self, iix):
        # idx = self.paper_res_list[iix]
        # idx = iix

        # idx = iix       #1_Steps
        # idx = iix + 10  #2_Steps
        # idx = iix + 20  #3_Steps
        # idx = iix + 30  #4_Steps
        # idx = iix + 40  #5_Steps
        # idx = iix + 50  #6_Steps
        # idx = iix + 60  #7_Steps
        
        # idx = iix + 70  #8_Steps
        # idx = iix + 80  #9_Steps
        # idx = iix + 90  #10_Steps

        # if iix == 0:
        #     idx = self.paper_res_list[iix]
        # else:
        #     idx = iix + 16

        # idx = iix+40+32
        # idx = iix+40

        # idx = iix + 60  #6_Steps
        
        # idx = iix + 50
  

        # idx = iix

        idx = iix + 12

        srcidx = self.srcidxs[idx]
        taridx = self.taridxs[idx]

        src = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[srcidx]))          #(256, 256, 256)
        srclabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[srcidx]))   #(256, 256, 256)
        tar = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[taridx]))          #(256, 256, 256)
        tarlabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[taridx]))   #(256, 256, 256)

        #convert to tensor
        src = torch.from_numpy(src).unsqueeze(0) #torch.Size([1, 256, 256, 256])
        srclabel = torch.from_numpy(srclabel).unsqueeze(0).type(torch.float32)
        tar = torch.from_numpy(tar).unsqueeze(0)
        tarlabel = torch.from_numpy(tarlabel).unsqueeze(0).type(torch.float32)

        
        
        if self.factor != 1:
            src = F.interpolate(src.unsqueeze(0), scale_factor=(self.factor), mode='trilinear', align_corners=False).squeeze(0)
            tar = F.interpolate(tar.unsqueeze(0), scale_factor=(self.factor), mode='trilinear', align_corners=False).squeeze(0)
            srclabel = F.interpolate(srclabel.unsqueeze(0), scale_factor=(self.factor), mode='nearest', align_corners=False).squeeze(0)
            tarlabel = F.interpolate(tarlabel.unsqueeze(0), scale_factor=(self.factor), mode='nearest', align_corners=False).squeeze(0)


        return {'src': src, 'tar': tar, 'srclabel': srclabel, 'tarlabel': tarlabel, 'indexs_src': torch.tensor([srcidx]), 'indexs_tar': torch.tensor([taridx])}

class Oasis3d(Dataset):
    def __init__(self, path, split, cur_img_size=64, img_size=64, pairs_per_epoch=100, templateIDX=-1, *args, **kwargs):
        #path = "/home/nellie/data/OASIS3-128/trainlist_dw.txt"
        #path = "/home/nellie/data/OASIS3-128/testlist_dw.txt"
        #path = "/home/nellie/data/OASIS3-128/testlist_disease.txt"
        self.pairs_per_epoch = pairs_per_epoch
        self.split = split
        self.data_dir = ("/").join(path.split("/")[:-1]) #"/home/nellie/data/OASIS3-128"
        self.templateIDX = templateIDX

        self.factor = img_size/cur_img_size

        with open(path, "r") as file:
            self.file_content = file.read()
        self.file_content = self.file_content.split("\n")
        
        self.img_list = [f"{self.data_dir}/affined_images_128/affined_{item}.nii.gz" for item in self.file_content]
        self.label_list = [f"{self.data_dir}/affined_labels_128/affined_{item}.nii.gz"  for item in self.file_content]
        self.len_of_files=len(self.img_list)

        print(path, self.len_of_files)
        # assert 3>234

    def __len__(self):
        return self.pairs_per_epoch


    def __getitem__(self, idx):
        if self.templateIDX == -1:
            indices = np.random.randint(self.len_of_files, size=2)
            srcidx = indices[0]; taridx = indices[1]
        else:
            srcidx = self.templateIDX
            indices = np.random.randint(self.len_of_files, size=1)
            taridx = indices[0]

        if taridx == srcidx:
            taridx = (taridx + 1) % self.len_of_files

        src = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[srcidx]))          #(256, 256, 256)
        srclabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[srcidx]))   #(256, 256, 256)

        tar = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[taridx]))          #(256, 256, 256)
        tarlabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[taridx]))   #(256, 256, 256)

        #convert to tensor
        src = torch.from_numpy(src).unsqueeze(0) #torch.Size([1, 256, 256, 256])
        srclabel = torch.from_numpy(srclabel).unsqueeze(0)
        tar = torch.from_numpy(tar).unsqueeze(0)
        tarlabel = torch.from_numpy(tarlabel).unsqueeze(0)

        
        src = (src - src.min()) / (src.max() - src.min())
        tar = (tar - tar.min()) / (tar.max() - tar.min())
        
        if self.factor != 1:
            src = F.interpolate(src.unsqueeze(0), scale_factor=(self.factor), mode='trilinear', align_corners=False).squeeze(0)
            tar = F.interpolate(tar.unsqueeze(0), scale_factor=(self.factor), mode='trilinear', align_corners=False).squeeze(0)
            srclabel = F.interpolate(srclabel.unsqueeze(0), scale_factor=(self.factor), mode='nearest', align_corners=False).squeeze(0)
            tarlabel = F.interpolate(tarlabel.unsqueeze(0), scale_factor=(self.factor), mode='nearest', align_corners=False).squeeze(0)


        return {'src': src, 'tar': tar, 'srclabel': srclabel, 'tarlabel': tarlabel, 'indexs_src': torch.tensor([srcidx]), 'indexs_tar': torch.tensor([taridx])}

class Oasis1d(Dataset):
    def __init__(self, path, split, cur_img_size=64, img_size=64, pairs_per_epoch=100, templateIDX=-1, *args, **kwargs):
        #path = "/home/nellie/data/OASIS3-128/trainlist_dw.txt"
        #path = "/home/nellie/data/OASIS3-128/testlist_dw.txt"
        #path = "/home/nellie/data/OASIS3-128/testlist_disease.txt"
        self.pairs_per_epoch = pairs_per_epoch
        self.split = split
        self.data_dir = ("/").join(path.split("/")[:-1]) #"/home/nellie/data/OASIS3-128"
        self.templateIDX = templateIDX

        self.factor = img_size/cur_img_size

        with open(path, "r") as file:
            self.file_content = file.read()
        self.file_content = self.file_content.split("\n")
        
        self.img_list = [f"{self.data_dir}/{item}/pad256_DT_128_nearest_aligned_norm.nii.gz" for item in self.file_content]
        self.label_list = [f"{self.data_dir}/{item}/pad256_DT_128_nearest_aligned_seg35.nii.gz"  for item in self.file_content]
        self.len_of_files=len(self.img_list)

        print(path, self.len_of_files)
        

    def __len__(self):
        return self.pairs_per_epoch


    def __getitem__(self, idx):
        if self.templateIDX == -1:
            indices = np.random.randint(self.len_of_files, size=2)
            srcidx = indices[0]; taridx = indices[1]
        else:
            srcidx = self.templateIDX
            indices = np.random.randint(self.len_of_files, size=1)
            taridx = indices[0]

        if taridx == srcidx:
            taridx = (taridx + 1) % self.len_of_files

        src = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[srcidx]))          #(256, 256, 256)
        srclabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[srcidx]))   #(256, 256, 256)

        tar = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[taridx]))          #(256, 256, 256)
        tarlabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[taridx]))   #(256, 256, 256)

        #convert to tensor
        src = torch.from_numpy(src).unsqueeze(0) #torch.Size([1, 256, 256, 256])
        srclabel = torch.from_numpy(srclabel).unsqueeze(0)
        tar = torch.from_numpy(tar).unsqueeze(0)
        tarlabel = torch.from_numpy(tarlabel).unsqueeze(0)

        src = (src - src.min()) / (src.max() - src.min())
        tar = (tar - tar.min()) / (tar.max() - tar.min())
        
        if self.factor != 1:
            src = F.interpolate(src.unsqueeze(0), scale_factor=(self.factor), mode='trilinear', align_corners=False).squeeze(0)
            tar = F.interpolate(tar.unsqueeze(0), scale_factor=(self.factor), mode='trilinear', align_corners=False).squeeze(0)
            srclabel = F.interpolate(srclabel.unsqueeze(0), scale_factor=(self.factor), mode='nearest', align_corners=False).squeeze(0)
            tarlabel = F.interpolate(tarlabel.unsqueeze(0), scale_factor=(self.factor), mode='nearest', align_corners=False).squeeze(0)

        return {'src': src, 'tar': tar, 'srclabel': srclabel, 'tarlabel': tarlabel, 'indexs_src': torch.tensor([srcidx]), 'indexs_tar': torch.tensor([taridx])}

class MixedOasis(Dataset):
    def __init__(self, path_OASIS1D, path_OASIS3D, cur_img_size=64, img_size=64, pairs_per_epoch=100, templateIDX=-1, *args, **kwargs):
        self.pairs_per_epoch = pairs_per_epoch
        self.templateIDX = templateIDX
        self.factor = img_size/cur_img_size

        ## OASIS1D
        self.data_dir_OASIS1D = ("/").join(path_OASIS1D.split("/")[:-1]) #"/home/nellie/data/OASIS3-128"
        with open(path_OASIS1D, "r") as file:
            self.file_content_OASIS1D = file.read()
        self.file_content_OASIS1D = self.file_content_OASIS1D.split("\n")
        self.img_list_OASIS1D = [f"{self.data_dir_OASIS1D}/{item}/pad256_DT_128_nearest_aligned_norm.nii.gz" for item in self.file_content_OASIS1D]
        self.label_list_OASIS1D = [f"{self.data_dir_OASIS1D}/{item}/pad256_DT_128_nearest_aligned_seg35.nii.gz"  for item in self.file_content_OASIS1D]
        self.len_of_files_OASIS1D=len(self.img_list_OASIS1D)
        print(path_OASIS1D, self.len_of_files_OASIS1D)

        ## OASIS3D
        self.data_dir_OASIS3D = ("/").join(path_OASIS3D.split("/")[:-1]) #"/home/nellie/data/OASIS3-128"
        with open(path_OASIS3D, "r") as file:
            self.file_content_OASIS3D = file.read()
        self.file_content_OASIS3D = self.file_content_OASIS3D.split("\n")
        self.img_list_OASIS3D = [f"{self.data_dir_OASIS3D}/affined_images_128/affined_{item}.nii.gz" for item in self.file_content_OASIS3D]
        self.label_list_OASIS3D = [f"{self.data_dir_OASIS3D}/affined_labels_128/affined_{item}.nii.gz"  for item in self.file_content_OASIS3D]
        self.len_of_files_OASIS3D=len(self.img_list_OASIS3D)
        print(path_OASIS3D, self.len_of_files_OASIS3D)

    def __len__(self):
        return self.pairs_per_epoch


    def __getitem__(self, idx):
        ###################################################################################
        if random.random() < 0.5:
            self.img_list = self.img_list_OASIS1D
            self.label_list = self.label_list_OASIS1D
            self.len_of_files = self.len_of_files_OASIS1D
        else:
            self.img_list = self.img_list_OASIS3D
            self.label_list = self.label_list_OASIS3D
            self.len_of_files = self.len_of_files_OASIS3D

        
        
        ###################################################################################
        if self.templateIDX == -1:
            indices = np.random.randint(self.len_of_files, size=2)
            srcidx = indices[0]; taridx = indices[1]
        else:
            srcidx = self.templateIDX
            indices = np.random.randint(self.len_of_files, size=1)
            taridx = indices[0]

        if taridx == srcidx:
            taridx = (taridx + 1) % self.len_of_files

        src = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[srcidx]))          #(256, 256, 256)
        srclabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[srcidx]))   #(256, 256, 256)

        tar = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[taridx]))          #(256, 256, 256)
        tarlabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[taridx]))   #(256, 256, 256)

        #convert to tensor
        src = torch.from_numpy(src).unsqueeze(0) #torch.Size([1, 256, 256, 256])
        srclabel = torch.from_numpy(srclabel).unsqueeze(0); srclabel = srclabel.type(torch.float32)
        tar = torch.from_numpy(tar).unsqueeze(0)
        tarlabel = torch.from_numpy(tarlabel).unsqueeze(0); tarlabel = tarlabel.type(torch.float32)

        src = (src - src.min()) / (src.max() - src.min())
        tar = (tar - tar.min()) / (tar.max() - tar.min())
        
        if self.factor != 1:
            src = F.interpolate(src.unsqueeze(0), scale_factor=(self.factor), mode='trilinear', align_corners=False).squeeze(0)
            tar = F.interpolate(tar.unsqueeze(0), scale_factor=(self.factor), mode='trilinear', align_corners=False).squeeze(0)
            srclabel = F.interpolate(srclabel.unsqueeze(0), scale_factor=(self.factor), mode='nearest', align_corners=False).squeeze(0)
            tarlabel = F.interpolate(tarlabel.unsqueeze(0), scale_factor=(self.factor), mode='nearest', align_corners=False).squeeze(0)

        # return {'src': src, 'tar': tar, 'srclabel': srclabel, 'tarlabel': tarlabel, 'indexs_src': torch.tensor([srcidx]), 'indexs_tar': torch.tensor([taridx])}
        
        # print(src.shape, tar.shape, srclabel.shape, tarlabel.shape)
        # print(src.dtype, tar.dtype, srclabel.dtype, tarlabel.dtype)
        # print("\n\n\n")
        
        return {'src': src, 'tar': tar, 'srclabel': srclabel, 'tarlabel': tarlabel}

class PairDense(Dataset):
    def __init__(self, path, pathmask, split="train", cur_img_size=64, img_size=64, pairs_per_epoch=1000, *args, **kwargs):
        self.split = split
        self.cur_img_size = cur_img_size
        self.img_size = img_size
        self.pairs_per_epoch = pairs_per_epoch

        # path="/home/nellie/revserse_cine64_12slice.mat"
        # pathmask="/home/nellie/revserse_cine64_12slice_mask.mat"
        self.data = sio.loadmat(path)['data']; self.data = torch.from_numpy(self.data.astype(np.float32))
        self.mask = sio.loadmat(pathmask)['data']; self.mask = torch.from_numpy(self.mask.astype(np.float32))
        
        # (527, 12, 64, 64) 
        # (527, 12, 64, 64) 
        factor = img_size/cur_img_size 
        if factor != 1:
            self.data = F.interpolate(self.data.unsqueeze(1), scale_factor=(factor), mode='bilinear', align_corners=False).squeeze(1)
            self.mask = F.interpolate(self.mask.unsqueeze(1), scale_factor=(factor), mode='nearest', align_corners=False).squeeze(1)

        if self.split == "train":
            self.data = self.data[:420]; self.mask = self.mask[:420]      #5052
        elif self.split == "valid":
            self.data = self.data[420:470]; self.mask = self.mask[420:470] #624
        elif self.split == "test":
            self.data = self.data[470:]; self.mask = self.mask[470:]    #648

        self.length_of_seq = self.data.shape[1]
        self.data_len = self.data.shape[0]

        self.len_of_num = 5
        
        print(self.data.shape, self.mask.shape) #torch.Size([5052, 64, 64]) torch.Size([5052, 64, 64])

    def __len__(self):
        return  int(self.pairs_per_epoch/self.len_of_num)

    def __getitem__(self, iix):
        idx = np.random.randint(self.data_len, size=1)
        data = self.data[idx][0]
        mask = self.mask[idx][0]
        # print(data.shape, mask.shape) 
        # # torch.Size([12, 64, 64])torch.Size([12, 64, 64])
       
        
        idx_size = 2*self.len_of_num
        indices = np.random.randint(self.length_of_seq, size=idx_size)
        if np.random.rand() > 0.5:
            srcidx = indices[0:self.len_of_num]; taridx = indices[self.len_of_num: 2*self.len_of_num]
        else:
            srcidx = indices[self.len_of_num: 2*self.len_of_num]; taridx = indices[0:self.len_of_num]

        #replace the same src and tar idx
        max_idx = self.length_of_seq
        srcidx = np.array(srcidx)
        taridx = np.array(taridx)
        taridx = np.where(srcidx == taridx, (taridx + 4) % max_idx, taridx)

        src = data[srcidx]
        tar = data[taridx]
        mask_src = mask[srcidx]
        mask_tar = mask[taridx]

        
        

        sample = {'src': src, 'tar': tar, 'srclabel': mask_src, 'tarlabel': mask_tar}

        return sample

oasis3_src = [0,9,36,12,24,13,20,15,24,14,31,35,34,36,17,31,11,0,12,6,3,15,15,5,0,36,3,13,10,23,30,18,14,1,2,0,30,2,13,8,14,11,13,24,25,33,6,7,15,11,29,24,12,22,32,37,2,20,35,23,26,27,33,24,36,5,17,27,3,33,20,5,2,22,6,19,25,27,0,36,3,23,7,0,25,22,4,7,5,15,4,27,34,23,16,9,6,7,25,6,25,23,30,33,23,7,0,17,1,4,10,20,35,10,3,1,36,34,28,9,7,20,21,7,0,9,21,0,34,14,33,4,2,36,13,34,26,15,0,7,30,18,28,36,29,2,10,1,34,19,20,2,5,36,17,8,22,1,12,8,3,27,18,17,30,24,33,33,32,25,9,3,18,29,1,28,2,36,12,32,26,24,28,24,19,10,35,19,20,2,33,1,11,2,5,15,21,25,6,26]
oasis3_tar = [3,19,23,24,23,25,16,0,29,32,32,23,28,5,15,1,35,27,20,4,12,14,23,21,31,0,29,21,0,2,35,35,27,36,11,32,13,3,8,26,3,3,11,29,16,19,36,13,18,15,1,24,3,5,11,10,27,23,3,9,3,10,21,34,0,16,1,36,35,29,29,37,27,11,7,29,9,9,17,16,8,29,8,17,36,14,18,19,13,15,28,10,18,18,30,35,8,16,15,17,2,24,23,20,27,22,9,36,3,16,30,31,27,18,12,0,35,3,33,32,33,8,27,20,5,30,37,25,23,26,6,29,35,15,4,23,26,12,16,13,15,7,2,27,33,16,26,0,15,37,6,6,37,13,31,25,37,37,26,14,5,25,14,4,24,36,14,17,34,0,18,12,7,3,20,26,9,25,10,37,35,18,8,28,3,0,27,34,0,33,23,10,6,12,8,1,12,13,21,30]

###########   For T-paired test  Make up MELBA experiments ##############
class D3_dataset(Dataset):
    def __init__(self, path, split, template=-1, testlen=10, size_half=False, transform=None, test_type='norm'):
        # print(path, "  split  ", split, "  template  ", template, "  testlen  ", testlen, "  size_half  ", size_half, "  transform  ", transform)
        # #    split   test   template   -1   testlen   200   size_half   False   transform   None
        # assert 3>333

        path = "/home/nellie/data/OASIS3-128/testlist_dw.txt"
        self.path = "/home/nellie/data/OASIS3-128/testlist_dw.txt" #/home/nellie/data/OASIS3-128/testlist_dw.txt
        
        self.testlen = 200
        self.size_half = False
        self.transform = transform  # using transform in torch!
        self.split = "test"
        self.data_dir = ("/").join(path.split("/")[:-1])
        self.test_type = test_type
        self.template = template
        
        # print(len(oasis3_src), len(oasis3_tar))
        # assert 2>198


        # print(path)
        # print("self.data_dir   ", self.data_dir) #/home/nellie/data/OASIS3-128
        # assert 3>333
        # /home/nellie/data/neurite_oasis1/subdirectories_list.txt
        # self.data_dir    /home/nellie/data/neurite_oasis1
        # assert 4>8
        
        with open(path, "r") as file:
            self.file_content = file.read()
        self.file_content = self.file_content.split("\n")
        if self.split == "train":
            self.img_list = [self.data_dir+"/affined_images_128/affined_"+item+".nii.gz" for item in self.file_content]
            self.label_list = [self.data_dir+"/affined_labels_128/affined_"+item+".nii.gz"  for item in self.file_content]
        else:
            if test_type == 'norm':
                self.img_list = [self.data_dir+"/affined_images_128/affined_"+item+".nii.gz" for item in self.file_content]
                self.label_list = [self.data_dir+"/affined_labels_128/affined_"+item+".nii.gz"  for item in self.file_content]
            elif test_type == 'disease':
                self.img_list = [self.data_dir+"/affined_images_256/affined_"+item+".nii.gz" for item in self.file_content]
                self.label_list = [self.data_dir+"/affined_labels_256/affined_"+item+".nii.gz"  for item in self.file_content]
            elif test_type == 'oasis1_downsample':
                self.img_list = [self.data_dir+"/"+item+"/pad256_DT_128_downsample_aligned_norm.nii.gz"  for item in self.file_content]
                self.label_list = [self.data_dir+"/"+item+"/pad256_DT_128_downsample_aligned_seg35.nii.gz"  for item in self.file_content]
            elif test_type == 'oasis1_nearest':
                print("\n\n\n^^^^^^^^^^^^^^^^^^^^")
                self.img_list = [self.data_dir+"/"+item+"/pad256_DT_128_nearest_aligned_norm.nii.gz"  for item in self.file_content]
                self.label_list = [self.data_dir+"/"+item+"/pad256_DT_128_nearest_aligned_seg35.nii.gz"  for item in self.file_content]
                
        # for i in range(len(self.img_list)):
        #     if not os.path.exists(self.img_list[i]):
        #         print(self.img_list[i])
        #     assert 4>8

        self.len = self.__len__()
        self.dirlen=len(self.img_list)

    def __len__(self):
        return 100 if self.split=="train" else self.testlen
        # return 4 if self.split=="train" else 4


    def __getitem__(self, idx):
        
        if self.template == -1 or self.split == "train":
            # print("~~",idx, "~~", idx, "~~",idx)
            # assert 3>198



            # t1 = default_timer()
            prob_same = 0.5
            indices = np.random.randint(self.dirlen, size=2)
            # print(indices)
            if np.random.rand() > prob_same:
                srcidx = indices[0]; taridx = indices[1]
            else:
                srcidx = indices[1]; taridx = indices[0]

            if srcidx > len(self.img_list):
                srcidx = srcidx % len(self.img_list)
            if taridx > len(self.img_list):
                taridx = taridx % len(self.img_list)

            srcidx = oasis3_src[idx]
            taridx = oasis3_tar[idx]
            # print(srcidx, taridx)
            # assert 3>198999
            src = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[srcidx]))          #(256, 256, 256)
            srclabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[srcidx]))   #(256, 256, 256)

            tar = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[taridx]))          #(256, 256, 256)
            tarlabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[taridx]))   #(256, 256, 256)
        else:
            # print(self.template)
            # assert 3>111
            srcidx = idx
            taridx = self.template

            if srcidx > len(self.img_list):
                srcidx = srcidx % len(self.img_list)
            if taridx > len(self.img_list):
                taridx = taridx % len(self.img_list)

            src = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[srcidx]))          #(256, 256, 256)
            srclabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[srcidx]))   #(256, 256, 256)
            tar = sitk.GetArrayFromImage(sitk.ReadImage(self.img_list[taridx]))          #(256, 256, 256)
            tarlabel = sitk.GetArrayFromImage(sitk.ReadImage(self.label_list[taridx]))   #(256, 256, 256)

        # print("src   ", self.img_list[srcidx], "  tar  ", self.img_list[taridx])
        src = (src - src.min()) / (src.max() - src.min())
        tar = (tar - tar.min()) / (tar.max() - tar.min())
        

        if(self.size_half or self.test_type=="disease"):
            src = src[::2,::2,::2]
            srclabel = srclabel[::2,::2,::2]
            tar = tar[::2,::2,::2]
            tarlabel = tarlabel[::2,::2,::2]

        # print(f"@#$%^&  src shape: {src.shape}, tar shape: {tar.shape}  *&^%$")
        sample = {'src': torch.from_numpy(src).unsqueeze(0), 'tar': torch.from_numpy(tar).unsqueeze(0), 'srclabel': torch.from_numpy(srclabel).unsqueeze(0), 'tarlabel': torch.from_numpy(tarlabel).unsqueeze(0)}



        if self.transform:
            sample = self.transform(sample)
        return sample