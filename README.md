## 🧭

Learning Geodesics of Geometric Shape Deformations From Images. (Under review of Journal MELBA)


## 📌 Setup

The main dependencies are listed below, the other packages can be easily installed with "pip install" according to the hints when running the code.

* python=3.10
* pytorch=2.0.0
* cuda11.8
* matplotlib
* numpy
* SimpleITK
* LagoMorph

Tips:
LagoMorph contains the core implementation for solving the geodesic shooting equation (i.e., the EPDiff equation) under the LDDMM framework.
The code repository is available at: https://github.com/jacobhinkle/lagomorph


## 🚀 Usage

Below is a **QuickStart guide** on how to train and test.

**Train**, run:

cd MELBA_main
python MixOASIS3D.py


**Test**, run:

cd MELBA_main
python MixOASIS3D_GDN.ipynb


## 🔹Qurick Review of Code logic

1. Put different paparmeter setting in MELBA_configs
2. The architecture of the main model, U-Net shape backbone is in the file: /GDN/lvdm/modules/modules2D3D/unet.py
3. The architecture of the main model, Neural Operator is in the file: /GDN/lvdm/modules/modules2D3D/nop.py
4. For the overflow of the whole algorithm, it's managed with PytorchLightning and the related function are in file: /GDN/lvdm/modules/modules2D3D/MELBAgdnAlter.py
