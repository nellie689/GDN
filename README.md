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


## 🔬Quick Review of Code Logic

**🔹1. Parameter Settings:**
Different parameter configurations are defined in: the MELBA_configs directory.

**🔹2. Model Architecture – U-Net Backbone:**
The architecture of the main model using the U-Net shape backbone is implemented in: /GDN/lvdm/modules/modules2D3D/unet.py

**🔹3. Model Architecture – Neural Operator:**
The architecture of the Neural Operator component is defined in: /GDN/lvdm/modules/modules2D3D/nop.py

**🔹4. Algorithm Workflow:**
The overall workflow of the algorithm is managed using PyTorch Lightning. The relevant functions are located in: /GDN/lvdm/modules/modules2D3D/MELBAgdnAlter.py
