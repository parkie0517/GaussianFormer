# Installation
Our code is tested on the following environment.

## 1. Create conda environment
```bash
conda create -n selfocc python=3.8.16
conda activate selfocc
```

## 2. Install PyTorch
```bash
conda install nvidia/label/cuda-11.8.0::cuda-nvcc
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install einops
pip install jaxtyping
```

## 3. Install packages from MMLab
```bash
pip install openmim
mim install mmcv==2.0.1
mim install mmdet==3.0.0
mim install mmsegmentation==1.0.0
mim install mmdet3d==1.1.1
```

## 4. Install other packages
```bash
pip install spconv-cu117
pip install timm
```

## 4. Install pointops
Clone this repository.
```bash
git clone https://github.com/POSTECH-CVLab/point-transformer.git
```
Comment `#include <THC/THC.h>` in the below files.
```bash

./point-transformer/lib/pointops/src/aggregation/aggregation_cuda.cpp
./point-transformer/lib/pointops/src/grouping/grouping_cuda.cpp
./point-transformer/lib/pointops/src/interpolation/interpolation_cuda.cpp
./point-transformer/lib/pointops/src/knnquery/knnquery_cuda.cpp
./point-transformer/lib/pointops/src/sampling/sampling_cuda.cpp
./point-transformer/lib/pointops/src/subtraction/subtraction_cuda.cpp
```
Install pointops.
```bash
cd ./point-transformer/lib/pointops
python3 setup.py install
```

## 4. Install custom CUDA ops
```bash
conda install conda-forge::cudatoolkit
conda install nvidia/label/cuda-11.8.0::cuda-cudart-dev
cd model/encoder/gaussian_encoder/ops && pip install -e .
cd model/head/localagg && pip install -e .
# for GaussianFormer-2
cd model/head/localagg_prob && pip install -e .
cd model/head/localagg_prob_fast && pip install -e .
```

## 5. (Optional) For visualization
```bash
pip install pyvirtualdisplay mayavi matplotlib==3.7.2 PyQt5
```