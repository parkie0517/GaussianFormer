## Getting Started

### Installation
Follow instructions [HERE](docs/installation.md) to prepare the environment.
<!-- The environment is almost the same as [SelfOcc](https://github.com/huang-yh/SelfOcc) except for two additional CUDA operations.

```
1. Follow instructions in SelfOcc to prepare the environment. Not that we do not need packages related to NeRF, so feel safe to skip them.
2. cd model/encoder/gaussian_encoder/ops && pip install -e .  # deformable cross attention with image features
3. cd model/head/localagg && pip install -e .  # Gaussian-to-Voxel splatting
``` -->

### Data Preparation
1. Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download).

2. Download the occupancy annotations from SurroundOcc [HERE](https://github.com/weiyithu/SurroundOcc) and unzip it.

3. Download pkl files [HERE](https://cloud.tsinghua.edu.cn/d/bb96379a3e46442c8898/).

**Folder structure**
```
GaussianFormer
├── ...
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
│   ├── nuscenes_cam/
│   │   ├── nuscenes_infos_train_sweeps_occ.pkl
│   │   ├── nuscenes_infos_val_sweeps_occ.pkl
│   │   ├── nuscenes_infos_val_sweeps_lid.pkl
│   ├── surroundocc/
│   │   ├── samples/
│   │   |   ├── xxxxxxxx.pcd.bin.npy
│   │   |   ├── ...
```

### Reproducing the Resutls
| Name  | Type | #Gaussians | mIoU | IoU  |
| :---: | :---: | :---: | :---: | :---: |
| Baseline | GaussianFormer | 144000 |  |  |
| NonEmpty | GaussianFormer | 25600  | 17.57 | 28.79 |

### Inference
We provide the following checkpoints trained on the SurroundOcc dataset:

| Name  | Type | #Gaussians | mIoU | Config | Weight |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | GaussianFormer | 144000 | 19.10 | [config](config/nuscenes_gs144000.py) | [weight](https://cloud.tsinghua.edu.cn/seafhttp/files/b751f8f7-9a28-4be7-aa4e-385c4349f1b0/state_dict.pth) |
| NonEmpty | GaussianFormer | 25600  | 19.31 | [config](config/nuscenes_gs25600_solid.py) | [weight](https://cloud.tsinghua.edu.cn/f/d1766fff8ad74756920b/?dl=1) |
| Prob-64  | GaussianFormer-2 | 6400 | 20.04 | [config](config/prob/nuscenes_gs6400.py) | [weight](https://cloud.tsinghua.edu.cn/f/d041974bd900419fb141/?dl=1) |
| Prob-128 | GaussianFormer-2 | 12800 | 20.08 | [config](config/prob/nuscenes_gs12800.py) | [weight](https://cloud.tsinghua.edu.cn/f/b6038dca93574244ad57/?dl=1) |
| Prob-256 | GaussianFormer-2 | 25600 | 20.33 | [config](config/prob/nuscenes_gs25600.py) | [weight](https://cloud.tsinghua.edu.cn/f/e30c9c92e4344783a7de/?dl=1) |


```
python eval.py --py-config config/xxxx.py --work-dir out/xxxx/ --resume-from out/xxxx/state_dict.pth
# for example
CUDA_VISIBLE_DEVICES=0,1 python eval.py --py-config config/nuscenes_gs144000.py --work-dir out/nuscenes_gs144000/ --resume-from out/nuscenes_gs144000/state_dict.pth
CUDA_VISIBLE_DEVICES=0,1 python eval.py --py-config config/nuscenes_gs25600_solid.py --work-dir out/nuscenes_gs25600_solid/ --resume-from out/nuscenes_gs25600_solid/state_dict.pth

# for gformer2
CUDA_VISIBLE_DEVICES=0,1 python eval.py --py-config config/prob/nuscenes_gs6400.py --work-dir out/prob/nuscenes_gs6400/ --resume-from out/prob/nuscenes_gs6400/state_dict.pth
```

### Train

Download the pretrained weights for the image backbone [HERE](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth) and put it inside ckpts.
```bash
CUDA_VISIBLE_DEVIES=0,1,2,3 python train.py --py-config config/xxxx.py --work-dir out/xxxx
# for example
CUDA_VISIBLE_DEVICES=0,1 python train.py --py-config config/nuscenes_gs144000.py --work-dir out/nuscenes_gs144000
CUDA_VISIBLE_DEVICES=0,1 python train.py --py-config config/nuscenes_gs25600_solid.py --work-dir out/nuscenes_gs25600_solid
```


### Visualize
Install packages for visualization according to the [documentation](docs/installation.md). Here is an example command where you can change --num-samples and --vis-index.
```bash
CUDA_VISIBLE_DEVICES=0 python visualize.py --py-config config/nuscenes_gs25600_solid.py --work-dir out/nuscenes_gs25600_solid --resume-from out/nuscenes_gs25600_solid/state_dict.pth --vis-occ --vis-gaussian --num-samples 3 --model-type base
```

## Related Projects

Our work is inspired by these excellent open-sourced repos:
[TPVFormer](https://github.com/wzzheng/TPVFormer)
[PointOcc](https://github.com/wzzheng/PointOcc)
[SelfOcc](https://github.com/huang-yh/SelfOcc)
[SurroundOcc](https://github.com/weiyithu/SurroundOcc) 
[OccFormer](https://github.com/zhangyp15/OccFormer)
[BEVFormer](https://github.com/fundamentalvision/BEVFormer)

Our code is originally based on [Sparse4D](https://github.com/HorizonRobotics/Sparse4D) and migrated to the general framework of [SelfOcc](https://github.com/huang-yh/SelfOcc).

