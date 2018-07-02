
# Recurrent 3D pose Sequence Machines.
By Mude Lin, Liang Lin, Xiaodan Liang, Keze Wang and Hui Cheng.

## Introduction

Estimating 3D human pose from monocular images has many applications, includings human computer interaction, virtual reality, 
motion-sensing games and so on. We propose a novel Recurrent 3D Pose Sequence Machines(RPSM).
You can use the code to train/evaluate a network for 3D pose estimation task. For more details, please refer to our paper.

## Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Models](#models)


## Installation

### Pre-require
1. Ubuntu 14.04
2. NVIDIA GPU with 6GB graphic memory
3. Torch
4. OpenCV 2.4.9 pyhon interface
5. torch-hdf5
6. nngraph
### Get the code. We will call the directory that you cloned into `$RPSM_ROOT`
```Shell
https://github.com/Geekking/RPSM.git
cd RPSM
```

## Preparation
Please see util/preprocess

## Train/Eval
### 1. Train your model and evaluate the model.

#### 1.1 Pretrain shared 2D pose module with MPII data.
The shared 2D pose module is trained with [CPM code](https://github.com/shihenw/convolutional-pose-machines-release). and converted to torch module, we have provided a model which are stored at `models/torch_model/caffe_d2_pose_module_shared.t7` 
in our Release models. You could unzip our provided model and run `cp models/torch_model/caffe_d2_pose_module_shared.t7 $RPSM_ROOT/models/torch_model/caffe_d2_pose_module_shared.t7`

#### 1.2 main training phase
RPSM with 3 stages versions

    ``` Shell
    cd $RPSM_ROOT/scrpts/rpsm and run bash train_rpsm_pretrained_rho3_t5.sh
    After 50 epoches, the MPJPE metircs should be about 73. 
    ```

the snapshots will be stored in exp/h3m/rpsm_1024_rho3_t5

### 2. Evaluate snapshots.
   
    ``` Shell
    Modify the `refineModel` parameter to you model in scripts/rpsm/test_rpsm_rho3.sh at line 17.
    and `cd scripts/rpsm/  && bash test_rpsm_rho3.sh`

    ```
   
## Models
Download trained model from [Baidu Yun](https://pan.baidu.com/s/1GUguVwGcRwKkFIXM5sQ8Cg), and cd scripts/rpsm/  && bash test_rpsm_rho3.sh.

Our predicted result on Human 3.6M dataset could be found at [Baidu Yun](https://pan.baidu.com/s/1i4TqeFV)





## Citation
If you like this work, please consider citing:

    @inproceedings{linCVPR17RPSM,
        title = {Recurrent 3D Pose Sequence Machines},
        author = {Mude Lin and Liang Lin and Xiaodan Liang and Keze Wang and Hui Chen},
        booktitle = {CVPR},
        year = {2017}
    }
