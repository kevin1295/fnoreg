# Experiment Record

- [Experiment Record](#experiment-record)
  - [环境配置](#环境配置)
  - [oasis\_exp141](#oasis_exp141)
  - [oasis\_exp142](#oasis_exp142)
  - [oasis\_v\_exp40](#oasis_v_exp40)
  - [oasis\_v\_exp41](#oasis_v_exp41)

## 环境配置

环境配置方法  
```sh
conda create -n fnoreg python=3.10.12
pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 torchvision==0.16.0+cu118 -f https://mirrors.aliyun.com/pytorch-wheels/cu118
pip install numpy==1.26.3 setuptools<82
pip install -r requirements
```

启动 tensorboard
```sh
tensorboard --logdir experiments_fourier
```

## oasis_exp141

- **模型**：FNOReg 2D
- **数据集**：Oasis 2D
- **训练分辨率**：$160 \times 192$

修改 [params.json](./deep_fourier_reg/params.json)
```json
  "model_name": "convfno",
```

训练  
`python train_oasis.py --exp_num 141 --size 160`

验证  
`python evaluate_oasis.py --exp_num 141 --size 160`  
`python evaluate_oasis.py --exp_num 141 --size 80`

```sh
(fnoreg) fish@CudaAcc:~/fnoreg/deep_fourier_reg$ python evaluate_oasis.py --exp_num 141 --size 160 --ckpt_epoch 60


Total Trainable Params: 22119138
Computing metrics...
  0%|                                                                       | 0/400 [00:00<?, ?it/s]/home/fish/miniconda3/envs/fnoreg/
lib/python3.10/site-packages/torch/nn/modules/conv.py:456: UserWarning: Applied workaround for CuDNN issue, install nvrtc.so (Triggere
d internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:80.)
  return F.conv2d(input, weight, bias, self.stride,
/home/fish/miniconda3/envs/fnoreg/lib/python3.10/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming re
lease, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
100%|█████████████████████████████████████████████████████████████| 400/400 [00:05<00:00, 74.74it/s]
0.00017968168 6.283162
--- Evaluation results for convfno ---

Mean initial dice: 0.544
Mean dice after registration: 0.769
Standard deviation of dice values: 0.038
Mean percent of folded pixels: 0.702
Std of percent of folded pixels: 0.368
Mean sdlogJ: 0.471
Std sdlogJ: 0.068
Mean inference time: 0.009 seconds
```

```sh
(fnoreg) fish@CudaAcc:~/fnoreg/deep_fourier_reg$ python evaluate_oasis.py --exp_num 141 --size 80 --ckpt_epoch 60


Total Trainable Params: 22119138
Computing metrics...
  0%|                                                                       | 0/400 [00:00<?, ?it/s]/home/fish/miniconda3/envs/fnoreg/
lib/python3.10/site-packages/torchvision/transforms/functional.py:1603: UserWarning: The default value of the antialias parameter of a
ll the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent acr
oss the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None 
(current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use anti
aliasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(
antialias=True).
  warnings.warn(
/home/fish/miniconda3/envs/fnoreg/lib/python3.10/site-packages/torch/nn/modules/conv.py:456: UserWarning: Applied workaround for CuDNN
 issue, install nvrtc.so (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:80.)
  return F.conv2d(input, weight, bias, self.stride,
/home/fish/miniconda3/envs/fnoreg/lib/python3.10/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming re
lease, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)     
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
100%|█████████████████████████████████████████████████████████████| 400/400 [00:04<00:00, 94.73it/s]
0.0006155281 6.2824993
--- Evaluation results for convfno ---

Mean initial dice: 0.551
Mean dice after registration: 0.743
Standard deviation of dice values: 0.040
Mean percent of folded pixels: 6.029
Std of percent of folded pixels: 1.568
Mean sdlogJ: 0.706
Std sdlogJ: 0.058
Mean inference time: 0.008 seconds
```

## oasis_exp142

- **模型**：Gated FNOReg 2D
- **数据集**：Oasis 2D
- **训练分辨率**：$160 \times 192$

修改 [params.json](./deep_fourier_reg/params.json)
```json
  "model_name": "gated_convfno",
```

训练  
`python train_oasis.py --exp_num 142 --size 160`

验证  
`python evaluate_oasis.py --exp_num 142 --size 160`  
`python evaluate_oasis.py --exp_num 142 --size 80`

```
(fnoreg) fish@CudaAcc:~/fnoreg/deep_fourier_reg$ python evaluate_oasis.py --exp_num 142 --size 160 --ckpt_epoch 60


Total Trainable Params: 22125282
Computing metrics...
  0%|                                                                       | 0/400 [00:00<?, ?it/s]/home/fish/miniconda3/envs/fnoreg/
lib/python3.10/site-packages/torch/nn/modules/conv.py:456: UserWarning: Applied workaround for CuDNN issue, install nvrtc.so (Triggere
d internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:80.)
  return F.conv2d(input, weight, bias, self.stride,
/home/fish/miniconda3/envs/fnoreg/lib/python3.10/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming re
lease, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
100%|█████████████████████████████████████████████████████████████| 400/400 [00:05<00:00, 71.16it/s]
0.0009252444 6.2827835
--- Evaluation results for gated_convfno ---

Mean initial dice: 0.544
Mean dice after registration: 0.774
Standard deviation of dice values: 0.037
Mean percent of folded pixels: 0.661
Std of percent of folded pixels: 0.335
Mean sdlogJ: 0.467
Std sdlogJ: 0.069
Mean inference time: 0.009 seconds
```

```sh
(fnoreg) fish@CudaAcc:~/fnoreg/deep_fourier_reg$ python evaluate_oasis.py --exp_num 142 --size 80 --ckpt_epoch 60


Total Trainable Params: 22125282
Computing metrics...
  0%|                                                                       | 0/400 [00:00<?, ?it/s]/home/fish/miniconda3/envs/fnoreg/
lib/python3.10/site-packages/torchvision/transforms/functional.py:1603: UserWarning: The default value of the antialias parameter of a
ll the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent acr
oss the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None 
(current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use anti
aliasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(
antialias=True).
  warnings.warn(
/home/fish/miniconda3/envs/fnoreg/lib/python3.10/site-packages/torch/nn/modules/conv.py:456: UserWarning: Applied workaround for CuDNN
 issue, install nvrtc.so (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:80.)
  return F.conv2d(input, weight, bias, self.stride,
/home/fish/miniconda3/envs/fnoreg/lib/python3.10/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming re
lease, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
100%|█████████████████████████████████████████████████████████████| 400/400 [00:04<00:00, 87.21it/s]
0.00010108355 6.282758
--- Evaluation results for gated_convfno ---

Mean initial dice: 0.551
Mean dice after registration: 0.744
Standard deviation of dice values: 0.039
Mean percent of folded pixels: 5.921
Std of percent of folded pixels: 1.559
Mean sdlogJ: 0.705
Std sdlogJ: 0.058
Mean inference time: 0.009 seconds
```

## oasis_v_exp40

- **模型**：FNOReg 3D
- **数据集**：Oasis 3D
- **训练分辨率**：$160 \times 192 \times 224$

修改 [params_3d.json](./deep_fourier_reg/params_3d.json)
```json
  "model_name": "fnoreg",
```

训练  
`python train_oasis3d.py --exp_num 142 --size 160`

验证  
`python evaluate_oasis3d.py --exp_num 142 --size 160`  
`python evaluate_oasis3d.py --exp_num 142 --size 80`

## oasis_v_exp41

- **模型**：FNOReg 3D
- **数据集**：Oasis 3D
- **训练分辨率**：$160 \times 192 \times 224$

修改 [params_3d.json](./deep_fourier_reg/params_3d.json)
```json
  "model_name": "gated_fnoreg",
```

训练  
`python train_oasis3d.py --exp_num 142 --size 160`

验证  
`python evaluate_oasis3d.py --exp_num 142 --size 160`  
`python evaluate_oasis3d.py --exp_num 142 --size 80`
