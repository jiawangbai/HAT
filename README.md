# HAT
https://paperswithcode.com/sota/image-classification-on-imagenet?tag_filter=171

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-vision-transformers-by-revisiting/domain-generalization-on-stylized-imagenet)](https://paperswithcode.com/sota/domain-generalization-on-stylized-imagenet?p=improving-vision-transformers-by-revisiting)

Implementation of HAT https://arxiv.org/pdf/2204.00993
```shell
@article{bai2022improving,
  title={Improving Vision Transformers by Revisiting High-frequency Components},
  author={Bai, Jiawang and Yuan, Li and Xia, Shu-Tao and Yan, Shuicheng and Li, Zhifeng and Liu, Wei},
  journal={arXiv preprint arXiv:2204.00993},
  year={2022}
}
```

## Data Preparation
We use the ImageNet-1K training and validation datasets by default.
Please save them in [your_imagenet_path].


## Requirements
torch>=1.7.0  
torchvision>=0.8.0  
timm==0.4.5  
tlt==0.1.0  
pyyaml  
apex-amp  

## Training
Training ViT models with HAT using the default settings in our paper on 8 GPUs:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 \
--data_dir [your_imagenet_path] \
--model [your_vit_model_name] \
--adv-epochs 200 \
--adv-iters 3 \
--adv-eps 0.00784314 \
--adv-kl-weight 0.01 \
--adv-ce-weight 3.0 \
--output [your_output_path] \
and_other_parameters_specified_for_your_vit_models...
```

For instance, we train Swin-T with the following command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 \
--data_dir [your_imagenet_path] \
--model swin_tiny_patch4_window7_224 \
--adv-epochs 200 \
--adv-iters 3 \
--adv-eps 0.00784314 \
--adv-kl-weight 0.01 \
--adv-ce-weight 3.0 \
--output [your_output_path] \
--batch-size 256 \
--drop-path 0.2 \
--lr 1e-3 \
--weight-decay 0.05 \
--clip-grad 1.0
```
For training variants of ViT, Swin Transformer, VOLO, we use the hyper-parameters in [3], [4], and [2], respectively.

## Validation

After training, we can use validate.py to evaluate the ViT model trained with HAT.

For instance, we evaluate Swin-T with the following command:
```shell
python3 -u validate.py \
--data_dir [your_imagenet_path] \
--model swin_tiny_patch4_window7_224 \
--checkpoint [your_checkpoint_path] \
--batch-size 128 \
--num-gpu 8 \
--apex-amp \
--results-file [your_results_file_path]
```

[1] Wightman, R. Pytorch image models. https://github.com/rwightman/pytorch-image-models , 2019.  
[2] Yuan, L. et al. Volo: Vision outlooker for visual recognition. arXiv, 2021.  
[3] Dosovitskiy, A. et al. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2020.  
[4] Liu, Z. et al. Swin transformer: Hierarchical vision transformer using shifted windows. ICCV, 2021.
