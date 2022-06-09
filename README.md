# HAT

Implementation of HAT https://arxiv.org/pdf/2204.00993
```shell
@article{bai2022improving,
  title={Improving Vision Transformers by Revisiting High-frequency Components},
  author={Bai, Jiawang and Yuan, Li and Xia, Shu-Tao and Yan, Shuicheng and Li, Zhifeng and Liu, Wei},
  journal={arXiv preprint arXiv:2204.00993},
  year={2022}
}
```
| Model   | Params | FLOPs | Test Size | Top-1 | +HAT Top-1 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| ViT-T   | 5.7M   | 1.6G  | 224       | 72.2  | **73.3**      |
| ViT-S   | 22.1M  | 4.7G  | 224       | 80.1  |  **80.9**     |
| ViT-B   | 86.6M  | 17.6G | 224       | 82.0  |  **83.2**     |
| Swin-T  | 28.3M  | 4.5G  | 224       | 81.2  |  **82.0**      |
| Swin-S  | 49.6M  | 8.7G  | 224       | 83.0  |  **83.3**      |
| Swin-B  | 87.8M  | 15.4G | 224       | 83.5  |  **84.0**       |
| VOLO-D1 | 26.6M  | 6.8G  | 224       | 84.2  |  **84.5**       |
| VOLO-D1 | 26.6M  | 22.8G | 384       | 85.2  |  **85.5**       |
| VOLO-D5 | 295.5M | 69.0G | 224       | 86.1  |  **86.3**       |
| VOLO-D5 | 295.5M | 304G  | 448       | 87.0  |  **87.2**      |
| VOLO-D5 | 295.5M | 412G  | 512       | 87.1  |  **87.3**      |



## Requirements
torch>=1.7.0  
torchvision>=0.8.0  
timm==0.4.5  
tlt==0.1.0  
pyyaml  
apex-amp  

## ImageNet Classification

### Data Preparation
We use the ImageNet-1K training and validation datasets by default.
Please save them in [your_imagenet_path].


### Training
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

### Validation

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

## Downstream Tasks

We first pretrain Swin-T/S/B on the ImageNet-1k dataset with our proposed HAT, and then transfer the models to the downstream tasks, including object detection, instance segmentation, and semantic segmentation. 

We use the codes in [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) and [Swin Transformer for Semantic Segmentaion](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation), and all their configurations.

| Backbone   | Params | FLOPs | AP_box | +HAT AP_box | AP_mask | +HAT AP_mask |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Swin-T  | 86M  | 745G  | 50.5  |  **50.9**      |43.7| **43.9**      |
| Swin-S  | 107M  | 838G | 51.8  |  **52.5**      |44.7| **45.4**      |
| Swin-B  | 145M  | 982G  | 51.9  |  **52.8**       |45.0| **45.6**      |

[1] Wightman, R. Pytorch image models. https://github.com/rwightman/pytorch-image-models , 2019.  
[2] Yuan, L. et al. Volo: Vision outlooker for visual recognition. arXiv, 2021.  
[3] Dosovitskiy, A. et al. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2020.  
[4] Liu, Z. et al. Swin transformer: Hierarchical vision transformer using shifted windows. ICCV, 2021.
