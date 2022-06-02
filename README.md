# HAT
Implementation of HAT https://arxiv.org/pdf/2204.00993


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

[1] Wightman, R.: Pytorch image models. https://github.com/rwightman/pytorch-image-models (2019). https://doi.org/10.5281/zenodo.4414861

[2] Yuan, L., Hou, Q., Jiang, Z., Feng, J., Yan, S.: Volo: Vision outlooker for visual recognition. arXiv preprint arXiv:2106.13112 (2021)

[3] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR (2020)

[4] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: ICCV (2021)
