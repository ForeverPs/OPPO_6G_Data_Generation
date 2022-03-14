# OPPO 6G Data Generation with an E2E Framework

## Datasets
- H1_32T4R.mat
- H2_32T4R.mat
- please put the original data in `data` folder.

## Data Augmentation Scheme
- Complex number is special : `a+bj` has a same square similarity with `-a-bj`, `b-aj` and `-b+aj`
- With the strategy above, you can quadruple the amount of training data compared to the raw.
- We randomly scale the data with a factor between 0.8~1.2, random gaussian noise with `mean` equals to 0 and `std` equals to 1e-4 are adopted.

## Architectures
- Auto encoder with reconstruction loss.
- ResNet18 as an Encoder.
- 3D Conv as a Decoder.
- Position Attention Module and Channel Attention Module are important.
- Normalization such as `BatchNorm2d` after Decoder is important.
- Latent Quantization.

## Pretrained Models
We provide several pretrained models in the folder of `saved_models`.
- sim : similarity score tested on the raw data.
- multi : multi score tested on the raw data.
- score : tested on the local raw data.
- feel free to use the pretrained weights or training from scratch.

## Training
- modify the `data_type` in `train.py`, maybe you have to choose a suitable GPU id.
- Online validation, only save the models with best scores so far.
- Benchmark : data1: local score approx 0.82~0.83
- Benchmark : data2: local score approx 0.76~0.77
- Hints : smaller batch size may result in higher similarity score and higher multi score.
- epochs : we perform no ablation study on this parameter, you can just let it run.

## Boost Scheme
- we use adaboost weights to ensemble several models for acquiring performance gain. 
- Without model ensembles, you can still achieve a online score up to 0.72 easily.

## Submit_pt
- you can just use the single model without ensembles which is much easier.
- Without ensembles, it is still trivial to achieve a score up to 0.72

## Reference
- [Deep Residual Learning for image recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR, 2016)
- [Dual Attention Network for Scene Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.pdf) (CVPR, 2019)
- [Deep Learning-based Implicit CSI Feedback in Massive MIMO](https://arxiv.org/pdf/2105.10100.pdf) (IEEE Transactions on Communications, 2021)

