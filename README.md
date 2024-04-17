# AstroYOLO: A CNN and Transformer Hybrid Deep Learning Object Detection Model for Blue Horizontal-branch Stars

![GitHub](https://img.shields.io/github/license/dzxrly/AstroYOLO) [![doi - 10.1093/pasj/psad071](https://img.shields.io/badge/doi-10.1093%2Fpasj%2Fpsad071-8be9fd)](https://doi.org/10.1093/pasj/psad071)

![Network Structure](./img/network_structure.png)

![RPCurve&PredictionVis](./img/img_2.png)

## Environment

> - Ubuntu Server 22.04 LTS
> - Python 3.10.8
> - CUDA 11.7
> - CUDNN 8.5

Create a new conda environment and install the required packages:

```shell
conda create -n astro_yolo python=3.10
conda activate astro_yolo
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip3 install astropy reproject opencv-python matplotlib scipy scikit-learn tqdm tensorboard tensorboardX torchinfo
```

Before training, check the `config/model_config.py` file to set your training configuration.

## Data Preparation

The dataset file structure is following the VOC2007 dataset format, the dataset directory should be like this: (
e.g. `dataset_example/dataset`)

```
├── dataset
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   │   ├── Annotations
│   │   │   │   ├── annotation_1.xml
│   │   │   │   ├── annotation_2.xml
│   │   │   │   ├── ...
│   │   │   ├── ImageSets
│   │   │   │   ├── Main
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── valid.txt
│   │   │   │   │   ├── test.txt
│   │   │   ├── JPEGImages
│   │   │   │   ├── dataset_image_1.npy
│   │   │   │   ├── dataset_image_2.npy
│   │   │   │   ├── ...
│   ├── train_annotation.txt
│   ├── valid_annotation.txt
│   ├── test_annotation.txt
```

Input images should be in the `.npy` format, including 3 channels. (e.g. `i, r, g`)

## Citation

```
@article{10.1093/pasj/psad071,
    author = {He, Yuchen and Wu, Jingjing and Wang, Wenyu and Jiang, Bin and Zhang, Yanxia},
    title = "{AstroYOLO: A hybrid CNN–Transformer deep-learning object-detection model for blue horizontal-branch stars}",
    journal = {Publications of the Astronomical Society of Japan},
    volume = {75},
    number = {6},
    pages = {1311-1323},
    year = {2023},
    month = {10},
    abstract = "{Blue horizontal-branch stars (BHBs) are ideal tracers for studying the Milky Way (MW) due to their bright and nearly constant magnitude. However, an incomplete screen of BHBs from a survey would result in bias of estimation of the structure or mass of the MW. With surveys of large sky telescopes like the Sloan Digital Sky Survey (SDSS), it is possible to obtain a complete sample. Thus, detecting BHBs from massive photometric images quickly and effectually is necessary. The current acquisition methods of BHBs are mainly based on manual or semi-automatic modes. Therefore, novel approaches are required to replace manual or traditional machine-learning detection. The mainstream deep-learning-based object-detection methods are often vanilla convolutional neural networks whose ability to extract global features is limited by the receptive field of the convolution operator. Recently, a new Transformer-based method has benefited from the global receptive field advantage brought by the self-attention mechanism, exceeded the vanilla convolution model in many tasks, and achieved excellent results. Therefore, this paper proposes a hybrid convolution and Transformer model called AstroYOLO to take advantage of the convolution in local feature representation and Transformer’s easier discovery of long-distance feature dependences. We conduct a comparative experiment on the 4799 SDSS DR16 photometric image dataset. The experimental results show that our model achieves 99.25\\% AP@50, 93.79\\% AP@75, and 64.45\\% AP@95 on the test dataset, outperforming the YOLOv3 and YOLOv4 object-detection models. In addition, we test on larger cutout images based on the same resolution. Our model can reach 99.02\\% AP@50, 92.00\\% AP@75, and 61.96\\% AP@95 respectively, still better than YOLOv3 and YOLOv4. These results also suggest that an appropriate size for cutout images is necessary for the performance and computation of object detection. Compared with the previous models, our model has achieved satisfactory object-detection results and can effectively improve the accuracy of BHB detection.}",
    issn = {2053-051X},
    doi = {10.1093/pasj/psad071},
    url = {https://doi.org/10.1093/pasj/psad071},
    eprint = {https://academic.oup.com/pasj/article-pdf/75/6/1311/54151743/psad071.pdf},
}
```
