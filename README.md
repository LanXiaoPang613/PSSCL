# PSSCL: A Progressive Sample Selection Framework with Contrastive Loss Designed for Noisy Labels

<h5 align="center">

*Qian Zhang, Yi Zhu, Filipe R. Cordeiro, Qiu Chen*

[[preprint]](https://xxx)
[[License: MIT License]](https://github.com/LanXiaoPang613/PSSCL/blob/main/LICENSE)

</h5>

The PyTorch implementation code of the paper, [PSSCL: A Progressive Sample Selection Framework with Contrastive Loss Designed for Noisy Labels](https://xxx).

**Abstract**


## Installation

```shell
# Please install PyTorch using the official installation instructions (https://pytorch.org/get-started/locally/).
pip install -r requirements.txt
```

## Training

To train on the CIFAR dataset(https://www.cs.toronto.edu/~kriz/cifar.html), run the following command:

```shell
# stage one for CDN noise
python Train_cifar_psscl_stage1_new1.py --r 0.2 --noise_mode 'sym' --lambda_u 0 --data_path './data/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
# stage two for CDN noise
python Train_cifar_psscl_stage2_new1.py --r 0.2 --noise_mode 'sym' --lambda_u 0 --data_path './data/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
```

```shell
# stage one for PMD Type-I IDN noise
python Train_cifar_psscl_stage1_new1_idn.py --noise_mode '1' --lambda_u 30 --data_path './data/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
# stage two for PMD Type-I IDN noise
python Train_cifar_psscl_stage2_new1_idn.py --noise_mode '1' --lambda_u 30 --data_path './data/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
```

```shell
# stage one for RoG RN IDN noise
python Train_cifar_psscl_stage1_new1_idn_rog.py --noise_mode '1' --lambda_u 30 --data_path './data/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
# stage two for RoG RN IDN noise
python Train_cifar_psscl_stage2_new1_idn_rog.py --noise_mode '1' --lambda_u 30 --data_path './data/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
```

To train on the Animal-10N dataset(https://dm.kaist.ac.kr/datasets/animal-10n/), run the following command:

```shell
# stage one for Animal-10N
python Train_animal_psscl_stage1.py --lambda_u 0 --data_path './data/Animal-10N' --dataset 'animal10N' --num_class 10
# stage two for Animal-10N
python Train_animal_psscl_stage2.py --lambda_u 0 --data_path './data/Animal-10N' --dataset 'animal10N' --num_class 10
```

To train on the WebVision dataset(https://data.vision.ee.ethz.ch/cvl/webvision/download.html), run the following command:

```shell
# stage one for WebVision
python Train_webvision_psscl_stage1.py --lambda_u 0 --data_path './data/WebVision1.0' --dataset 'WebVision'
# stage two for WebVision
python Train_webvision_psscl_stage2.py --lambda_u 0 --data_path './data/WebVision1.0' --dataset 'WebVision'
```

## Citation

If you have any questions, do not hesitate to contact zhangqian@jsou.edu.cn

Also, if you find our work useful please consider citing our work:

```bibtex
Zhang, Qian and Zhu, Yi and Cordeiro, Filipe and Chen, Qiu, 
Psscl: A Progressive Sample Selection Framework with Contrastive Loss Designed for Noisy Labels. 
Available at SSRN: https://ssrn.com/abstract=4782767 or http://dx.doi.org/10.2139/ssrn.4782767
```

## Acknowledgement

* [DivideMix](https://github.com/LiJunnan1992/DivideMix): The algorithm that our framework is based on.
* [UNICON](https://github.com/nazmul-karim170/UNICON-Noisy-Label): Inspiration for the webvision dataset code.
* [LongReMix](https://github.com/filipe-research/LongReMix): Inspiration for our framework.
