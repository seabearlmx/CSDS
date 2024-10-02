## CSDS (TITS 2024 Under Review): Official Project Webpage
This repository provides the official PyTorch implementation of the following paper:
> **CSDS:** Class-Balanced Sampling and Discriminative Stylization for Domain Generalization Semantic Segmentation<br>
> TITS 2024, Under Review<br>

> **Abstract:** 
*Existing domain generalization semantic segmentation (DGSS) methods have achieved remarkable performance on unseen domains by generating stylized images to increase the diversity of training data. However, since the training data is usually class-imbalanced, uniform style randomization is unable to generate diverse minority classes. This means that models may overfit to the minority classes, resulting in suboptimal performance on the minority classes. In addition, the image-level style randomization may also corrupt the class-discriminative regions of objects, leading to a loss of the class-discriminative representation. To address these issues, a novel class-balanced sampling and discriminative stylization (CSDS) approach is proposed for DGSS. Specifically, first, a pixel-level class-balanced sampling (PCS) strategy is proposed to adaptively sample patches of the minority classes from the source domain images and paste the sampled patches on the input images. Unlike existing class sampling strategies that fix the minority classes, the PCS strategy dynamically determines the minority classes by estimating the class distribution after each sampling. Then, a class-discriminative style randomization (CSR) strategy is proposed to increase the style diversity of the sampled patches while preserving the class-discriminative regions. Finally, since the pasting positions of the sampled patches are uncertain, which may confuse the semantic relations between the classes, a semantic consistency constraint is proposed to ensure the learning of reliable semantic relations. Extensive experiments demonstrate that the proposed approach achieves superior performance compared to existing DGSS methods on multiple benchmarks. The source code has been released on https://github.com/seabearlmx/CSDS.*<br>

## Pytorch Implementation
### Installation
Clone this repository.
```
git clone https://github.com/seabearlmx/CSDS.git
cd CSDS
```
Install following packages.
```
conda create --name csds python=3.7
conda activate csds
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install scipy==1.1.0
conda install tqdm==4.46.0
conda install scikit-image==0.16.2
pip install tensorboardX
pip install thop
pip install kmeans1d
imageio_download_bin freeimage
```
### How to Run CSDS
We evaludated CSDS on [Cityscapes](https://www.cityscapes-dataset.com/), [BDD-100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/),[Synthia](https://synthia-dataset.net/downloads/) ([SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/download/808/)), [GTAV](https://download.visinf.tu-darmstadt.de/data/from_games/) and [Mapillary Vistas](https://www.mapillary.com/dataset/vistas?pKey=2ix3yvnjy9fwqdzwum3t9g&lat=20&lng=0&z=1.5).

We adopt Class uniform sampling proposed in [this paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.pdf) to handle class imbalance problems. [GTAVUniform](https://github.com/shachoi/RobustNet/blob/0538c69954c030273b3df952f90347572ecac53b/datasets/gtav.py#L306) and [CityscapesUniform](https://github.com/shachoi/RobustNet/blob/0538c69954c030273b3df952f90347572ecac53b/datasets/cityscapes.py#L324) are the datasets to which Class Uniform Sampling is applied.


1. For Cityscapes dataset, download "leftImg8bit_trainvaltest.zip" and "gtFine_trainvaltest.zip" from https://www.cityscapes-dataset.com/downloads/<br>
Unzip the files and make the directory structures as follows.
```
cityscapes
 └ leftImg8bit_trainvaltest
   └ leftImg8bit
     └ train
     └ val
     └ test
 └ gtFine_trainvaltest
   └ gtFine
     └ train
     └ val
     └ test
```
```
bdd-100k
 └ images
   └ train
   └ val
   └ test
 └ labels
   └ train
   └ val
```
```
mapillary
 └ training
   └ images
   └ labels
 └ validation
   └ images
   └ labels
 └ test
   └ images
   └ labels
```

#### We used [GTAV_Split](https://download.visinf.tu-darmstadt.de/data/from_games/code/read_mapping.zip) to split GTAV dataset into training/validation/test set. Please refer the txt files in [split_data](https://github.com/seabearlmx/CSDS/tree/main/split_data).

```
GTAV
 └ images
   └ train
     └ folder
   └ valid
     └ folder
   └ test   
     └ folder
 └ labels
   └ train
     └ folder
   └ valid
     └ folder
   └ test   
     └ folder
```

#### We randomly splitted [Synthia dataset](http://synthia-dataset.net/download/808/) into train/val set. Please refer the txt files in [split_data](https://github.com/seabearlmx/CSDS/tree/main/split_data).

```
synthia
 └ RGB
   └ train
   └ val
 └ GT
   └ COLOR
     └ train
     └ val
   └ LABELS
     └ train
     └ val
```

2. You should modify the path in **"<path_to_csds>/config.py"** according to your dataset path.
```
#Cityscapes Dir Location
__C.DATASET.CITYSCAPES_DIR = <YOUR_CITYSCAPES_PATH>
#Mapillary Dataset Dir Location
__C.DATASET.MAPILLARY_DIR = <YOUR_MAPILLARY_PATH>
#GTAV Dataset Dir Location
__C.DATASET.GTAV_DIR = <YOUR_GTAV_PATH>
#BDD-100K Dataset Dir Location
__C.DATASET.BDD_DIR = <YOUR_BDD_PATH>
#Synthia Dataset Dir Location
__C.DATASET.SYNTHIA_DIR = <YOUR_SYNTHIA_PATH>
```
3. You should download pre-trained photo_wct.pth and move it on **"<path_to_csds>/pretrain"**.
4. You can train CSDS with following commands.
```
<path_to_csds>$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_r50os16_gtav_base.sh # Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet50, DeepLabV3+
<path_to_csds>$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_r50os16_gtav_ibn.sh # Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet50, IBN-Net
```

## Acknowledgments
Our pytorch implementation is heavily derived from [RobustNet](https://github.com/shachoi/RobustNet).
Thanks to the RobustNet implementations.
