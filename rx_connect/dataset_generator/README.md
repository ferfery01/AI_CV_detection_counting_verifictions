# Dataset Generator

Dataset Generator is a Python-based tool to generate synthetic datasets for counting and detecting pills. This repository creates synthetic images by adding pills to background images (e.g., a plate) to help develop and test machine learning models for pill counting and pill segmentation tasks.


## Usage
### Generate Masks

To use Dataset Generator, you will first need to generate masks for all the pill images. This can be done by running the following command:
```
mask_generator
```
There are two distinct methodologies for creating these masks: using a Deep Neural Network (DNN) or employing traditional Computer Vision (CV) algorithms. To use the DNN, the `--use-sam` flag needs to be passed to the `mask_generator`. This invokes the SAM-HQ mask to generate segmentation masks of the pills. Although using DNN yields higher quality masks, it requires more computational time. Hence, it is advisable to use a GPU server for this process. The mask for each pill image will be saved in the `./data/masks` directory.

### Generate Synthetic Dataset

One can create a synthetic dataset by running the following command:
```
dataset_generator
```
The generated dataset will be stored in the `./data/synthetic` directory, with the following structure dependeing on the mode (detection or segmentation):
```
data/
    └── synthetic/
        └── detection (or segmentation)/
            ├── images/
            ├── labels (or comp_masks)/
            └── pill_info.csv
```

The `images` folder holds the synthetic images of pills composed on various backgrounds. This `image` folder can be a local directory or a remote directory. By default, it uses the images and masks stored on the remote GPU server at `RxConnectShared/ePillID/pills/`. The number of pills in each image is embedded in the filename. For instance, a file named `0_18.jpg` signifies that there are `18` pills in the image.

For the `detection` mode, a `labels` folder is created. This folder contains bounding box annotations for each generated image in the YOLO format. These annotations serve as valuable assets for training and evaluating object detection models, especially designed for counting and detecting pills. Additionally, the `pill_info.csv` file is generated which contains the mapping of composed image file name to reference pill image path. This can be used during the downstream task to get the reference pill used in the composed image.

In the `segmentation` mode, a `comp_mask` folder is generated. This folder contains the instance segmentation mask for each synthesized image, which can be utilized for training and evaluating instance segmentation models, particularly aimed at segmenting pills from the background.
