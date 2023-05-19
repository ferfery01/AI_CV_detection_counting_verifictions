# Dataset Generator

Dataset Generator is a Python-based tool to generate synthetic datasets for counting and detecting pills. This repository creates synthetic images by adding pills to background images (e.g., a plate) to help develop and test machine learning models for pill counting and detection tasks.


## Usage
### Generate Masks

To use Dataset Generator, you will first need to generate masks for the pill images in the `./data/pills/images` directory. This can be done by running the following command:
```
mask_generator
```
This will create mask images for each pill image and save them in the `./data/pills/masks` directory.

### Generate Synthetic Dataset

After obtaining all the masks, you can create a synthetic dataset by running the following command:
```
dataset_generator
```
The generated dataset will be stored in the `./dataset/synthetic` directory, with the following structure:
```
dataset/
    └── synthetic/
        ├── images/
        └── labels/
```

The `images` folder contains the synthetic images of pills composed on the background. The number of pills in each image is indicated in the filename. For example, a file named `0_18.jpg` implies that there are `18` pills in the image.

The `labels` folder contains bounding box annotations for each generated image, stored in the YOLO format. These annotations can be used for training and evaluating object detection models, specifically tailored for counting and detecting pills in images.
