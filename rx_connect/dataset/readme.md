# NIH Pill Image Processing Scripts

This folder contains three specialized scripts designed for processing and manipulating images from the [NIH Pill Image Recognition Challenge](https://data.lhncbc.nlm.nih.gov/public/Pills/index.html). These scripts facilitate the download, background extraction, and segmentation of the images.

## 1. Scrape NIH Pill Images

The `scrape_nih_pill_images` script automates the download process of consumer images from the NIH Pill Image Recognition Challenge. It supports various file formats including image and video extensions. All the downloaded images, as well as associated XML files, are saved to the specified directory.

The downloaded files will be structured as follows:

```shell
{data_dir}
    ├── AllXML
    │   ├── PillProjectDisc1.xml
    │   ├── PillProjectDisc2.xml
    │   ├── ...
    │   └── PillProjectDisc110.xml
    └── images
        ├── C3PI_Reference
        │   ├── 00000001.jpg
        │   ├── 00000002.jpg
        │   ├── ...
        ├── C3PI_Test
        │   ├── 00000001.jpg
        │   ├── 00000002.jpg
        │   ├── ...
        ├── ...
        └── SPL_SPLIMAGE_V3.0
            ├── 00000001.jpg
            ├── 00000002.jpg
            ├── ...
```

### Usage

You can run the `scrape_nih_images.py` script using the following command:

```shell
python scrape_nih_images.py -d <download-directory> -i <start-index> <end-index>
```

**Notes**:

- Supports both image and video extensions.
- The projects are named as `PillProjectDisc1`, `PillProjectDisc2`, ..., `PillProjectDisc110`, so the start and end indices must be between 1 and 110.

## 2. Extract Background

The extract_background.py script extracts background patches from consumer-grade images. Utilizing the YOLO-NAS model, it detects the bounding boxes of the pills and saves the background patches for further use, such as synthetic image generation.

These extracted background patches can be utilized to generate synthetic images of pills, making them valuable for training machine learning models.

### Important Note

The accuracy of extraction is dependent on the YOLO-NAS model, so manual inspection may be required to remove patches containing pills.

### Usage

You can run the script using the following command:

```shell
python extract_background.py -d <data-directory> -o <output-directory> -mh <minimum-background-height> -mw <minimum-background-width>
```

### Pre-requisites

Make sure to download the `directory_consumer_grade_images.xlsx` file from [this](https://data.lhncbc.nlm.nih.gov/public/Pills/directory_consumer_grade_images.xlsx) link and save it as a CSV file in the specified data directory.

## 3. Segmenting Pills

The segment_pills.py script segments pills within the `MC_C3PI_REFERENCE_SEG_V1.6` layout. Operations include cropping metadata, filling contours, bounding box manipulation, and resizing. Cropped images and masks are saved to the defined output directory.

### Usage

Execute the script using the following command:

```shell
python segment_pills.py -d <DATA_DIR> -o <OUTPUT_DIR> -b <BOTTOM_CROP_PIXELS> -e <EXPAND_PIXELS> -h <HEIGHT> -w <WIDTH> -c <NUM_CPU>
```
