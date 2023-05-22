# Pill Counting Module
The Pill Counting module is a part of Rx-Connect system, responsible for accurately detecting and counting the number of pills within an image. It leverages the power of YOLO-NAS model for efficient and accurate results.

## Data Preparation
The training process requires a properly structured dataset. Use the following template to structure your data directory:
```
data_dir/
    └── train/test/val
        ├── images/
        └── labels/
```
For dataset generation, we are currently using `dataset_generator` which aligns with the aforementioned format.

## Train Model
With your dataset in place, initiate the model training process using the command below:
```
train_pill_counter --model <MODEL_TYPE> --data-dir </path/to/data/directory> --experiment_name <EXPERIMENT_NAME> --classes "Pill"
```
Replace <MODEL_TYPE>, </path/to/data/directory>, and <EXPERIMENT_NAME> with the model type, the path to your dataset, and a unique identifier for the training experiment, respectively.

## Inference

Post-training, leverage the trained model to infer pill counts from a directory of test images:
```
predict_pill_count --model <MODEL_TYPE> --checkpoint-path </path/to/checkpoint.pth> --classes "Pill" \
        --test-dir </path/to/test/dir> --show-conf
```
Replace <MODEL_TYPE>, </path/to/model/checkpoint>, and </path/to/test/directory> with the model type, the path to your saved model checkpoint, and the directory containing your test images, respectively.

The resulting predictions will be saved in pred_dir, with cropped images of individual pills stored in crops_dir.
