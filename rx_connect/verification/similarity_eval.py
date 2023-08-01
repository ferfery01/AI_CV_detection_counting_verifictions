import random
from pathlib import Path

import click
import numpy as np
import pandas as pd
from skimage.io import imread

from rx_connect.pipelines.detection import RxDetection
from rx_connect.pipelines.image import RxVision
from rx_connect.pipelines.segment import RxSegmentation
from rx_connect.pipelines.vectorizer import RxVectorizerSift

"""_summary_

Method # 1------------------------------::
Here, we calculate the recall / precision / and F1-score for the pill similarity model

Step #1: We first convert the pill similarity probability to a 0-1 binary score by defining a
threshold between 0 and 1 (this threshold needs to be optimized based on ROC-AUC in the next steps.

Step #2: For each pill tray images, we pull its true reference pill and a random false reference pill.
It is with the assumptions that there is only one pill type in each pill tray, which
intuitively makes sense.

Step #3: For each image, we compare it against a) its true reference pill and 2) its false reference pill.
We assign y_pred list for each pill tray image as a
list of 1 (ones), and 0 (zeros) for true & false reference pill respectively

Step #4: calculate the binary classification metrics for evaluation report (recall / precision / and F1-score)

return: the recall/precision / and F1-score

Method #2------------------------------:

This analysis aims to calculate the average probability differences
between the model's predicted pill similarity and the corresponding labels (0 or 1).

Step 1: For each pill tray image, we select its true reference pill and a
random false reference pill. The assumption here is that each pill tray
contains only one type of pill, which is reasonable.

Step 2: For each image, we compare it against two references: a) the
true reference pill and b) the false reference pill. We create a list
called y_pred"for each pill tray image, where we assign 1 for the
true reference pill and 0 for the false reference pill.

Step 3: For each pill tray, we calculate the average absolute value
of the probability difference between the model's predictions and
the labels (1 or 0) for the true and false reference pills, respectively.

Step 4: Calculate the average probability differences across all the pill trays.

Returns:
The final result is the calculated average probability differences.

How to test:
    # python eval_binary_score.py
    # --epill_path /Users/ztakbi6y/ai-lab-RxConnect/.cache/images
    # --data_dir /Users/ztakbi6y/ai-lab-RxConnect/rx_connect/generator/scripts/data/synthetic/detection
    # --threshold 0.8

"""


def load_pill_images_and_references(
    epill_path: str, epill_ref_list: list, images_dir: str, row: str
) -> tuple:
    # reading the file and its reference from the csv file
    reference_pills_true, image_name = Path(row[1]).name, row[0]

    # generate a false reference pill for the pill tray
    reference_pills_false = reference_pills_true
    while reference_pills_false == reference_pills_true:
        reference_pills_false = random.choice(epill_ref_list)

    refname_true = f"{epill_path}/{reference_pills_true}"
    refname_false = f"{epill_path}/{reference_pills_false}"
    image_name = f"{images_dir}/{image_name}"

    reference_true_image = imread(refname_true)
    reference_false_image = imread(refname_false)
    pill_image = imread(image_name)

    return (reference_true_image, reference_false_image, pill_image)


@click.command()
@click.option(
    "-e",
    "--epill_path",
    type=str,
    help="path to the ePill registry to find the reference pill",
)
@click.option(
    "-d",
    "--data_dir",
    type=str,
    help="path to the data directory",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    help="similarity confidence score to binarizing the probability",
)
def main(epill_path: str, data_dir: str, threshold: float) -> None:
    images_dir = f"{data_dir}/images"
    csv_dir = f"{data_dir}/pill_info.csv"

    # Call image generator
    imageObj = RxVision()  # class that includes all functions

    # Call pill counter
    counterObj = RxDetection()

    # Call segmentation object
    segmentObj = RxSegmentation()

    # Call vectorizer object
    vectorizerObj = RxVectorizerSift()

    imageObj.set_counter(counterObj)
    imageObj.set_vectorizer(vectorizerObj)
    imageObj.set_segmenter(segmentObj)

    pos, true_pos, true_neg = 0, 0, 0

    epill_ref_list = [
        f.name for f in list(Path(epill_path).glob("*.png")) + list(Path(epill_path).glob("*.jpg"))
    ]
    df = pd.read_csv(csv_dir, header=None, delimiter=":")

    for _, row in df.iterrows():
        img_true_ref, img_false_ref, image = load_pill_images_and_references(
            epill_path, epill_ref_list, images_dir, row
        )
        imageObj.load_image(image)  # load the image of the pill tray
        imageObj.load_ref_image(img_true_ref)  # load true ref_images

        pos_similarity_scores = imageObj.similarity_scores
        pos += len(pos_similarity_scores)
        true_pos += sum(pos_similarity_scores > threshold)
        prob_diff_pos = np.mean([abs(1.0 - x) for x in pos_similarity_scores])

        imageObj.load_image(image)  # load the image of the pill tray
        imageObj.load_ref_image(img_false_ref)  # load the false ref_images

        neg_similarity_scores = imageObj.similarity_scores
        true_neg += sum(neg_similarity_scores < threshold)
        prob_diff_neg = np.mean([abs(0.0 - x) for x in neg_similarity_scores])

    Precision = true_pos / (true_pos + true_neg)
    Recall = true_pos / pos

    print(f"Precision = {Precision:.3f}")
    print(f"Recall = {Recall:.3f}")
    print(f"F1-score = {Precision*Recall/(Precision+Recall):.3f}")

    # We average the prob differences over all pill trays
    # we use length (n) as it indicates the number of images we iterated over

    n = df.shape[0]
    avg_prob_diff_pos = round(prob_diff_pos / n, 2)
    avg_prob_diff_neg = round(prob_diff_neg / n, 2)
    prob_diff_all = round((prob_diff_pos + prob_diff_neg) / (2 * n), 2)

    print(f"avg_prob_diff_pos = {avg_prob_diff_pos:.3f}")
    print(f"avg_prob_diff_neg = {avg_prob_diff_neg:.3f}")
    print(f"avg_prob_diff_all = {prob_diff_all:.3f}")


if __name__ == "__main__":
    main()
