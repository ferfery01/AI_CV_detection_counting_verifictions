import random
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
import pandas as pd

from rx_connect.pipelines.detection import RxDetection
from rx_connect.pipelines.image import RxVision
from rx_connect.pipelines.segment import RxSegmentation
from rx_connect.pipelines.vectorizer import RxVectorizerSift
from rx_connect.tools.logging import setup_logger

logger = setup_logger()

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

Method #2 A & B------------------------------:

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
    epill_path: str, epill_ref_list: List[str], images_dir: str, row: str, nfalse_ref: int
) -> Tuple[str, List[str], str]:
    # reading the file and its reference from the csv file
    reference_pills_true, image_name = Path(row[1]).name, row[0]

    # generate a false reference pill for the pill tray

    refname_false = []
    for _ in range(nfalse_ref + 1):
        reference_pills_false = reference_pills_true
        while reference_pills_false == reference_pills_true:
            reference_pills_false = random.choice(epill_ref_list)
        refname_false.append(f"{epill_path}/{reference_pills_false}")

    refname_true = f"{epill_path}/{reference_pills_true}"
    im_path = f"{images_dir}/{image_name}"

    return (refname_true, refname_false, im_path)


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
    default=0.8,
    show_default=True,
    type=float,
    help="similarity confidence score to binarizing the probability",
)
@click.option(
    "-n",
    "--nfalse_ref",
    default=10,
    show_default=True,
    type=int,
    help="number of false pill references to compare each pill tray image against",
)
def main(epill_path: str, data_dir: str, threshold: float, nfalse_ref: int) -> tuple:
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

    epill_ref_list = [
        f.name for f in list(Path(epill_path).glob("*.png")) + list(Path(epill_path).glob("*.jpg"))
    ]
    df = pd.read_csv(csv_dir, header=None, delimiter=":")
    # counting the number of positive similarity scores and number of negative similarity scores

    n_pos, n_neg = 0, 0
    prob_diff_pos_dict, prob_diff_neg_dict = {}, {}
    prob_diff_pos_sum, prob_diff_neg_sum = 0, 0

    pos, true_pos, false_pos = 0, 0, 0

    for _, row in df.iterrows():
        refname_true, refname_false, im_path = load_pill_images_and_references(
            epill_path, epill_ref_list, images_dir, row, nfalse_ref
        )
        imageObj.load_image(im_path)  # load the image of the pill tray
        imageObj.load_ref_image(refname_true)  # load true ref_images
        pos_similarity_scores = imageObj.similarity_scores

        """
        please note that because we already know that we are comparing
        the similarity score against the "true pill reference", we know that we do not have
        any FP or TN. That's why we only count the TP
        """
        pos += len(pos_similarity_scores)
        true_pos += sum(pos_similarity_scores > threshold)
        n_pos += 1

        logger.assertion(
            np.all(pos_similarity_scores <= 1.0), "Some values of similarity scores is out of range."
        )
        # prob_diff_pos is in fact np.mean([1.0 - x for x in pos_similarity_scores])
        prob_diff_pos = 1.00 - np.mean(pos_similarity_scores)
        prob_diff_pos_sum += prob_diff_pos

        prob_diff_pos_dict[Path(im_path).name] = (prob_diff_pos, len(pos_similarity_scores))

        # for each pill tray image, we compare it against multiple randomly chosen
        # false pill references to minimize the bias

        for i in range(nfalse_ref + 1):
            imageObj.load_image(im_path)  # load the image of the pill tray
            imageObj.load_ref_image(refname_false[i])  # load the false ref_images
            neg_similarity_scores = imageObj.similarity_scores
            """
            please note that because we already know that we are comparing
            the similarity score against the "false pill reference", we know that we do not have
            any TP. That's why we only count the FP. We do have TN as well
            but it's not used in Recall/Precision, so we skipped that as well.
            """
            false_pos += sum(neg_similarity_scores > threshold)
            n_neg += 1

            # x in fact is abs(0.0 - x) -> prob_diff_neg = np.mean([x for x in neg_similarity_scores])
            prob_diff_neg = np.mean(neg_similarity_scores)
            prob_diff_neg_sum += prob_diff_neg
            prob_diff_neg_dict[Path(im_path).name] = (prob_diff_neg, len(neg_similarity_scores))

    Precision = true_pos / (true_pos + false_pos)
    Recall = true_pos / pos
    F1_score = Precision * Recall / (Precision + Recall)

    # Method 2A: We average the prob differences over all pill trays

    avg_prob_diff_pos = prob_diff_pos_sum / n_pos
    avg_prob_diff_neg = prob_diff_neg_sum / n_neg
    prob_diff_all = (prob_diff_pos_sum + prob_diff_neg_sum) / (n_pos + n_neg)

    # METHOD#2B for calculating the average probability differences

    sum_p, sum_pcount = 0, 0
    for mean_p, count_pills in prob_diff_pos_dict.values():
        sum_p += mean_p * count_pills
        sum_pcount += count_pills
    prob_diff_positive = sum_p / sum_pcount

    sum_n, sum_ncount = 0, 0
    for mean_n, count_pills in prob_diff_neg_dict.values():
        sum_n += mean_n * count_pills
        sum_ncount += count_pills
    prob_diff_negative = sum_n / sum_ncount

    print("binary score results :")
    print(f"Precision = {Precision:.3f}")
    print(f"Recall = {Recall:.3f}")
    print(f"F1-score = {F1_score:.3f}")

    print("\n")
    print("Probability error (method # 2A)")
    print(f"avg_prob_diff_pos = {avg_prob_diff_pos:.3f}")
    print(f"avg_prob_diff_neg = {avg_prob_diff_neg:.3f}")
    print(f"avg_prob_diff_all = {prob_diff_all:.3f}")

    print("\n")
    print("Probability error (method # 2B)")
    print(f"prob_diff_positive = {prob_diff_positive:.3f}")
    print(f"prob_diff_negative = {prob_diff_negative:.3f}")

    return (
        Precision,
        Recall,
        F1_score,
        avg_prob_diff_pos,
        avg_prob_diff_neg,
        prob_diff_all,
        prob_diff_positive,
        prob_diff_negative,
    )


if __name__ == "__main__":
    (
        Precision,
        Recall,
        F1_score,
        avg_prob_diff_pos,
        avg_prob_diff_neg,
        prob_diff_all,
        prob_diff_positive,
        prob_diff_negative,
    ) = main()
