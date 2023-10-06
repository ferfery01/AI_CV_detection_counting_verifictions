from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import pandas as pd

from rx_connect import ROOT_DIR, SHARED_PILL_VERIFICATION_DIR, SHARED_RXIMAGE_DATA_DIR
from rx_connect.core.utils.io_utils import get_matching_files_in_dir
from rx_connect.pipelines.detection import RxDetection
from rx_connect.pipelines.image import RxVision
from rx_connect.pipelines.segment import RxSegmentation

# Call vectorizer object
from rx_connect.pipelines.vectorizer import (
    RxVectorizer,
    RxVectorizerCEL,
    RxVectorizerColorhist,
    RxVectorizerColorMomentHash,
    RxVectorizerML,
    RxVectorizerSift,
)
from rx_connect.tools import is_remote_dir
from rx_connect.tools.data_tools import (
    fetch_file_paths_from_remote_dir,
    fetch_from_remote,
)
from rx_connect.tools.logging import setup_logger
from rx_connect.verification.evaluation.eval_loading_utils import (
    load_pill_images_and_references,
)
from rx_connect.verification.evaluation.metrics import BinaryClassificationEvaluator

logger = setup_logger()

# python opt_thres.py

"""
Note: True Positive Rate (TPR) and False Positive Rate (FPR)
This script returns the optimal threshold for binarizing the
similarity. It also returns ROC curve saved in the plot folder
in the verification/evaluation directory.

The optimal threshod is where the difference between
TPR and (1-FPR) are minimum utilizing a heuristic method
"""

vectorizer_registery: Dict[str, RxVectorizer] = {
    "ColorHist": RxVectorizerColorhist(),
    "ColorMomentHash": RxVectorizerColorMomentHash(),
    "ML": RxVectorizerML(),
    "Sift": RxVectorizerSift(),
    "Cel": RxVectorizerCEL(),
}


@click.command()
@click.option(
    "-s",
    "--sample-dir",
    default=SHARED_RXIMAGE_DATA_DIR,
    type=click.Path(path_type=Path),
    help="Sample directory of RxImages to find 'images' folder containing reference pills",
)
@click.option(
    "-d",
    "--data-dir",
    default=SHARED_PILL_VERIFICATION_DIR,
    type=click.Path(path_type=Path),
    help="data dir to generated sample pill images for similarity evaluations: It can be found",
)
@click.option(
    "-n",
    "--nfalse-ref",
    default=1,
    show_default=True,
    type=int,
    help="number of false pill references to compare each pill tray image against",
)
@click.option(
    "-v",
    "--vectorizer-model",
    default="ColorMomentHash",
    show_default=True,
    type=click.Choice(["ColorHist", "ColorMomentHash", "ML", "Sift", "Cel"]),
    help="choose the vectorizer model you'd like to evaluate from the choices provided",
)
@click.option(
    "-p",
    "--plot-path",
    default=ROOT_DIR / "rx_connect/verification/evaluation/plots",
    type=click.Path(path_type=Path),
    help="path to where save the ROC plots",
)
def main(
    sample_dir: Path,
    data_dir: Path,
    nfalse_ref: int,
    vectorizer_model: str,
    plot_path: Path,
    random_seed: int = 0,
) -> None:
    """
    Random seed initializes the randomness for the first
    reference images and increments from there. However, for
    the sake of consistency and the ability to track different
    models' improvements, it's recommended not to change it.
    """

    # Call image generator
    imageObj = RxVision()  # class that includes all functions

    # Call pill counter
    counterObj = RxDetection()

    # Call segmentation object
    segmentObj = RxSegmentation()
    vectorizerObj = vectorizer_registery[vectorizer_model]

    imageObj.set_counter(counterObj)
    imageObj.set_vectorizer(vectorizerObj)
    imageObj.set_segmenter(segmentObj)

    sample_dir = sample_dir / "images"

    if is_remote_dir(sample_dir):
        ref_list = [Path(f.name) for f in fetch_file_paths_from_remote_dir(sample_dir)]
    else:
        ref_list = get_matching_files_in_dir(sample_dir, wildcard_patterns="*.[jp][pn]g")

    csv_path = fetch_from_remote(f"{data_dir}/pill_info.csv")
    df = pd.read_csv(csv_path, header=None, delimiter=":")

    # counting the number of positive similarity scores and number of negative similarity scores

    predicted: List[float] = []
    target: List[float] = []
    pos_predicted: List[float] = []
    neg_predicted: List[float] = []
    prob_diff_pos_dict, prob_diff_neg_dict = {}, {}

    for j, row in df.iterrows():
        if j < 2:
            true_ref, img_name = Path(row[1]).name, row[0]
            refname_true, refname_false, im_path, random_seed = load_pill_images_and_references(
                sample_dir, ref_list, data_dir, true_ref, img_name, nfalse_ref, random_seed
            )

            try:
                imageObj.load_image(im_path)  # load the image of the pill tray
                imageObj.load_ref_image(refname_true)  # load true ref_images
                pos_similarity_scores = imageObj.similarity_scores

                pos_predicted += pos_similarity_scores.tolist()
                predicted += pos_similarity_scores.tolist()
                target += [1.0] * len(pos_similarity_scores.tolist())

                # prob_diff_pos is in fact np.mean([1.0 - x for x in pos_similarity_scores])

                mean_pos = 1.00 - np.mean(pos_similarity_scores)
                prob_diff_pos_dict[im_path.name] = (mean_pos, pos_similarity_scores.size)

                assert np.all(
                    pos_similarity_scores <= 1.0
                ), "Some values of similarity scores is out of range."

                # for each pill tray image, we compare it against multiple randomly chosen
                # false pill references to minimize the bias

                for i in range(nfalse_ref):
                    imageObj.load_image(im_path)  # load the image of the pill tray
                    imageObj.load_ref_image(refname_false[i])  # load the false ref_images
                    neg_similarity_scores = imageObj.similarity_scores

                    neg_predicted += neg_similarity_scores.tolist()
                    predicted += neg_similarity_scores.tolist()
                    target += [0.00] * len(neg_similarity_scores.tolist())

                    # prob_diff_neg is abs(0.0 - prob_diff_neg)
                    mean_neg = np.mean(neg_similarity_scores, dtype=np.float64)
                    prob_diff_neg_dict[im_path.name] = (mean_neg, neg_similarity_scores.size)

            except Exception:
                logger.warning("error loading image", im_path)

                logger.info(
                    "check the image path printed above."
                    "The error is most likely is due to the inability"
                    "of the moSdel to detect either the bounding"
                    "boxes or shape. This happens mostly for"
                    "very unique shapes of the pills or when"
                    "the noise ratio exists in the background"
                    "is relatively high"
                )

    binary_cls_eval = BinaryClassificationEvaluator(
        target, predicted, pos_predicted, neg_predicted, vectorizer_model, plot_path
    )
    Precision, Recall, F1_score, opt_threshold = binary_cls_eval.binary_metrics()
    binary_cls_eval.plots()

    logger.info("\n")
    logger.info("\u253C" * 40)
    logger.info(
        f"Binary evaluation metrics using Youden's J statistic for vectorizer model = {vectorizer_model} "
    )
    logger.info(f"Optimal threshold  = {opt_threshold:.2f}")
    logger.info(f"Precision  = {Precision: .2f}")
    logger.info(f"Recall  = {Recall: .2f}")
    logger.info(f"F1_score = {F1_score: .2f}")

    binary_cls_eval.plots()
    (
        Precision_beta,
        Recall_beta,
        F1_score_beta,
        opt_threshold_beta,
        beta,
    ) = binary_cls_eval.custome_binary_metrics()

    logger.info("\n")
    logger.info("\u253C" * 40)
    logger.info(
        f"Binary evaluation metrics using customized "
        f"percision-recall curve for vectorizer model = "
        f"{vectorizer_model} and beta = {beta}:"
    )

    logger.info(f"Optimal threshold  = {opt_threshold_beta:.2f}")
    logger.info(f"Precision  = {Precision_beta: .2f}")
    logger.info(f"Recall  = {Recall_beta: .2f}")
    logger.info(f"F1_score = {F1_score_beta: .2f}")

    prob_diff_positive, prob_diff_negative = binary_cls_eval.prob_metrics(
        prob_diff_pos_dict, prob_diff_neg_dict
    )
    logger.info("\u253C" * 40)
    logger.info(f"Probability error for {vectorizer_model} vectorizer model:")
    logger.info(f"Mean probability error for true references = {prob_diff_positive:.2f}")
    logger.info(f"Mean probability error for false references = {prob_diff_negative:.2f}")
    logger.info("\u253C" * 40)


if __name__ == "__main__":
    main()
