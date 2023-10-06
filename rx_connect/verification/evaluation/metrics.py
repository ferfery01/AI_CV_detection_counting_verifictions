from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


class BinaryClassificationEvaluator:
    threshold_label = "Optimal Threshold"
    """Base class for evaluating the optimal threshold and ROC curve for a vectorized model.

    Args:
        Target: A list containing the true values of similarity scores.
        predicted: A list containing predicted similarity scores.
        model: A string representing the name or type of the model.
        plot_path: The path where you want to save the ROC curve for the selected model.
        pos_predicted: A list containing only predicted similarity scores compared against the true references.
        neg_predicted: A list containing only predicted similarity scores compared against the false references.

        Return:
            Tuple(Opt_threshold, Precision, Recall, F1_score)
    """

    def __init__(
        self,
        target: List[float],
        predicted: List[float],
        pos_predicted: List[float],
        neg_predicted: List[float],
        model: str,
        plot_path: Path,
    ) -> None:
        self.plot_path = plot_path
        self.predicted = predicted
        self.target = target

        self.pos_predicted = pos_predicted
        self.neg_predicted = neg_predicted
        self.model = model

    def _youden(self) -> None:
        """
        Youden's J statistic: https://en.wikipedia.org/wiki/Youden%27s_J_statistic
        Find the optimal probability cutoff point for a classification model based on the Youden's J statistic.
        """

        # Calculate ROC curve
        self.fpr, self.tpr, threshold = roc_curve(self.target, self.predicted)

        # get the best threshold: where J is maximum and J is defined as follow
        #  or J = Sensitivity + Specificity – 1
        #  or J = TPR + (1 – FPR) – 1
        #  or J = sqrt(tpr*(1-fpr))

        J = self.tpr - self.fpr
        self.ix = np.argmax(J)
        self.opt_threshold = threshold[self.ix]

    def F_score_metrics(self, t=None, beta=1.0) -> Tuple[float, float, float, Union[float, None]]:
        # t is the threshold
        # F1-score is defined where β (beta) is 1 as usual.
        # A β value less than 1 favors the percision metric,
        # while values greater than 1 favor the recall metric.
        # beta = 0.5 and =2 are the most commonly used measures other than F1 scores.

        if t is None:
            t = self.opt_threshold

        self._binary(t)
        n_pos = len(self.pos_predicted)
        Precision = self.true_pos / (self.true_pos + self.false_pos)
        Recall = self.true_pos / n_pos

        # The Fβ_score score is useful when we want to prioritize
        # one measure while preserving results from the other measure.
        F_score = (1 + beta**2) * (Precision * Recall) / ((beta**2 * Precision) + Recall)
        return Precision, Recall, F_score, t

    def _binary(self, t) -> None:
        # apply threshold (t) to probabilities to create binary (0 and 1) labels
        self.true_pos = sum(self.pos_predicted > t)
        self.false_pos = sum(self.neg_predicted > t)

    def binary_metrics(self) -> Tuple[float, float, float, Union[float, None]]:
        self._youden()
        return self.F_score_metrics()

    def custome_binary_metrics(self) -> Tuple[float, float, float, float, float]:
        thresholds = np.arange(0, 1, 0.001)

        # F1-score is defined where β is 1. A β value less than 1
        # favors the precision metric, while values greater than 1
        # favor the recall metric. beta = 0.5 and 2 are the most
        # commonly used measures other than F1 scores. Here we use beta = 0.5

        beta = 0.5
        Precision_beta, Recall_beta, F_beta_scores = (
            [0.0] * len(thresholds),
            [0.0] * len(thresholds),
            [0.0] * len(thresholds),
        )

        for i, t in enumerate(thresholds):
            p, r, f, _ = self.F_score_metrics(t, beta)

            Precision_beta[i] = p
            Recall_beta[i] = r
            F_beta_scores[i] = f

        # get optimal threshold
        ix = np.argmax(F_beta_scores)
        return Precision_beta[ix], Recall_beta[ix], F_beta_scores[ix], thresholds[ix], beta

    def prob_metrics(
        self,
        prob_diff_pos_dict: Dict[str, Tuple[float, int]],
        prob_diff_neg_dict: Dict[str, Tuple[float, int]],
    ) -> Tuple[float, float]:
        """
        Calculate error probability summary.

        Args:
            prob_diff_pos_dict (Dict[str, Tuple[float, int]]): A dictionary containing positive error probability values
                with keys representing names and values as tuples containing:
                - mean_p (float): The mean error probability.
                - count_pills (int): The count of pills for the corresponding error probability.
            prob_diff_neg_dict (Dict[str, Tuple[float, int]]): A dictionary containing negative error probability values
                with keys representing names and values as tuples containing:
                - mean_n (float): The mean error probability.
                - count_pills (int): The count of pills for the corresponding error probability.

        Returns:
            Tuple[float, float]: A tuple containing two float values:
                - prob_diff_positive: The overall positive error probability.
                - prob_diff_negative: The overall negative error probability.
        """
        sum_p, sum_pcount = 0.0, 0
        for mean_p, count_pills in prob_diff_pos_dict.values():
            sum_p += mean_p * count_pills
            sum_pcount += count_pills
        prob_diff_positive = sum_p / sum_pcount

        sum_n, sum_ncount = 0.0, 0
        for mean_n, count_pills in prob_diff_neg_dict.values():
            sum_n += mean_n * count_pills
            sum_ncount += count_pills
        prob_diff_negative = sum_n / sum_ncount

        return prob_diff_positive, prob_diff_negative

    def plots(self) -> None:
        self._save_plot()

    def _save_plot(self) -> None:
        self._youden()

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], linestyle="--", label="1:1")
        ax.plot(self.fpr, self.tpr, linewidth=1.5)
        ax.plot(self.fpr[self.ix], self.tpr[self.ix], "bo", ms=15)
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(f"ROC Curve for {self.model} ({self.threshold_label}={self.opt_threshold:.2f})")
        self.plot_path.mkdir(parents=True, exist_ok=True)
        plot_filename = self.plot_path / f"{self.model}.png"
        fig.savefig(plot_filename)
