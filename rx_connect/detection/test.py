from typing import List, Optional

import click
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_val,
)
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)

from rx_connect import CKPT_DIR
from rx_connect.tools.logging import setup_logger
from rx_connect.tools.pretty_print import print_dict

logger = setup_logger()


@click.command()
@click.option(
    "-m",
    "--yolo-model",
    type=click.Choice(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]),
    help="""YOLO model variant to use. If not provided, the model name will be inferred
    directly from the experiment name.""",
)
@click.option(
    "-d",
    "--data-dir",
    default="/media/RxConnectShared/synthetic",
    show_default=True,
    help="Path to the synthetic dataset.",
)
@click.option(
    "-c",
    "--ckpt-path",
    help="""Path to the model checkpoint to test. If not provided, the best model under the
    experiment name will be used.""",
)
@click.option("-e", "--experiment_name", help="Name of the experiment.")
@click.option(
    "-i",
    "--input-dim",
    nargs=2,
    type=int,
    default=[640, 640],
    show_default=True,
    help="Input dimension of the model.",
)
@click.option(
    "--classes",
    multiple=True,
    type=str,
    default=["Pill"],
    show_default=True,
    help="Classes to train on.",
)
@click.option("-b", "--batch-size", default=48, show_default=True, help="Batch size.")
@click.option("-nw", "--num-workers", default=8, show_default=True, help="Number of workers.")
def main(
    yolo_model: Optional[str],
    data_dir: str,
    ckpt_path: Optional[str],
    experiment_name: Optional[str],
    input_dim: List[int],
    classes: List[str],
    batch_size: int,
    num_workers: int,
) -> None:
    # Check that either experiment name or ckpt path is provided
    logger.assertion(
        experiment_name is not None or ckpt_path is not None,
        "Either experiment name or ckpt path must be provided.",
    )

    # Infer the model name from the experiment name
    if yolo_model is None and experiment_name is not None:
        yolo_model = experiment_name.rsplit(":", 1)[1]
    logger.assertion(yolo_model is not None, "YOLO model must be provided when ckpt path is provided.")

    if ckpt_path is not None and experiment_name is not None:
        logger.warning("Defaulting to model provided by the checkpoint.")

    # Load dataloaders
    test_data = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": data_dir,
            "images_dir": "test/images",
            "labels_dir": "test/labels",
            "classes": classes,
            "input_dim": input_dim,
        },
        dataloader_params={"batch_size": batch_size, "num_workers": num_workers},
    )

    # Initialize training parameters
    n_classes: int = len(classes)

    # Use the trainer to train the model
    experiment_name = yolo_model if experiment_name is None else experiment_name
    trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=f"{CKPT_DIR}/detection")

    # Load the model from the checkpoint
    ckpt_path = (
        ckpt_path if ckpt_path is not None else f"{trainer.checkpoints_dir_path}/{trainer.ckpt_best_name}"
    )
    best_model = models.get(
        yolo_model,
        num_classes=n_classes,
        checkpoint_path=ckpt_path,
    )

    # Test the model using the best model
    eval_res = trainer.test(
        model=best_model,
        test_loader=test_data,
        test_metrics_list=DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=n_classes,
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7
            ),
        ),
    )
    print_dict(eval_res, key_col="Metric", value_col="Value")


if __name__ == "__main__":
    main()
