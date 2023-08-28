from pathlib import Path
from typing import Union

import click
import determined as det
import yaml

from rx_connect import PROJECT_DIR
from rx_connect.segmentation.semantic.model_def import SegTrial

r"""This script is mainly used for local testing of the model using the PyTorchTrial API.

NOTE: Try not to run this script from the root directory of the project. Otherwise, it will upload all the
data from the project inside the checkpoint directory. Instead, run this script from the directory where
this script is located.

NOTE: The checkpoints are currently saved in /Users/<USERNAME>/Library/Application Support/determined/. As of
now, there is no way to change this path from the code. One workaround is to create a symlink to the desired
location. For example, if you want to save the checkpoints in the project directory, you can run the following
command from the project directory:
    ln -s /Users/<USERNAME>/Library/Application\ Support/determined/ checkpoints
"""


@click.command()
@click.option(
    "-c",
    "--config",
    default=PROJECT_DIR / "segmentation/semantic/configs/local_const.yaml",
    help="Path to the experiment config file.",
)
@click.option(
    "-e",
    "--epochs",
    default=20,
    help="Number of epochs to train the model.",
)
def main(config: Union[str, Path], epochs: int):
    with open(str(config), "r") as file:
        experiment_config = yaml.load(file, Loader=yaml.FullLoader)

    with det.pytorch.init(  # type: ignore[attr-defined]
        hparams=experiment_config["hyperparameters"], exp_conf=experiment_config
    ) as train_context:
        trial = SegTrial(train_context)
        trainer = det.pytorch.Trainer(trial, train_context)  # type: ignore[attr-defined]
        trainer.fit(
            max_length=det.pytorch.Epoch(epochs),  # type: ignore[attr-defined]
            validation_period=det.pytorch.Epoch(1),  # type: ignore[attr-defined]
        )


if __name__ == "__main__":
    main()
