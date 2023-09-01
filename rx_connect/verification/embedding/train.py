import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from rx_connect import CKPT_DIR
from rx_connect.verification.embedding.base import ResNetEmbeddingModel
from rx_connect.verification.embedding.lightning_model import EmbeddingLightningModel

logger = TensorBoardLogger("lightning_logs", name="RL_vectorizer")

EmbModel = ResNetEmbeddingModel("resnet50")
model = EmbeddingLightningModel(EmbModel)
checkpoint_callback = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename="{epoch}-{epoch_avg_margin:.2f}",
    save_top_k=10,
    monitor="epoch_avg_margin",
    mode="max",
)
trainer = lightning.Trainer(
    accelerator="gpu",
    devices=[1],
    callbacks=[checkpoint_callback],
    logger=logger,
    max_epochs=-1,
    log_every_n_steps=1,
)
trainer.fit(model=model)
