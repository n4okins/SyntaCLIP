# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
from utils.clogging import getColoredLogger
from utils.initialize import initializer

from syntaclip.utils.datasets import cc12m

if __name__ == "__main__":
    logger = getColoredLogger(__name__)
    logger.setLevel("DEBUG")
    PROJECT_ROOT = initializer(globals(), logger=logger)
    logger.info(f"{PROJECT_ROOT=}")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    CC12M_DATA_DIR = Path.home() / "datasets" / "WebDataset" / "CC12M"
    dataset = cc12m.CC12MDataset(CC12M_DATA_DIR)
    dataset.download(enable_wandb=True)
    dataloader = dataset.build_dataloader(
        batch_size=1,
        num_threads=4,
        shuffle=True,
    )
    logger.info(f"{len(dataloader)=}")
    for batch, *_ in dataloader:
        images, metadata = dataset.batch_extract(batch)
        captions = [datum["caption"] for datum in metadata]
        logger.info(f"{images.shape=} {captions[0]=}")
        break

    logger.info(f"{images.min()=} {images.max()=}")
    plt.imshow(images[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.title(captions[0])
    plt.axis("off")
    plt.show()
# %%
