from abc import ABC, abstractmethod
from pathlib import Path

from nvidia.dali.plugin.pytorch import DALIGenericIterator
from utils.clogging import getColoredLogger
from wds2idx import IndexCreator

logger = getColoredLogger(__name__)

__all__ = ["DaliDataset"]


class DaliDataset(ABC):
    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir).absolute()
        self.split_dataset_dir = self.dataset_dir / "split"

    @property
    def tarfiles(self):
        return tuple(map(str, sorted(self.split_dataset_dir.glob("*.tar"))))

    @property
    def idxfiles(self):
        return tuple(map(str, sorted(self.split_dataset_dir.glob("*.idx"))))

    @abstractmethod
    def download(self):
        pass

    def wds2idx(self):
        assert self.tarfiles, "No tar files found. Is the dataset downloaded?"
        if self.idxfiles:
            return

        for tar in map(Path, self.tarfiles):
            with IndexCreator(str(tar), str(tar.with_suffix(".idx"))) as c:
                logger.info(f"Creating index for {tar} ...")
                c.create_index()

    @abstractmethod
    def build_dataloader(self) -> DALIGenericIterator:
        pass

    @abstractmethod
    def build_pipeline(self):
        pass
