from pathlib import Path

import pytest

from syntaclip.utils.datasets import cc3m


def test_cc3m():
    CC3M_DATA_DIR = Path.home() / "datasets" / "WebDataset" / "CC3M"
    if not CC3M_DATA_DIR.exists():
        return
    dataset = cc3m.CC3MDataset(CC3M_DATA_DIR, split="train")
    dataset.download()
    dataloader = dataset.build_dataloader()
    assert dataloader is not None
    assert len(dataloader) > 0
