import json
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import torch as T
import zstandard  # type: ignore
from torch.utils.data import Dataset
from typing_extensions import Self

SMT_LIB_RELEASE = "https://zenodo.org/records/16740866"

DATA_DIR = Path(__file__).parents[3] / "data"


class SMTDataset(Dataset):
    def __init__(
        self: Self, benchmark: str, split_name: str, split: dict[str, float]
    ) -> None:
        """
        Args:
            benchmark (str): A benchmark in the format of "LOGIC/benchmark_name"
            like QF_NIA/CInteger
        """

        self.logic, self.benchmark_name = benchmark.split("/")

        DATA_DIR.mkdir(exist_ok=True)

        if not (DATA_DIR / "non-incremental" / self.logic).exists():
            self.download_logic_benchmark(self.logic)

        self.benchmark_dir = self.find_benchmark_dir(self.logic, self.benchmark_name)

        self.benchmark_files = [*self.benchmark_dir.rglob("*.smt2")]

        # Information about what idxs belong to which split, so we can instantiate diff objects during training and testing
        split_info_file = self.benchmark_dir / "split.json"

        if split_info_file.exists():
            self.split_info = json.load(open(split_info_file, "rt"))
        else:
            self.split_info = self.create_benchmark_split(split)
            json.dump(self.split_info, open(split_info_file, "wt"))

        self.split_name = split_name
        self.idxs = self.split_info[self.split_name]

    def download_logic_benchmark(self: Self, logic: str) -> None:
        """
        Downloads the logic benchmark from SMT-LIB, and uncompresses then untars the directory
        """

        url = f"{SMT_LIB_RELEASE}/files/{logic}.tar.zst"
        zst_destination = DATA_DIR / f"{logic}.tar.zst"

        urlretrieve(url, zst_destination)

        with open(zst_destination, "rb") as compressed:
            decompressor = zstandard.ZstdDecompressor()

            tar_desination = DATA_DIR / zst_destination.stem

            with open(tar_desination, "wb") as tar_file:
                decompressor.copy_stream(compressed, tar_file)

            with tarfile.open(tar_desination) as tar_file:
                tar_file.extractall(DATA_DIR, filter="data")

        zst_destination.unlink()
        tar_desination.unlink()

    def create_benchmark_split(
        self: Self, split: dict[str, float]
    ) -> dict[str, list[int]]:
        idxs = T.randperm(len(self.benchmark_files))

        split_infos = {}

        prev = 0.0

        for name, amount in split.items():
            start_idx = int(prev * len(self.benchmark_files))
            end_idx = int((prev + amount) * len(self.benchmark_files))

            split_infos[name] = idxs[start_idx:end_idx].tolist()

            prev += amount

        return split_infos

    def find_benchmark_dir(self: Self, logic: str, benchmark_name: str) -> Path:
        return next(
            file
            for file in (DATA_DIR / "non-incremental" / logic).rglob("*")
            if file.is_dir() and file.name == benchmark_name
        )

    def __len__(self: Self) -> int:
        return len(self.idxs)

    def __getitem__(self: Self, idx: int) -> Path:
        return self.benchmark_files[self.idxs[idx]]
