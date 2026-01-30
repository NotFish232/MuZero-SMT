import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import zstandard  # type: ignore
from natsort import natsorted
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore
from typing_extensions import Self

SMT_LIB_RELEASE = "https://zenodo.org/records/16740866"

DATA_DIR = Path(__file__).parents[3] / "data"


class SMTDataset(Dataset):
    def __init__(self: Self, benchmark: str) -> None:
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

        self.benchmark_files = natsorted(self.benchmark_dir.rglob("*.smt2"))

    def download_logic_benchmark(self: Self, logic: str) -> None:
        """
        Downloads the logic benchmark from SMT-LIB, and uncompresses then untars the directory
        """

        url = f"{SMT_LIB_RELEASE}/files/{logic}.tar.zst"
        zst_destination = DATA_DIR / f"{logic}.tar.zst"

        print(f"Downloading {logic} benchmark from SMT-LIB...")

        p_bar = tqdm()

        def update_progress_bar(count: int, block_size: int, total_size: int) -> None:
            if p_bar.total is None:
                p_bar.total = total_size
                p_bar.refresh()

            if count % 10 == 0:
                p_bar.n = count * block_size
                p_bar.refresh()

        urlretrieve(url, zst_destination, update_progress_bar)

        p_bar.close()

        tar_desination = DATA_DIR / zst_destination.stem

        print("Decompressing zstd file...", end="", flush=True)

        with open(zst_destination, "rb") as compressed:
            decompressor = zstandard.ZstdDecompressor()

            with open(tar_desination, "wb") as uncompressed_tar_file:
                decompressor.copy_stream(compressed, uncompressed_tar_file)

        print(" Done!")

        print("Extracting tar file...", end="", flush=True)

        with tarfile.open(tar_desination) as tar_file:
            tar_file.extractall(DATA_DIR, filter="data")

        print(" Done!")

        zst_destination.unlink()
        tar_desination.unlink()

    def find_benchmark_dir(self: Self, logic: str, benchmark_name: str) -> Path:
        return next(
            file
            for file in (DATA_DIR / "non-incremental" / logic).rglob("*")
            if file.is_dir() and file.name == benchmark_name
        )

    def __len__(self: Self) -> int:
        return len(self.benchmark_files)

    def __getitem__(self: Self, idx: int) -> Path:
        return self.benchmark_files[idx]
