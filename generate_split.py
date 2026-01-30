import json
import math
import time
from pathlib import Path
from time import perf_counter

import pandas as pd
import ray
import z3  # type: ignore
from matplotlib import pyplot as plt
from ray.actor import ActorProxy
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # type: ignore

from mu_zero_smt.environments.smt.dataset import SMTDataset
from mu_zero_smt.shared_storage import SharedStorage
from mu_zero_smt.utils import load_config


@ray.remote
def eval_z3_worker(
    benchmark: str,
    shared_storage: ActorProxy[SharedStorage],
    solving_timeout: float,
    worker_id: int,
    num_workers: int,
) -> None:
    dataset = SMTDataset(benchmark)

    batch_size = math.ceil(len(dataset) / num_workers)
    batch_start = worker_id * batch_size
    batch_end = min(batch_start + batch_size, len(dataset))

    for idx in range(batch_start, batch_end):
        solver = z3.Solver()

        solver.set("timeout", 1000 * solving_timeout)
        solver.add(z3.parse_smt2_file(str(dataset[idx])))

        start_time = perf_counter()

        res = solver.check()

        end_time = perf_counter()

        shared_storage.update_info.remote(
            "results",
            {
                "id": idx,
                "time": end_time - start_time,
                "successful": res in (z3.sat, z3.unsat),
            },
        )


def main() -> None:
    config = load_config()

    benchmark = config.env_config["benchmark"]
    solving_timeout = config.env_config["solving_timeout"]

    total = len(SMTDataset(benchmark))

    ray.init(num_cpus=config.num_test_workers)

    shared_storage = (
        ray.remote(SharedStorage)
        .options(name="shared_storage_worker", num_cpus=0)
        .remote({"results": []})
    )

    for worker_id in range(config.num_test_workers):
        eval_z3_worker.options(name="eval_z3_worker", num_cpus=1).remote(
            benchmark,
            shared_storage,
            solving_timeout,
            worker_id,
            config.num_test_workers,
        )

    p_bar = tqdm(total=total)

    while True:
        info = ray.get(shared_storage.get_info.remote("results"))

        num_successful = sum(x["successful"] for x in info)
        num_completed = len(info)

        p_bar.set_description(
            f"%: {num_successful / num_completed if num_completed != 0 else 0: .3%}"
        )
        p_bar.n = num_completed
        p_bar.update()

        if num_completed == total:
            break

        time.sleep(1)

    results = ray.get(shared_storage.get_info.remote("results"))

    ray.shutdown()

    df = pd.DataFrame(results)

    split = config.env_config["split"]

    df["difficulty_bin"] = pd.qcut(
        df["time"],
        q=5,
    )

    train_df, temp_df = train_test_split(
        df,
        test_size=split["test"] + split["eval"],
        stratify=df["difficulty_bin"],
    )

    test_df, eval_df = train_test_split(
        temp_df,
        test_size=split["eval"] / (split["test"] + split["eval"]),
        stratify=temp_df["difficulty_bin"],
    )

    train_data = train_df.drop(columns=["difficulty_bin"]).to_dict(orient="records")
    test_data = test_df.drop(columns=["difficulty_bin"]).to_dict(orient="records")
    eval_data = eval_df.drop(columns=["difficulty_bin"]).to_dict(orient="records")

    stratified_split = {
        "train": [x["id"] for x in train_data],
        "test": [x["id"] for x in test_data],
        "eval": [x["id"] for x in eval_data],
    }

    split_dir = Path(__file__).parent / "splits"

    with open(f"{split_dir}/{config.experiment_name}.json", "w+") as f:
        json.dump(stratified_split, f)

    plt.hist(
        [train_df["time"], test_df["time"], eval_df["time"]],
        bins=50,
        stacked=True,
        density=True,
        alpha=0.7,
        label=["train", "test", "eval"],
    )

    plt.savefig(f"{split_dir}/{config.experiment_name}.png")


if __name__ == "__main__":
    main()
