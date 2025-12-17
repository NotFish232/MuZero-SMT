import json
import math
import time
from pathlib import Path
from time import perf_counter

import ray
import z3  # type: ignore
from ray.actor import ActorProxy
from tqdm import tqdm  # type: ignore

from mu_zero_smt.environments.smt.dataset import SMTDataset
from mu_zero_smt.shared_storage import SharedStorage
from mu_zero_smt.utils.config import load_config
from mu_zero_smt.utils.utils import RunMode


@ray.remote
def eval_z3_worker(
    benchmark: str,
    batch_split: dict[RunMode, tuple[int, int]],
    solving_timeout: float,
    shared_storage: ActorProxy[SharedStorage],
) -> None:
    for split_name, (start, end) in batch_split.items():
        dataset = SMTDataset(benchmark, split_name)

        for idx in range(start, end):
            solver = z3.Solver()

            solver.set("timeout", 1000 * solving_timeout)
            solver.add(z3.parse_smt2_file(str(dataset[idx])))

            start_time = perf_counter()

            res = solver.check()

            end_time = perf_counter()

            shared_storage.update_info.remote(
                split_name,
                {
                    "id": dataset.idxs[idx],
                    "name": dataset[idx].stem,
                    "time": end_time - start_time,
                    "result": str(res),
                    "successful": res in (z3.sat, z3.unsat),
                },
            )


def main() -> None:
    config = load_config()

    experiment_dir = Path(__file__).parent / "results" / config.experiment_name
    experiment_dir.mkdir(exist_ok=True, parents=True)

    benchmark = config.env_config["benchmark"]
    split = config.env_config["split"]
    solving_timeout = config.env_config["solving_timeout"]

    batch_splits = []

    for worker_id in range(config.num_test_workers):
        worker_split = {}

        for split_name in split.keys():
            dataset = SMTDataset(benchmark, split_name)

            batch_size = math.ceil(len(dataset) / config.num_test_workers)

            batch_start = worker_id * batch_size
            batch_end = min(batch_start + batch_size, len(dataset))

            worker_split[split_name] = (batch_start, batch_end)

        batch_splits.append(worker_split)

    total = len(SMTDataset(benchmark, "train", split).benchmark_files)

    ray.init(num_cpus=config.num_test_workers)

    shared_storage = (
        ray.remote(SharedStorage)
        .options(name="shared_storage_worker", num_cpus=0)
        .remote({split_name: [] for split_name in split.keys()}, None)
    )

    for batch_split in batch_splits:
        eval_z3_worker.options(name="eval_z3_worker", num_cpus=1).remote(
            benchmark, batch_split, solving_timeout, shared_storage
        )

    p_bar = tqdm(total=total)

    while True:
        info = ray.get(shared_storage.get_info_batch.remote(list(split.keys())))

        num_successful = sum(x["successful"] for v in info.values() for x in v)
        num_completed = sum(len(v) for v in info.values())

        p_bar.set_description(
            f"%: {num_successful / num_completed if num_completed != 0 else 0: .3%}"
        )
        p_bar.n = num_completed
        p_bar.update()

        if num_completed == total:
            break

        time.sleep(1)

    results = ray.get(shared_storage.get_info_batch.remote(list(split.keys())))

    ray.shutdown()

    with open(f"{experiment_dir}/z3_results.json", "w+") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
