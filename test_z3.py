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
from mu_zero_smt.utils import RunMode, load_config, load_dataset_split


@ray.remote
def eval_z3_worker(
    benchmark: str,
    dataset_split: dict[RunMode, list[int]],
    solving_timeout: float,
    shared_storage: ActorProxy[SharedStorage],
    worker_id: int,
    num_workers: int,
) -> None:
    for split_name, episode_ids in dataset_split.items():
        dataset = SMTDataset(benchmark)

        batch_size = math.ceil(len(episode_ids) / num_workers)
        batch_start = worker_id * batch_size
        batch_end = min(batch_start + batch_size, len(episode_ids))

        for idx in range(batch_start, batch_end):
            episode_id = episode_ids[idx]

            solver = z3.Solver()

            solver.set("timeout", 1000 * solving_timeout)
            solver.add(z3.parse_smt2_file(str(dataset[episode_id])))

            start_time = perf_counter()

            res = solver.check()

            end_time = perf_counter()

            shared_storage.update_info.remote(
                split_name,
                {
                    "id": episode_id,
                    "name": dataset[episode_id].stem,
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
    dataset_split = load_dataset_split(config)
    solving_timeout = config.env_config["solving_timeout"]

    total = len(SMTDataset(benchmark))

    ray.init(num_cpus=config.num_test_workers)

    shared_storage = (
        ray.remote(SharedStorage)
        .options(name="shared_storage_worker", num_cpus=0)
        .remote({split_name: [] for split_name in dataset_split.keys()})
    )

    for worker_id in range(config.num_test_workers):
        eval_z3_worker.options(name="eval_z3_worker", num_cpus=1).remote(
            benchmark,
            dataset_split,
            solving_timeout,
            shared_storage,
            worker_id,
            config.num_test_workers,
        )

    p_bar = tqdm(total=total)

    while True:
        info = ray.get(shared_storage.get_info_batch.remote(list(dataset_split.keys())))

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

    results = ray.get(shared_storage.get_info_batch.remote(list(dataset_split.keys())))

    ray.shutdown()

    with open(f"{experiment_dir}/z3_results.json", "w+") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
