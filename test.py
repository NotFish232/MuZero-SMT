import json
import time
from pathlib import Path

import ray
import torch as T
from natsort import natsorted
from tqdm import tqdm  # type: ignore

from mu_zero_smt.environments.smt import SMTEnvironment
from mu_zero_smt.self_play import SelfPlay
from mu_zero_smt.shared_storage import SharedStorage
from mu_zero_smt.utils import RunMode, load_config, load_dataset_split


def main() -> None:
    config = load_config()

    dataset_split = load_dataset_split(config)

    modes: list[RunMode] = ["train", "eval", "test"]

    checkpoint_dir = natsorted(
        f
        for f in (Path(__file__).parent / "results" / config.experiment_name).iterdir()
        if f.is_dir()
    )[-1]

    checkpoint = T.load(f"{checkpoint_dir}/model.checkpoint", weights_only=False)

    checkpoint = (
        {f"{mode}_weights": checkpoint["best_weights"] for mode in modes}
        | {f"{mode}_results": [] for mode in modes}
        | {f"finished_{mode}_workers": [] for mode in modes}
        | {"terminate": False}
    )

    ray.init(num_cpus=config.num_test_workers)

    shared_storage_worker = (
        ray.remote(SharedStorage)
        .options(name="shared_storage_worker", num_cpus=0)
        .remote(checkpoint)
    )

    # Give one worker to train / eval batch each and then the remaining to test
    test_workers = []

    for i in range(config.num_test_workers):
        mode: RunMode = "train" if i == 0 else "eval" if i == 1 else "test"
        episode_ids = dataset_split[mode]

        worker_id = 0 if i <= 1 else i - 2
        num_workers = 1 if i <= 1 else config.num_eval_workers - 2

        test_workers.append(
            ray.remote(SelfPlay)
            .options(name=f"test_worker_{i + 1}", num_cpus=1)
            .remote(
                config,
                checkpoint,
                SMTEnvironment,
                episode_ids,
                mode,
                True,
                worker_id,
                num_workers,
            )
        )

    for test_worker in test_workers:
        test_worker.continuous_self_play.remote(shared_storage_worker, None)

    total_episodes = len(SMTEnvironment(**config.env_config).unique_episodes())

    p_bar = tqdm(total=total_episodes)

    while True:
        keys = [f"{mode}_results" for mode in modes] + [
            f"finished_{mode}_workers" for mode in modes
        ]
        info = ray.get(shared_storage_worker.get_info_batch.remote(keys))

        num_successful = sum(
            x["successful"] for mode in modes for x in info[f"{mode}_results"]
        )
        num_completed = sum(len(info[f"{mode}_results"]) for mode in modes)

        p_bar.set_description(
            f"%: {num_successful / num_completed if num_completed != 0 else 0: .3%}"
        )
        p_bar.n = num_completed
        p_bar.update()

        if (
            sum(len(info[f"finished_{mode}_workers"]) for mode in modes)
            == config.num_test_workers
        ):
            break

        time.sleep(1)

    results = ray.get(
        shared_storage_worker.get_info_batch.remote(
            [f"{mode}_results" for mode in modes]
        )
    )

    results = {k.removesuffix("_results"): v for k, v in results.items()}

    ray.shutdown()

    with open(f"{checkpoint_dir}/test_results.json", "w+") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
