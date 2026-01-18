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
from mu_zero_smt.utils import load_config


def main() -> None:
    config = load_config()

    checkpoint_dir = natsorted(
        f
        for f in (Path(__file__).parent / "results" / config.experiment_name).iterdir()
        if f.is_dir()
    )[-1]

    checkpoint = T.load(f"{checkpoint_dir}/model.checkpoint", weights_only=False)

    ray.init(num_cpus=config.num_test_workers)

    shared_storage_worker = (
        ray.remote(SharedStorage)
        .options(name="shared_storage_worker", num_cpus=0)
        .remote(
            {
                "test_results": [],
                "finished_test_workers": [],
                "weights": checkpoint["best_weights"],
                "terminate": False,
            }
        )
    )

    test_workers = [
        ray.remote(SelfPlay)
        .options(name=f"test_worker_{i + 1}", num_cpus=1)
        .remote(
            checkpoint,
            SMTEnvironment,
            "test",
            config,
            i,
        )
        for i in range(config.num_test_workers)
    ]

    for test_worker in test_workers:
        test_worker.continuous_self_play.remote(shared_storage_worker, None)

    p_bar = tqdm(
        total=len(SMTEnvironment(mode="test", **config.env_config).unique_episodes())
    )

    while True:
        info = ray.get(
            shared_storage_worker.get_info_batch.remote(
                ["test_results", "finished_test_workers"]
            )
        )

        num_successful = sum(x["successful"] for x in info["test_results"])
        num_completed = len(info["test_results"])

        p_bar.set_description(
            f"%: {num_successful / num_completed if num_completed != 0 else 0: .3%}"
        )
        p_bar.n = num_completed
        p_bar.update()

        if len(info["finished_test_workers"]) == config.num_test_workers:
            break

        time.sleep(1)

    results = ray.get(shared_storage_worker.get_info.remote("test_results"))

    ray.shutdown()

    with open(f"{checkpoint_dir}/test_results.json", "w+") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
