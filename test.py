import json
import time
from pathlib import Path

import ray
import torch as T
from tqdm import tqdm  # type: ignore

from mu_zero_smt.environments.smt import SMTEnvironment
from mu_zero_smt.self_play import SelfPlay
from mu_zero_smt.shared_storage import SharedStorage

CHECKPOINT_DIR = f"{next(f for f in Path("results/smt").rglob("*") if f.is_dir())}"


@T.no_grad()
def main() -> None:
    config = SMTEnvironment.get_config()

    checkpoint = T.load(f"{CHECKPOINT_DIR}/model.checkpoint", weights_only=False)

    ray.init(num_cpus=config.num_test_workers)

    shared_storage_worker = (
        ray.remote(SharedStorage)
        .options(name="shared_storage_worker", num_cpus=0)
        .remote(
            {
                "test_results": [],
                "finished_test_workers": [],
                "weights": checkpoint["weights"],
                "training_step": 0,
                "terminate": False,
            },
            config,
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
            config.seed + i,
            i,
        )
        for i in range(config.num_test_workers)
    ]

    for test_worker in test_workers:
        test_worker.continuous_self_play.remote(shared_storage_worker, None)

    p_bar = tqdm(total=len(SMTEnvironment(mode="test").unique_episodes()))

    while True:
        info = ray.get(
            shared_storage_worker.get_info_batch.remote(
                ["test_results", "finished_test_workers"]
            )
        )

        p_bar.n = len(info["test_results"])
        p_bar.update()

        if len(info["finished_test_workers"]) == config.num_test_workers:
            break

        time.sleep(1)

    results = ray.get(shared_storage_worker.get_info.remote("test_results"))

    ray.shutdown()

    with open(f"{CHECKPOINT_DIR}/test_results.json", "w+") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
