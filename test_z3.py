import json
import sys
from pathlib import Path

import z3  # type: ignore
from tqdm import tqdm  # type: ignore

from mu_zero_smt.environments.smt.dataset import SMTDataset
from mu_zero_smt.utils.config import load_config


def main() -> None:
    experiment_name = sys.argv[1]

    config = load_config(experiment_name)

    experiment_dir = Path(__file__).parent / "results" / config.experiment_name

    benchmark = config.env_config["benchmark"]
    split = config.env_config["split"]
    solving_timeout = config.env_config["solving_timeout"]

    results = {}

    p_bar = tqdm(total=len(SMTDataset(benchmark, "train", split).benchmark_files))

    for split_name in split.keys():
        dataset = SMTDataset(benchmark, split_name, split)

        split_results = []

        for i in range(len(dataset)):
            solver = z3.Solver()

            solver.set("timeout", solving_timeout * 1000)
            solver.add(z3.parse_smt2_file(str(dataset[i])))

            res = solver.check()

            split_results.append(
                {
                    "id": dataset.idxs[i],
                    "name": dataset[i].stem,
                    "result": str(res),
                    "successful": res in (z3.sat, z3.unsat),
                }
            )

            p_bar.update()

        results[split_name] = split_results

    with open(f"{experiment_dir}/z3_results.json", "w+") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
