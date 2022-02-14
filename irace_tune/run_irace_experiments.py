import subprocess
import os
from pathlib import Path

if __name__ == '__main__':
    budgets = [200, 400]
    exper_runs = 2

    for budget in budgets:
        for run in range(1, exper_runs + 1):
            tmp_dir = f"temp_dir/irace-out-b{budget}-r{run}"
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
            stdout_fn = f"{tmp_dir}/stdout.txt"
            with open(stdout_fn, "w+") as file_open:
                subprocess.run(["/ufs/schoonho/R/x86_64-redhat-linux-gnu-library/4.0/irace/bin/irace",
                    "-s", "scenario.txt",
                    f"--exec-dir={tmp_dir}",
                    "--max-experiments", str(budget)],
                    stdout=file_open)
                    #stderr=f"{tmp_dir}/stderr.txt")
