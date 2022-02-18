import subprocess
import os
from pathlib import Path
from timeit import default_timer as timer


if __name__ == '__main__':
    budgets = [200, 400, 800, 1600, 3200, 6400]
    exper_runs = 10

    start = timer()
    for budget in budgets:
        print("RUNNING BUDGET", budget, "...")
        for run in range(1, exper_runs + 1):
            print("RUN", run, "...")
            tmp_dir = f"temp_dir4/irace-out-b{budget}-r{run}"
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
            stdout_fn = f"{tmp_dir}/stdout.txt"
            with open(stdout_fn, "w+") as file_open:
                subprocess.run(["/ufs/schoonho/R/x86_64-redhat-linux-gnu-library/4.0/irace/bin/irace",
                    "-s", "scenario.txt",
                    f"--exec-dir={tmp_dir}",
                    "--max-experiments", str(budget),
                    '--debug-level', str(3),
                    '--first-test', str(5),
                    '--num-configurations', str(50)],
                    stdout=file_open)
                    #stderr=f"{tmp_dir}/stderr.txt")

    end = timer()
    elapsed = end - start
    print("time in seconds:", elapsed)
    print("time in minutes:", elapsed/60.0)
    print("time in hours:", elapsed/3600.0)
