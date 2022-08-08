# slurm_launch.py
# Usage:
# python launcher/slurm_launch.py -e specs/1d_5x32/exp_6.yaml specs/1d_5x32/exp_7.yaml -n 4

import argparse
from datetime import datetime
import subprocess
import sys

from pathlib import Path

TEMPLATE_FILE = Path(__file__).parent / "slurm_template.sh"
JOB_NAME = "${JOB_NAME}"
OUTPUT_FILE = "${OUTPUT_FILE}"
NUM_NODES = "${NUM_NODES}"
GPUS_PER_NODE_OPTION = "${GPUS_PER_NODE_OPTION}"
CPUS_PER_NODE_OPTION = "${CPUS_PER_NODE_OPTION}"
PARTITION_OPTION = "${PARTITION_OPTION}"
QOS_OPTION = "${QOS_OPTION}"
TIME_OPTION = "${TIME_OPTION}"
MAIL_OPTION = "${MAIL_OPTION}"
COMMAND_PLACEHOLDER = "${COMMAND_PLACEHOLDER}"
GIVEN_NODE = "${GIVEN_NODE}"
LOAD_ENV = "${LOAD_ENV}"
# MAX_PENDING_TRIALS = "${MAX_PENDING_TRIALS}"

DEFAULT_RUNNER = "specs/runner.yaml"
DEFAULT_RUNNER_RESUME = "specs/runner_resume.yaml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slurm launcher")
    parser.add_argument(
        "--exp-files",
        "-e",
        nargs="+",
        type=str,
        required=True,
        help="Experiment spec files.",
    )
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Experiment runner spec file.",
    )
    parser.add_argument(
        "--repo-dir",
        "-d",
        type=str,
        help="The root directory of the repository.",
    )
    parser.add_argument(
        "--num-nodes", "-n", type=int, default=1, help="Number of nodes to use."
    )
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        help="The specified nodes to use. Same format as the "
        "return of 'sinfo'. Default: ''.",
    )
    parser.add_argument(
        "--num-gpus",
        "-ng",
        type=int,
        default=1,
        help="Number of GPUs to use in each node.",
    )
    parser.add_argument(
        "--num-cpus",
        "-nc",
        type=int,
        help="Number of CPUs to use in each node.",
    )
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
    )
    parser.add_argument(
        "--qos",
        "-q",
        type=str,
    )
    parser.add_argument(
        "--time",
        "-t",
        type=str,
        default="1:00",
        help="Set a limit on the total run time of the job allocation. "
        "Acceptable time formats include "
        "`minutes`, `minutes:seconds`, `hours:minutes:seconds`, `days-hours`, "
        "`days-hours:minutes` and `days-hours:minutes:seconds`",
    )
    parser.add_argument(
        "--mail",
        "-m",
        default="",
        type=str,
        help="User to receive email notification of state changes.",
    )
    parser.add_argument(
        "--load-env",
        "-l",
        type=str,
        default="source launcher/prepare.sh",
        help="The script to load your environment ('module load cuda/10.1')",
    )

    args = parser.parse_args()
    runner_file = DEFAULT_RUNNER_RESUME if args.resume else DEFAULT_RUNNER
    repo_dir = (
        Path(__file__).absolute().parents[1]
        if args.repo_dir is None
        else Path(args.repo_dir)
    )

    for exp_file in args.exp_files:
        exp_file = Path(exp_file)
        log_dir = repo_dir / Path("results") / exp_file.parent.name / "runs"

        log_dir.mkdir(parents=True, exist_ok=True)
        command = f"python -u rllib_torch.py --slurm --sweep=5" 
        # command = f"python -u {repo_dir}/main.py -e {exp_file} -r {runner_file}"


        exp_name = f"{exp_file.stem}_{datetime.now():%m_%d_%H_%M}"
        job_name = f"{exp_file.parent.name}_{exp_name}"

        output_file = log_dir / f"{exp_name}.log"
        partition_option = (
            f"#SBATCH --partition={args.partition}" if args.partition else ""
        )
        qos_option = f"#SBATCH --qos={args.qos}" if args.qos else ""
        time_option = f"#SBATCH --time={args.time}" if args.time else ""
        mail_option = (
            f"#SBATCH --mail-user={args.mail}\n#SBATCH --mail-type=ALL"
            if args.mail
            else ""
        )
        cpus_option = (
            f"#SBATCH --cpus-per-task={args.num_cpus}" if args.num_cpus else ""
        )
        gpus_option = f"#SBATCH --gres=gpu:{args.num_gpus}" if args.num_gpus else ""
        node_info = f"#SBATCH -w {args.node}" if args.node else "".format()

        # ===== Modified the template script =====
        with open(TEMPLATE_FILE, "r") as f:
            text = f.read()
        text = text.replace(JOB_NAME, job_name)
        text = text.replace(OUTPUT_FILE, str(output_file))
        text = text.replace(NUM_NODES, str(args.num_nodes))
        text = text.replace(GPUS_PER_NODE_OPTION, gpus_option)
        text = text.replace(CPUS_PER_NODE_OPTION, cpus_option)
        text = text.replace(PARTITION_OPTION, partition_option)
        text = text.replace(QOS_OPTION, qos_option)
        text = text.replace(TIME_OPTION, time_option)
        text = text.replace(MAIL_OPTION, mail_option)
        text = text.replace(COMMAND_PLACEHOLDER, str(command))
        text = text.replace(LOAD_ENV, str(args.load_env))
        text = text.replace(GIVEN_NODE, node_info)
        text = text.replace(
            "# THIS SCRIPT IS A TEMPLATE!",
            "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
            "RUNNABLE!",
        )

        # ===== Save the script =====
        script_file = log_dir / f"{exp_name}.sh"
        with script_file.open(mode="w") as f:
            f.write(text)

        # TODO: Try to run sweep config and to 
        # sweep_id = wandb.sweep(sweep_config, project="loop_tool_agent")
        # wandb.agent(sweep_id=sweep_id, function=submit_job, count=sweep_count)

        # ===== Submit the job =====
        print("Starting to submit job!")
        subprocess.Popen(["sbatch", "-D", str(repo_dir), str(script_file)])
        print(
            f"Job submitted! Script file is at: {script_file} . Log file is at: {output_file}"
        )

    sys.exit(0)