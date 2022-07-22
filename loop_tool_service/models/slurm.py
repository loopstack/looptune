# Setup Submitit for a specific cluster
from pathlib import Path
import copy
from datetime import datetime
import submitit


class SubmititJobSubmitter:
    def __init__(
        self,
        slurm_type="fb",
        output_dir=".",
        partition="",
        slurm_array_parallelism=100,
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=1,
        slurm_mem="32GB",
        timeout_min=240,
        debug=False,
    ) -> None:
        self.slurm_type = slurm_type
        self.output_dir = output_dir
        self.partition = partition
        self.slurm_array_parallelism = slurm_array_parallelism
        self.gpus_per_node = gpus_per_node
        self.tasks_per_node = tasks_per_node
        self.cpus_per_task = cpus_per_task
        self.slurm_mem = slurm_mem
        self.timeout_min = timeout_min
        self.debug = debug

    def get_executor(self):
        d = datetime.today()
        submitit_logdir = Path(self.output_dir) / "submitit_logs"
        submitit_logdir.mkdir(exist_ok=True, parents=True)
        if self.debug:
            executor = submitit.AutoExecutor(folder=submitit_logdir, cluster="debug")
        else:
            executor = submitit.AutoExecutor(
                folder=submitit_logdir,
            )
        executor.update_parameters(
            timeout_min=self.timeout_min,
            slurm_array_parallelism=self.slurm_array_parallelism,
            tasks_per_node=self.tasks_per_node,
            cpus_per_task=self.cpus_per_task,
            slurm_mem=self.slurm_mem,
        )
        
        if self.slurm_type == "fb":
            executor.update_parameters(
                slurm_partition="learnlab"
                if len(self.partition) == 0
                else self.partition,
                gpus_per_node=8,
            )
        elif self.slurm_type == "mila":
            executor.update_parameters(
                slurm_partition="long" if len(self.partition) == 0 else self.partition,
                slurm_additional_parameters={"gres": "gpu:rtx8000:1"},
            )

        return executor
