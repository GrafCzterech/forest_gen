from .task import task_spec as task

task.register(
    "task", skrl_cfg_entry_point=f"{__name__}:skrl_task_cfg.yaml"
)
