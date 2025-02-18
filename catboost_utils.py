import logging
import os
from dataclasses import dataclass
from typing import Optional

from ray.train.backend import Backend, BackendConfig
from ray.train._internal.worker_group import WorkerGroup

logger = logging.getLogger(__name__)

@dataclass
class CatBoostConfig(BackendConfig):
    """Configuration for CatBoost distributed training.

    Args:
        task_type: The type of task to run. Can be "CPU" or "GPU".
        devices: The devices to use for training. For GPU training,
            this should be a comma-separated list of GPU indices.
    """
    task_type: str = "CPU"
    devices: Optional[str] = None

    def __post_init__(self):
        if self.task_type not in ["CPU", "GPU"]:
            raise ValueError(
                f"task_type must be either 'CPU' or 'GPU', got {self.task_type}"
            )

    def backend_cls(self):
        return CatBoostBackend


class CatBoostBackend(Backend):
    """Backend for distributed CatBoost training."""

    def on_start(self, worker_group: WorkerGroup, backend_config: Optional[CatBoostConfig] = None):
        # Get worker IPs for distributed training
        worker_ips = [worker.metadata.node_ip_address for worker in worker_group.workers]
        worker_addresses = [f"{ip}:0" for ip in worker_ips]
        worker_hosts = ",".join(worker_addresses)

        def set_env_vars(task_type: str, devices: Optional[str], worker_hosts: str):
            os.environ["CATBOOST_HOSTS"] = worker_hosts
            if task_type == "GPU":
                os.environ["CUDA_VISIBLE_DEVICES"] = devices or ""

        # Set environment variables on all workers
        worker_group.execute(
            set_env_vars,
            task_type=backend_config.task_type,
            devices=backend_config.devices,
            worker_hosts=worker_hosts,
        )

    def on_shutdown(self, worker_group: WorkerGroup, backend_config: CatBoostConfig):
        def cleanup_env():
            os.environ.pop("CATBOOST_HOSTS", None)
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        worker_group.execute(cleanup_env)


class RayTrainReportCallback:
    """Callback for reporting metrics to Ray Train during CatBoost training.

    Example:
        >>> import catboost
        >>> from ray.train.catboost import RayTrainReportCallback
        >>> model = catboost.CatBoostRegressor()
        >>> model.fit(X, y, eval_set=[(X_val, y_val)],
        ...          callbacks=[RayTrainReportCallback()])
    """

    def after_iteration(self, info):
        """Report metrics after each iteration."""
        import ray.train

        metrics = {}
        for metric_name, value in info.metrics.items():
            for dataset_name, metric_value in value.items():
                # Convert metric names to format: {dataset_name}_{metric_name}
                metrics[f"{dataset_name}_{metric_name}"] = metric_value[-1]

        ray.train.report(metrics)
        return True  # Continue training

    @staticmethod
    def get_model(checkpoint):
        """Get the CatBoost model from a checkpoint.

        Args:
            checkpoint: Checkpoint object returned by ray.train.get_checkpoint().

        Returns:
            The CatBoost model.
        """
        import catboost
        return catboost.CatBoost().load_model(checkpoint.to_directory())
