import logging
from typing import Any, Callable, Dict, Optional, Union

import ray.train
from ray.train import Checkpoint
from ray.train.trainer import GenDataset
from ray.train.v2.api.config import RunConfig, ScalingConfig
from ray.train.v2.api.data_parallel_trainer import DataParallelTrainer

from ray.util.annotations import Deprecated

from catboost_utils import CatBoostConfig

logger = logging.getLogger(__name__)


class CatBoostTrainer(DataParallelTrainer):
    """A Trainer for distributed data-parallel CatBoost training.

    Example:
        >>> import catboost
        >>> import ray.data
        >>> from ray.train.catboost import RayTrainReportCallback
        >>> from ray.train.catboost import CatBoostTrainer
        >>> 
        >>> def train_fn_per_worker(config: dict):
        ...     # Get the dataset shard for the worker
        ...     train_ds = ray.train.get_dataset_shard("train")
        ...     eval_ds = ray.train.get_dataset_shard("validation")
        ...     
        ...     # Convert to pandas for CatBoost
        ...     train_df = train_ds.to_pandas()
        ...     eval_df = eval_ds.to_pandas()
        ...     
        ...     train_X = train_df.drop("y", axis=1)
        ...     train_y = train_df["y"]
        ...     eval_X = eval_df.drop("y", axis=1)
        ...     eval_y = eval_df["y"]
        ...     
        ...     # Initialize and train the model
        ...     model = catboost.CatBoostRegressor(
        ...         iterations=10,
        ...         learning_rate=1e-3,
        ...         depth=6
        ...     )
        ...     
        ...     model.fit(
        ...         train_X,
        ...         train_y,
        ...         eval_set=[(eval_X, eval_y)],
        ...         callbacks=[RayTrainReportCallback()]
        ...     )
        ... 
        >>> train_ds = ray.data.from_items([{"x": x, "y": x + 1} for x in range(32)])
        >>> eval_ds = ray.data.from_items([{"x": x, "y": x + 1} for x in range(16)])
        >>> trainer = CatBoostTrainer(
        ...     train_fn_per_worker,
        ...     datasets={"train": train_ds, "validation": eval_ds},
        ...     scaling_config=ray.train.ScalingConfig(num_workers=4),
        ... )
        >>> result = trainer.fit()

    Args:
        train_loop_per_worker: The training function to execute on each worker.
        train_loop_config: Configuration to pass to train_loop_per_worker.
        catboost_config: Configuration for CatBoost distributed training.
        scaling_config: Configuration for scaling training.
        run_config: Configuration for the training run.
        datasets: Datasets to use for training, validation, etc.
        dataset_config: Configuration for dataset ingestion.
        resume_from_checkpoint: Checkpoint to resume training from.
        metadata: Additional metadata for the training run.
    """

    def __init__(
        self,
        train_loop_per_worker: Union[Callable[[], None], Callable[[Dict], None]],
        *,
        train_loop_config: Optional[Dict] = None,
        catboost_config: Optional[CatBoostConfig] = None,
        scaling_config: Optional[ScalingConfig] = None,
        run_config: Optional[RunConfig] = None,
        datasets: Optional[Dict[str, GenDataset]] = None,
        dataset_config: Optional[ray.train.DataConfig] = None,
        resume_from_checkpoint: Optional[Checkpoint] = None,
    ):
        super(CatBoostTrainer, self).__init__(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=train_loop_config,
            backend_config=catboost_config or CatBoostConfig(),
            scaling_config=scaling_config,
            dataset_config=dataset_config,
            run_config=run_config,
            datasets=datasets,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    @classmethod
    @Deprecated
    def get_model(cls, checkpoint: Checkpoint):
        """Retrieve the CatBoost model stored in this checkpoint."""
        raise DeprecationWarning(
            "`CatBoostTrainer.get_model` is deprecated. "
            "Use `RayTrainReportCallback.get_model` instead."
        )
