import ray
import numpy as np
from ray import train

from catboost_trainer import CatBoostTrainer
from catboost_utils import RayTrainReportCallback


# Initialize Ray
ray.init()

# Generate some sample data
def generate_data(size):
    X = np.random.rand(size, 4)
    # Add some categorical features
    X_cat = np.random.randint(0, 3, size=(size, 2))
    X = np.hstack([X, X_cat])
    # Generate target: a simple function of features plus noise
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, size)
    return X, y

# Create training and validation datasets
train_size, val_size = 1000, 200
X_train, y_train = generate_data(train_size)
X_val, y_val = generate_data(val_size)

# Convert to Ray Dataset
train_dataset = ray.data.from_numpy(X_train).add_column("label", y_train)
val_dataset = ray.data.from_numpy(X_val).add_column("label", y_val)

def train_func(config):
    import catboost
    from ray import train
    
    # Get dataset shards
    train_shard = train.get_dataset_shard("train")
    val_shard = train.get_dataset_shard("validation")
    
    # Convert to pandas
    train_df = train_shard.to_pandas()
    val_df = val_shard.to_pandas()
    
    # Separate features and labels
    train_labels = train_df.pop("label")
    val_labels = val_df.pop("label")
    
    # Define categorical feature indices (last two columns)
    cat_features = list(range(4, 6))
    
    # Initialize model with config parameters
    model = catboost.CatBoostRegressor(
        iterations=config.get("iterations", 100),
        learning_rate=config.get("learning_rate", 0.1),
        depth=config.get("depth", 6),
        loss_function='RMSE',
        cat_features=cat_features,
        train_dir="catboost_info"  # Directory for catboost logs
    )
    
    # Train the model
    model.fit(
        train_df,
        train_labels,
        eval_set=[(val_df, val_labels)],
        callbacks=[RayTrainReportCallback()],
        verbose=50  # Print metrics every 50 iterations
    )
    
    # Save the model in the checkpoint
    import os
    checkpoint_dir = train.get_context().get_trial_dir()
    model_path = os.path.join(checkpoint_dir, "model.cbm")
    model.save_model(model_path)
    train.report({"model_path": model_path})

# Configure the trainer
trainer = CatBoostTrainer(
    train_func,
    train_loop_config={
        "iterations": 200,
        "learning_rate": 0.1,
        "depth": 6
    },
    scaling_config=train.ScalingConfig(num_workers=2),
    # For GPU training, uncomment the following:
    # catboost_config=CatBoostConfig(task_type="GPU", devices="0"),
    datasets={
        "train": train_dataset,
        "validation": val_dataset
    }
)

# Train the model
result = trainer.fit()

# Get the final model
final_model = RayTrainReportCallback.get_model(result.checkpoint)

# Make predictions on validation data
val_predictions = final_model.predict(X_val)

# Calculate and print final RMSE
final_rmse = np.sqrt(np.mean((val_predictions - y_val) ** 2))
print(f"Final validation RMSE: {final_rmse}")
