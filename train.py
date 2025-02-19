import ray
import numpy as np
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
import catboost
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ray.init()

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def train_catboost(config):
    # Convert data to CatBoost format
    train_dataset = catboost.Pool(
        data=X_train, 
        label=y_train
    )
    val_dataset = catboost.Pool(
        data=X_val, 
        label=y_val
    )
    
    model = catboost.CatBoostClassifier(  # Changed to classifier since this is a binary classification task
        iterations=config["iterations"],
        learning_rate=config["learning_rate"],
        depth=config["depth"],
        loss_function='Logloss',  # Changed to Logloss for binary classification
        train_dir="catboost_info"
    )
    
    model.fit(
        train_dataset,
        eval_set=[val_dataset],
        verbose=50
    )
    
    val_predictions = model.predict(X_val)
    accuracy = np.mean(val_predictions == y_val)
    
    train.report({
        "accuracy": accuracy,
        "done": True
    })

if __name__ == "__main__":
    config = {
        "iterations": tune.choice([100, 200, 300]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "depth": tune.randint(4, 10),
    }

    tuner = tune.Tuner(
        train_catboost,
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=ASHAScheduler(
                max_t=300,
                grace_period=20,
                reduction_factor=2
            ),
            num_samples=5
        ),
        param_space=config,
    )

    results = tuner.fit()

    best_result = results.get_best_result()
    print(f"Best hyperparameters found were: {best_result.config}")
    print(f"Best accuracy: {best_result.metrics['accuracy']}")
