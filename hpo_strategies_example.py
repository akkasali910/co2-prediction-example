#!/usr/bin/env python3
"""
This script demonstrates how to launch SageMaker Hyperparameter Tuning (HPO) jobs
with different search strategies: Bayesian, Random, and Hyperband.

It provides a template for comparing how each strategy explores the hyperparameter
space to find the best model for a given task.

Prerequisites:
- A SageMaker execution role.
- A training script (e.g., 'scripts/train.py') uploaded to your project.
- Training and validation data located in S3.
"""
import sagemaker
from sagemaker.tuner import (
    HyperparameterTuner,
    IntegerParameter,
    ContinuousParameter,
    CategoricalParameter,
)
from sagemaker.xgboost.estimator import XGBoost

def main():
    """
    Defines and launches SageMaker HPO jobs with different strategies.
    """
    # --- Basic Configuration ---
    try:
        role = sagemaker.get_execution_role()
        sagemaker_session = sagemaker.Session()
        default_bucket = sagemaker_session.default_bucket()
        print("✅ Running in a SageMaker environment. Role and session detected.")
    except ValueError:
        print("❌ Could not get execution role. Please run this from a SageMaker environment (e.g., Studio, Notebook).")
        return

    # --- S3 Data Locations (Update with your actual paths) ---
    # This assumes you have already run a processing job like in your pipeline
    train_s3_path = f"s3://{default_bucket}/co2-output/train"
    validation_s3_path = f"s3://{default_bucket}/co2-output/validation"
    output_path = f"s3://{default_bucket}/hpo-output"

    # --- Base Estimator ---
    # This is the XGBoost estimator that the tuner will use for each training job.
    # The entry_point script must be accessible.
    xgb_estimator = XGBoost(
        entry_point="scripts/train.py", # Assumes a train.py script exists in a 'scripts' folder
        framework_version="1.7-1",
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        output_path=output_path,
    )

    # --- Hyperparameter Ranges ---
    # These are the ranges the tuner will search within.
    hyperparameter_ranges = {
        "n_estimators": IntegerParameter(100, 500),
        "max_depth": IntegerParameter(3, 10),
        "learning_rate": ContinuousParameter(0.01, 0.3),
        "subsample": ContinuousParameter(0.6, 1.0),
    }

    # --- Objective Metric ---
    # The tuner's goal is to optimize this metric.
    # The metric name must match what your training script prints (e.g., "validation:rmse").
    objective_metric_name = "validation:rmse"
    objective_type = "Minimize"

    # --- Strategy 1: Bayesian Search (Intelligent Search) ---
    # Builds a probabilistic model to predict which hyperparameters are most promising.
    # Finds better models with fewer jobs. Ideal for most cases.
    tuner_bayesian = HyperparameterTuner(
        estimator=xgb_estimator,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        objective_type=objective_type,
        max_jobs=20,
        max_parallel_jobs=5,
        strategy="Bayesian",
        base_tuning_job_name="xgb-bayesian",
    )

    # --- Strategy 2: Random Search ---
    # Randomly samples from the hyperparameter space. Highly parallelizable and
    # can sometimes find unexpected good combinations.
    tuner_random = HyperparameterTuner(
        estimator=xgb_estimator,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        objective_type=objective_type,
        max_jobs=20,
        max_parallel_jobs=5,
        strategy="Random",
        base_tuning_job_name="xgb-random",
    )

    # --- Strategy 3: Hyperband (Adaptive Resource Allocation) ---
    # A multi-fidelity method that starts many jobs and quickly stops underperforming ones.
    # Very efficient for large-scale searches.
    tuner_hyperband = HyperparameterTuner(
        estimator=xgb_estimator,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        objective_type=objective_type,
        max_jobs=20,
        max_parallel_jobs=5,
        strategy="Hyperband",
        base_tuning_job_name="xgb-hyperband",
    )

    # --- Launching the Jobs ---
    # You can uncomment the tuner you want to run.
    # Running them all at once will create three separate HPO jobs.
    
    print("🚀 Launching Bayesian HPO Job...")
    tuner_bayesian.fit(
        inputs={"train": train_s3_path, "validation": validation_s3_path},
        wait=False # Set to True to block until the job completes
    )
    print(f"✅ Bayesian job '{tuner_bayesian.latest_tuning_job_name}' started.")

if __name__ == "__main__":
    main()
