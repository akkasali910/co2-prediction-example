# CO2 Prediction Project: Workflow and Script Guide

This document provides a comprehensive overview of the files in this project and a step-by-step guide on how to run the end-to-end MLOps workflow.

---

## 1. Project Overview

This project implements a complete MLOps pipeline on AWS to train, evaluate, deploy, and monitor a CO2 prediction model using Amazon SageMaker. The workflow is automated, reproducible, and includes advanced features like hyperparameter tuning, bias detection, and automated retraining.

The core workflow is:
**Prepare Data -> Tune Model -> Train Best Model -> Check for Bias -> Evaluate -> Deploy -> Monitor**

---

## 2. Core Pipeline Scripts

These are the Python scripts that perform the main ML tasks. They are executed by SageMaker jobs within the pipeline and are not meant to be run directly by the user.

### `preprocess.py`
*   **Purpose**: Performs data preparation and feature engineering.
*   **Functionality**:
    *   Loads the raw `co2_data.csv`.
    *   Creates time-series features like lags and rolling averages.
    *   Splits the data into training, validation, and test sets.
    *   Saves the processed datasets as Parquet files in S3.
*   **Executed By**: The `PrepareCO2Data` `ProcessingStep` in `launch_pipeline.py`.

### `train.py`
*   **Purpose**: Trains the XGBoost regression model.
*   **Functionality**:
    *   Accepts hyperparameters (like `n_estimators`, `max_depth`) as command-line arguments.
    *   Loads the preprocessed training and validation data.
    *   Can combine training and validation data to train the final model on the full dataset.
    *   Fits the XGBoost model and saves the `model.joblib` artifact.
*   **Executed By**: The `TuneCO2Model` (`TuningStep`) and `TrainBestCO2Model` (`TrainingStep`) in `launch_pipeline.py`.

### `evaluate.py`
*   **Purpose**: Evaluates the trained model's performance.
*   **Functionality**:
    *   Loads the trained model artifact and the test dataset.
    *   Makes predictions on the test set.
    *   Calculates performance metrics (R-squared and MSE).
    *   Saves the metrics to `evaluation.json`, which is used by the pipeline to make decisions.
*   **Executed By**: The `EvaluateCO2Model` `ProcessingStep` in `launch_pipeline.py`.

---

## 3. Pipeline Orchestration & Execution

These scripts are used to define, launch, and manage the SageMaker Pipelines.

### `launch_pipeline.py`
*   **Purpose**: The main entry point for the entire MLOps workflow.
*   **Functionality**:
    *   Defines every step of the pipeline: data prep, quality checks, tuning, training, bias checks, evaluation, and conditional deployment.
    *   Connects the inputs and outputs of each step.
    *   Upserts (creates or updates) the pipeline definition in your AWS account.
    *   Starts a new execution of the pipeline.
*   **How to Use**: Run this script from a SageMaker environment (like Studio) to start the entire process.
    ```bash
    python launch_pipeline.py
    ```

### `cleanup_pipeline.py`
*   **Purpose**: Defines and runs a separate, dedicated pipeline to delete the deployed SageMaker endpoint and its associated resources.
*   **Functionality**:
    *   Contains a single `LambdaStep` that invokes the `cleanup.py` Lambda function.
    *   Useful for tearing down resources manually or on a schedule to control costs.
*   **How to Use**: Run this script when you want to delete the production endpoint.
    ```bash
    python cleanup_pipeline.py
    ```

---

## 4. AWS Lambda Function Scripts

These scripts contain the code for AWS Lambda functions that perform specific serverless tasks. You need to create Lambda functions in the AWS console and paste this code into them.

### `deploy.py`
*   **Purpose**: Deploys a registered SageMaker model to a real-time endpoint.
*   **Functionality**:
    *   Receives a model package ARN from the pipeline.
    *   Creates a SageMaker Model, Endpoint Configuration (with data capture enabled), and Endpoint.
    *   Handles updates if the endpoint already exists.
*   **Used By**: The `DeployCO2Model` `LambdaStep` in `launch_pipeline.py`.

### `cleanup.py`
*   **Purpose**: Deletes a SageMaker endpoint and all its associated resources.
*   **Functionality**:
    *   Receives an endpoint name.
    *   Deletes the Endpoint, Endpoint Configuration, and the underlying Model(s).
*   **Used By**: The `CleanupOldEndpoint` `LambdaStep` in `launch_pipeline.py` and the `cleanup_pipeline.py` script.

---

## 5. Monitoring and Automation Scripts

These are one-time setup scripts to configure monitoring, alerting, and automated retraining for your deployed endpoint.

### `setup_model_monitor.py`
*   **Purpose**: Sets up a schedule to monitor the live endpoint for **data drift**.
*   **Functionality**: Creates a `DefaultModelMonitor` schedule that runs hourly, compares live traffic against the baseline created by the pipeline, and generates drift reports.
*   **How to Use**: Run this script once after the endpoint is successfully deployed.
    ```bash
    python setup_model_monitor.py
    ```

### `setup_bias_monitor.py`
*   **Purpose**: Sets up a schedule to monitor the live endpoint for **bias drift**.
*   **Functionality**: Creates a `ModelBiasMonitor` schedule that runs hourly, recalculates bias metrics on live traffic, and compares them to the Clarify baseline from the pipeline.
*   **How to Use**: Run this script once after the endpoint is successfully deployed.
    ```bash
    python setup_bias_monitor.py
    ```

### `setup_alerts.py`
*   **Purpose**: Configures email alerts for when model drift (data or bias) is detected.
*   **Functionality**: Creates an SNS Topic and an EventBridge rule. The rule listens for monitoring jobs that complete "with violations" and sends a notification to your email.
*   **How to Use**: Run this script once after setting up your monitors. **Remember to edit the script to add your email address.**
    ```bash
    python setup_alerts.py
    ```

### `setup_retraining_trigger.py`
*   **Purpose**: Sets up a fully automated, closed-loop MLOps system.
*   **Functionality**: Creates an EventBridge rule that not only sends an alert but also **triggers a new execution of the main training pipeline** (`launch_pipeline.py`) whenever drift is detected.
*   **How to Use**: Use this *instead of* `setup_alerts.py` if you want automated retraining.
    ```bash
    python setup_retraining_trigger.py
    ```

---

## 6. Utility and Testing Scripts

### `test_endpoint.py`
*   **Purpose**: A simple script to test if the deployed endpoint is working correctly.
*   **Functionality**: Sends a sample payload (a CSV string of feature values) to the endpoint and prints the predicted CO2 value.
*   **How to Use**: Run this after the pipeline has successfully deployed the endpoint to verify it's live.
    ```bash
    python test_endpoint.py
    ```

### `hpo_strategies_example.py`
*   **Purpose**: An experimental script to compare different hyperparameter tuning strategies (Bayesian, Random, Hyperband).
*   **Functionality**: Defines and launches standalone SageMaker Tuning jobs. This is useful for research to determine the most effective tuning strategy before embedding it in the main pipeline.
*   **How to Use**: Modify the script to run the desired strategy and execute it from a SageMaker environment.
    ```bash
    python hpo_strategies_example.py
    ```

---

## 7. Recommended Workflow

1.  **Initial Setup**:
    *   Upload your raw data (`co2_data.csv`) to the S3 bucket specified in `launch_pipeline.py`.
    *   Create the Lambda functions (`deploy.py`, `cleanup.py`) in the AWS Console and update the ARNs in `launch_pipeline.py`.

2.  **Launch the Main Pipeline**:
    *   Run `python launch_pipeline.py`.
    *   Monitor the execution in the SageMaker Studio UI.

3.  **Verify Deployment**:
    *   Once the pipeline succeeds, run `python test_endpoint.py` to confirm the endpoint is live and returning predictions.

4.  **Set Up Monitoring**:
    *   Run `python setup_model_monitor.py` to enable data drift detection.
    *   Run `python setup_bias_monitor.py` to enable bias drift detection.

5.  **Configure Automation**:
    *   Choose your desired level of automation:
        *   For alerts only: `python setup_alerts.py`
        *   For alerts and automated retraining: `python setup_retraining_trigger.py`
    *   Remember to confirm your email subscription for SNS.

6.  **Cleanup (When Needed)**:
    *   Run `python cleanup_pipeline.py` to tear down the endpoint and associated resources to save costs.
