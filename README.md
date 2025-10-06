# CO2 Prediction Project: Workflow and Script Guide

This document provides a comprehensive overview of the files in this project and a step-by-step guide on how to run the end-to-end MLOps workflow.

---

## 1. Project Overview

This project implements a complete MLOps pipeline on AWS to train, evaluate, deploy, and monitor a CO2 prediction model using Amazon SageMaker. The workflow is automated, reproducible, and includes advanced features like hyperparameter tuning, bias detection, and automated retraining.

The core workflow is:
**Prepare Data -> Tune Model -> Train Best Model -> Check for Bias -> Evaluate -> Deploy -> Monitor**

---

## 2. File & Script Guide

This section explains the purpose of the key scripts in the project.

### Core Pipeline Scripts (`/scripts`)
*   **`preprocess.py`**: Performs data preparation, feature engineering (lags, rolling averages), and splits the data into train, validation, and test sets.
*   **`train.py`**: Trains the XGBoost regression model. It accepts hyperparameters and can train on the full dataset.
*   **`evaluate.py`**: Evaluates the trained model's performance on the test set and generates an `evaluation.json` report.

### Pipeline Orchestration
*   **`launch_pipeline.py`**: The main entry point for the entire MLOps workflow. It defines all pipeline steps and starts an execution.
*   **`cleanup_pipeline.py`**: A separate pipeline to delete the deployed SageMaker endpoint and associated resources, useful for cost control.

### AWS Lambda Scripts (`/scripts`)
*   **`deploy.py`**: Code for a Lambda function that deploys a registered model to a SageMaker endpoint.
*   **`cleanup.py`**: Code for a Lambda function that deletes an endpoint and its resources.

### Monitoring & Automation
*   **`setup_model_monitor.py`**: A one-time script to set up a schedule to monitor the live endpoint for **data drift**.
*   **`setup_bias_monitor.py`**: A one-time script to set up a schedule to monitor for **bias drift**.
*   **`setup_alerts.py`**: Configures email alerts for when model drift is detected.
*   **`setup_retraining_trigger.py`**: Sets up a closed-loop system that automatically triggers a new pipeline run when drift is detected.

### Utility & Testing
*   **`test_endpoint.py`**: A simple script to send a test prediction request to the live endpoint.
*   **`hpo_strategies_example.py`**: An experimental script to compare different hyperparameter tuning strategies.

---

## 3. Recommended Workflow

Follow these steps to execute the entire MLOps lifecycle.

### Step 1: Initial Setup

1.  **Upload Data**: Upload your raw `co2_data.csv` to the S3 bucket and path specified in `launch_pipeline.py`.
2.  **Create Lambda Functions**: In the AWS Console, create the Lambda functions using the code from `deploy.py` and `cleanup.py`.
3.  **Update ARNs**: Paste the ARNs of your new Lambda functions into the `lambda_func_arn` fields in `launch_pipeline.py`.

### Step 2: Launch the Main Pipeline

This is the main entry point for creating and running the entire pipeline.

*   **How to Run**: Ensure you are in a SageMaker environment (like a Studio Notebook) with the correct execution role configured.
```bash
python launch_pipeline.py
```

### 4. Monitor the Pipeline

You can monitor the progress of the pipeline in the Amazon SageMaker console under the **Pipelines** section. The visual graph will show the status of each step in real-time. Once complete, you can find the registered model in the **Model Registry**.
