"""
This script evaluates the trained model on the test dataset.
It is designed to be run as a SageMaker Processing Job.

Input: The trained model artifact and the test dataset.
Output: An evaluation report in JSON format.
"""
import argparse
import logging
import os
import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run(args):
    logging.info("Starting model evaluation...")

    # Load model
    model_path = os.path.join(args.model_path, "model.joblib")
    model = joblib.load(model_path)

    # Load test data
    test_df = pd.read_parquet(os.path.join(args.test_data, "test.parquet"))
    X_test, y_test = test_df.drop("co2", axis=1), test_df["co2"]

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    logging.info(f"Evaluation Metrics: MSE={mse:.4f}, R2={r2:.4f}")

    # Save evaluation report
    report_dict = {"regression_metrics": {"mse": {"value": mse}, "r2_score": {"value": r2}}}
    os.makedirs(args.output_evaluation, exist_ok=True)
    with open(os.path.join(args.output_evaluation, "evaluation.json"), "w") as f:
        json.dump(report_dict, f)

    logging.info("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--output-evaluation", type=str, required=True)
    args = parser.parse_args()
    run(args)
