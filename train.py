"""
This script trains an XGBoost regression model.
It is designed to be run as a SageMaker Training Job.

Input: Preprocessed train and validation data in Parquet format.
Output: A trained model artifact (model.tar.gz).
"""
import argparse
import logging
import os
import pandas as pd
import xgboost as xgb
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run(args):
    logging.info("Starting model training...")

    # Load data
    train_path = os.path.join(args.train, "train.parquet")
    validation_path = os.path.join(args.validation, "validation.parquet")
    
    train_df = pd.read_parquet(train_path)
    logging.info(f"Loaded training data from {train_path}, shape: {train_df.shape}")

    # If the validation channel contains data, concatenate it with the training data.
    # This allows the final model to be trained on the full (train + validation) dataset.
    if os.path.exists(validation_path) and os.path.getsize(validation_path) > 0:
        logging.info(f"Validation data found at {validation_path}. Concatenating with training data.")
        val_df = pd.read_parquet(validation_path)
        train_df = pd.concat([train_df, val_df], ignore_index=True)
        logging.info(f"Combined dataset shape: {train_df.shape}")

    X_train, y_train = train_df.drop("co2", axis=1), train_df["co2"]

    # Configure and train the XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=42
    )

    logging.info(f"Training XGBoost model with hyperparameters: {model.get_params()}")
    # Note: When training on the full dataset, we don't have a separate validation set
    # for early stopping. The number of rounds (n_estimators) should be one of the
    # optimal hyperparameters found during the tuning step.
    # The original HPO step uses early stopping, this step does not.
    model.fit(X_train, y_train, verbose=False)

    # Save the trained model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # SageMaker environment variables and hyperparameters
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    args = parser.parse_args()
    run(args)
