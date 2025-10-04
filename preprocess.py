"""
This script performs data preparation and feature engineering for the CO2 regression model.
It is designed to be run as a SageMaker Processing Job.

Input: Raw CSV data with 'timestamp', 'co2', 'temperature', 'occupancy' columns.
Output: train, validation, and test datasets in Parquet format.
"""
import argparse
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run(args):
    logging.info("Starting data preparation and feature engineering...")

    # Load data from input path
    input_data_path = os.path.join(args.input_data, "co2_data.csv")
    logging.info(f"Loading data from {input_data_path}")
    df = pd.read_csv(input_data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    # --- Feature Engineering (as designed in Data Wrangler) ---
    logging.info("Creating time-series features...")
    # Lag features
    for col in ['co2', 'temperature', 'occupancy']:
        for lag in [1, 3, 6]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Rolling window features
    for col in ['co2', 'temperature']:
        df[f'{col}_rolling_mean_3'] = df[col].rolling(window=3).mean()
        df[f'{col}_rolling_std_3'] = df[col].rolling(window=3).std()

    # Time-based features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    # Drop rows with NaN values created by shifts/rolling windows
    df.dropna(inplace=True)
    logging.info(f"Dataset shape after feature engineering: {df.shape}")

    # Split data into train, validation, and test sets
    target = 'co2'
    features = [col for col in df.columns if col != target]
    X = df[features]
    y = df[target]

    # First split: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    # Second split: Split training set into 75% train, 25% validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=False)

    logging.info(f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}")

    # Save datasets
    pd.concat([y_train, X_train], axis=1).to_parquet(os.path.join(args.output_train, "train.parquet"))
    pd.concat([y_val, X_val], axis=1).to_parquet(os.path.join(args.output_validation, "validation.parquet"))
    pd.concat([y_test, X_test], axis=1).to_parquet(os.path.join(args.output_test, "test.parquet"))

    logging.info("Data preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--output-train", type=str, required=True)
    parser.add_argument("--output-validation", type=str, required=True)
    parser.add_argument("--output-test", type=str, required=True)
    args = parser.parse_args()
    run(args)
