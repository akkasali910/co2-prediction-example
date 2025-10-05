#!/usr/bin/env python3
"""
This script tests the deployed SageMaker endpoint by sending a sample payload
and printing the model's prediction.
"""
import sagemaker
import json

def main():
    """
    Invokes the SageMaker endpoint and prints the result.
    """
    # --- Configuration ---
    endpoint_name = "pando2-co2-prediction-endpoint"
    
    # --- Create a Predictor object ---
    # The sagemaker.Predictor object provides a high-level interface for inference.
    try:
        predictor = sagemaker.predictor.Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker.Session(),
        )
        print(f"✅ Successfully connected to endpoint: {endpoint_name}")
    except Exception as e:
        print(f"❌ Error connecting to endpoint: {e}")
        print("Please ensure the pipeline has run successfully and the endpoint is 'InService'.")
        return

    # --- Prepare a Sample Payload ---
    # The payload must be a CSV string with the same number of features the model was trained on.
    # The features are defined by the `preprocess.py` script. This example payload
    # has been updated to include the new Temperature feature and its derivatives.
    # The order of features must match the columns in the data from preprocess.py.
    # Example: Occupancy, Temperature, Air_Exchanges, Last_Vent_Maint, CO2_lag_1, Temp_lag_1, etc.
    sample_payload = (
        "5,22.5,3.1,45,"  # Original features
        "850,22.4,4,840,22.2,5,830,22.0,3,"  # Lag features for CO2, Temp, Occupancy
        "845.0,0.5,22.3,0.2,"  # Rolling features for CO2, Temp
        "14,4"  # Time-based features (hour, day_of_week)
    )
    
    print(f"\n🚀 Sending payload:\n{sample_payload}")

    # --- Invoke the Endpoint ---
    response = predictor.predict(sample_payload, initial_args={'ContentType': 'text/csv'})
    
    # The response is a CSV string (as defined in the RegisterModel step).
    predicted_co2 = float(response.decode('utf-8'))
    
    print(f"\n✅ Model Prediction (CO2 PPM): {predicted_co2:.2f}")

if __name__ == "__main__":
    main()