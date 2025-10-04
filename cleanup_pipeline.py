#!/usr/bin/env python3
"""
This script defines and launches a dedicated SageMaker Pipeline for cleaning up
ML resources, specifically SageMaker endpoints.

This pipeline is designed to be run independently of the main training pipeline.
"""
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import LambdaStep
from sagemaker.workflow.parameters import ParameterString

def get_cleanup_pipeline(role, pipeline_name, lambda_cleanup_arn):
    """
    Creates a SageMaker Pipeline that consists of a single Lambda step
    to delete a SageMaker endpoint.
    """
    # Define a pipeline parameter for the endpoint name to make it reusable.
    # You can override the default value when you start a pipeline execution.
    endpoint_name_param = ParameterString(
        name="EndpointName",
        default_value="pando2-co2-prediction-endpoint",
    )

    # Define the Lambda step that will execute the cleanup logic.
    # This step invokes the Lambda function created from 'scripts/cleanup.py'.
    step_cleanup = LambdaStep(
        name="CleanupEndpointResources",
        lambda_func_arn=lambda_cleanup_arn,
        inputs={
            "endpoint_name": endpoint_name_param,
        },
    )

    # Construct the pipeline with the single cleanup step.
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[endpoint_name_param],
        steps=[step_cleanup],
    )
    return pipeline

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Update the ARN to point to your cleanup Lambda function.
    LAMBDA_CLEANUP_ARN = "arn:aws:lambda:YOUR_REGION:YOUR_ACCOUNT_ID:function:sagemaker-cleanup-function"
    PIPELINE_NAME = "Pando2-Resource-Cleanup-Pipeline"
    # -------------------

    try:
        role = sagemaker.get_execution_role()
        sagemaker_session = sagemaker.Session()
        print("✅ Running in a SageMaker environment. Role and session detected.")
    except ValueError:
        print("❌ Could not get execution role. Please run from a SageMaker environment.")
        exit()

    pipeline = get_cleanup_pipeline(role, PIPELINE_NAME, LAMBDA_CLEANUP_ARN)
    
    print(f"Upserting cleanup pipeline: {PIPELINE_NAME}...")
    pipeline.upsert(role_arn=role)
    print("✅ Cleanup pipeline upserted successfully.")
    
    print("Starting cleanup pipeline execution...")
    execution = pipeline.start()
    print(f"✅ Cleanup pipeline execution started with ARN: {execution.arn}")