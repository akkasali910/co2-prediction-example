"""
This script is intended to be run as an AWS Lambda function.
It receives a model package ARN and deploys it to a SageMaker endpoint.
"""
import boto3
import logging
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    """
    Lambda handler function to deploy a SageMaker model.
    """
    sagemaker_client = boto3.client("sagemaker")

    # Extract model package ARN from the event passed by the pipeline
    model_package_arn = event["model_package_arn"]
    endpoint_name = event["endpoint_name"]
    instance_type = event.get("instance_type", "ml.m5.large")
    instance_count = event.get("instance_count", 1)
    s3_bucket = event["s3_bucket"] # Get the S3 bucket from the event
    
    logger.info(f"Received request to deploy model: {model_package_arn}")
    logger.info(f"Endpoint name: {endpoint_name}")

    # Create a unique model name
    model_name = f"{endpoint_name}-model-{int(time.time())}"

    # Create a SageMaker Model resource
    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "ModelPackageName": model_package_arn
        },
        ExecutionRoleArn=event["role_arn"]
    )
    logger.info(f"Created SageMaker model: {model_name}")

    # Create an Endpoint Configuration
    endpoint_config_name = f"{endpoint_name}-config-{int(time.time())}"

    # Add Data Capture Configuration to log requests and responses
    data_capture_config = {
        'EnableCapture': True,
        'InitialSamplingPercentage': 100,
        'DestinationS3Uri': f"s3://{s3_bucket}/monitoring/data-capture",
        'CaptureOptions': [
            {'CaptureMode': 'Input'},
            {'CaptureMode': 'Output'}
        ],
        'CsvContentTypes': ['text/csv']
    }

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': instance_count,
            'InstanceType': instance_type
        }]
        ,
        DataCaptureConfig=data_capture_config
    )
    logger.info(f"Created endpoint configuration: {endpoint_config_name}")

    # Create or update the Endpoint
    try:
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        logger.info(f"Successfully initiated endpoint creation for {endpoint_name}.")
    except sagemaker_client.exceptions.ResourceLimitExceeded as e:
        # If endpoint already exists, update it
        logger.warning(f"Endpoint {endpoint_name} already exists. Updating it.")
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        logger.info(f"Successfully initiated endpoint update for {endpoint_name}.")

    return {"statusCode": 200, "body": f"Deployment for {endpoint_name} initiated."}
