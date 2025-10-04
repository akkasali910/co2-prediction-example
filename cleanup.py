"""
This script is intended to be run as an AWS Lambda function.
It deletes a specified SageMaker endpoint and its associated EndpointConfig and Model resources.
"""
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    """
    Lambda handler function to delete a SageMaker endpoint and its configuration.
    """
    sagemaker_client = boto3.client("sagemaker")

    # Extract endpoint name from the event passed by the pipeline
    endpoint_name = event["endpoint_name"]
    logger.info(f"Received request to cleanup resources for endpoint: {endpoint_name}")

    try:
        # 1. Describe the endpoint to get the EndpointConfigName
        endpoint_desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = endpoint_desc['EndpointConfigName']
        logger.info(f"Found EndpointConfig: {endpoint_config_name}")

        # 2. Describe the EndpointConfig to get the ModelName(s)
        endpoint_config_desc = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        model_names = [variant['ModelName'] for variant in endpoint_config_desc['ProductionVariants']]
        logger.info(f"Found {len(model_names)} associated models: {model_names}")

        # 3. Delete the endpoint
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"Successfully initiated deletion for endpoint: {endpoint_name}")

        # 4. Delete the endpoint configuration
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        logger.info(f"Successfully initiated deletion for endpoint config: {endpoint_config_name}")

        # 5. Delete the associated models
        for model_name in model_names:
            try:
                sagemaker_client.delete_model(ModelName=model_name)
                logger.info(f"Successfully initiated deletion for model: {model_name}")
            except sagemaker_client.exceptions.ClientError as model_error:
                # It's possible the model is shared or already deleted, so log and continue
                logger.warning(f"Could not delete model '{model_name}': {model_error}")
        
        return {"statusCode": 200, "body": f"Cleanup initiated for {endpoint_name} and its resources."}
        
    except sagemaker_client.exceptions.ClientError as e:
        # If the endpoint doesn't exist, it's a success from a cleanup perspective
        if e.response['Error']['Code'] == 'ValidationException' and "Could not find endpoint" in e.response['Error']['Message']:
            logger.warning(f"Endpoint '{endpoint_name}' not found. Nothing to delete.")
            return {"statusCode": 200, "body": "Endpoint not found."}
        else:
            logger.error(f"Error deleting endpoint: {e}")
            raise e