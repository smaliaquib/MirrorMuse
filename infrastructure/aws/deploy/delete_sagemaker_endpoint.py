import os
from loguru import logger

# try:
import boto3
from botocore.exceptions import ClientError
# except ModuleNotFoundError:
#     logger.warning("Couldn't load AWS or SageMaker imports. Run 'poetry install --with aws' to support AWS.")


from dotenv import load_dotenv

load_dotenv()


def delete_endpoint_and_config(endpoint_name) -> None:
    """
    Deletes an AWS SageMaker endpoint and its associated configuration.
    Args:
    endpoint_name (str): The name of the SageMaker endpoint to delete.
    Returns:
    None
    """

    try:
        sagemaker_client = boto3.client(
            "sagemaker",
            region_name=os.getenv('AWS_REGION'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('AWS_SECRET_KEY')
        )
    except Exception:
        logger.exception("Error creating SageMaker client")

        return

    # Get the endpoint configuration name
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        config_name = response["EndpointConfigName"]
    except ClientError:
        logger.error("Error getting endpoint configuration and modelname.")

        return

    # Delete the endpoint
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"Endpoint '{endpoint_name}' deletion initiated.")
    except ClientError:
        logger.error("Error deleting endpoint")

    try:
        response = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
        model_name = response["ProductionVariants"][0]["ModelName"]
    except ClientError:
        logger.error("Error getting model name.")

    # Delete the endpoint configuration
    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
        logger.info(f"Endpoint configuration '{config_name}' deleted.")
    except ClientError:
        logger.error("Error deleting endpoint configuration.")

    # Delete models
    try:
        sagemaker_client.delete_model(ModelName=model_name)
        logger.info(f"Model '{model_name}' deleted.")
    except ClientError:
        logger.error("Error deleting model.")


if __name__ == "__main__":
    endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_INFERENCE")
    logger.info(f"Attempting to delete endpoint: {endpoint_name}")
    delete_endpoint_and_config(endpoint_name=endpoint_name)
