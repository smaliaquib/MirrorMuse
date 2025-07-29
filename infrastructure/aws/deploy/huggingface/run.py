from loguru import logger

# try:
from sagemaker.enums import EndpointType
from sagemaker.huggingface import get_huggingface_llm_image_uri
import boto3
# except ModuleNotFoundError:
#     logger.warning("Couldn't load SageMaker imports. Run 'pip install --with aws' to support AWS.")

from model.utils import ResourceManager
from dotenv import load_dotenv
import os

# Load env vars
load_dotenv()

from .config import hugging_face_deploy_config, model_resource_config
from .sagemaker_huggingface import DeploymentService, SagemakerHuggingfaceStrategy


def delete_existing(endpoint_name: str, endpoint_config_name: str, model_name: str):
    sagemaker = boto3.client("sagemaker", region_name=os.getenv("AWS_REGION"))

    # Delete endpoint if exists
    try:
        sagemaker.describe_endpoint(EndpointName=endpoint_name)
        logger.info(f"Deleting existing endpoint: {endpoint_name}")
        sagemaker.delete_endpoint(EndpointName=endpoint_name)
    except sagemaker.exceptions.ClientError:
        logger.info("Endpoint does not exist.")

    # Delete endpoint config if exists
    try:
        sagemaker.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        logger.info(f"Deleting endpoint config: {endpoint_config_name}")
        sagemaker.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    except sagemaker.exceptions.ClientError:
        logger.info("Endpoint config does not exist.")

    # Delete model if exists
    try:
        sagemaker.describe_model(ModelName=model_name)
        logger.info(f"Deleting model: {model_name}")
        sagemaker.delete_model(ModelName=model_name)
    except sagemaker.exceptions.ClientError:
        logger.info("Model does not exist.")


def create_endpoint(endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED) -> None:
    assert os.getenv("AWS_ARN_ROLE"), "AWS_ARN_ROLE is not set in the .env file."

    endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_INFERENCE")
    endpoint_config_name = os.getenv("SAGEMAKER_ENDPOINT_CONFIG_INFERENCE")
    model_name = f"huggingface-pytorch-tgi-inference-{os.getenv('DEPLOY_TIMESTAMP')}" if os.getenv("DEPLOY_TIMESTAMP") else f"{endpoint_name}-model"

    logger.info(f"Creating endpoint with endpoint_type = {endpoint_type} and model_id = {os.getenv('HF_MODEL_ID')}")

    # Cleanup old resources
    delete_existing(endpoint_name, endpoint_config_name, model_name)

    llm_image = get_huggingface_llm_image_uri("huggingface", version="2.3.1")

    resource_manager = ResourceManager()
    deployment_service = DeploymentService(resource_manager=resource_manager)

    SagemakerHuggingfaceStrategy(deployment_service).deploy(
        role_arn=os.getenv("AWS_ARN_ROLE"),
        llm_image=llm_image,
        config=hugging_face_deploy_config,
        endpoint_name=endpoint_name,
        endpoint_config_name=endpoint_config_name,
        gpu_instance_type=os.getenv("GPU_INSTANCE_TYPE"),
        resources=model_resource_config,
        endpoint_type=endpoint_type,
    )


if __name__ == "__main__":
    create_endpoint(endpoint_type=EndpointType.MODEL_BASED)
