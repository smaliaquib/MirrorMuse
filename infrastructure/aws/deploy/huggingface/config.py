import json

from loguru import logger

# try:
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
# except ModuleNotFoundError:
#     logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from dotenv import load_dotenv
import os

# Load from .env file in current or parent directory
from dotenv import load_dotenv
load_dotenv()

hugging_face_deploy_config = {
    "HF_MODEL_ID": os.getenv("HF_MODEL_ID"),
    "HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
    "SM_NUM_GPUS": str(int(os.getenv("SM_NUM_GPUS"))),
    "MAX_INPUT_LENGTH": str(int(os.getenv("MAX_INPUT_LENGTH"))),
    "MAX_TOTAL_TOKENS": str(int(os.getenv("MAX_TOTAL_TOKENS"))),
    "MAX_BATCH_TOTAL_TOKENS": str(int(os.getenv("MAX_BATCH_TOTAL_TOKENS"))),
    "MAX_BATCH_PREFILL_TOKENS": str(int(os.getenv("MAX_BATCH_TOTAL_TOKENS"))),
    "HF_MODEL_QUANTIZE": "bitsandbytes",
}


model_resource_config = ResourceRequirements(
    requests={
        "copies": int(os.getenv("COPIES")),  # Number of replicas.
        "num_accelerators": int(os.getenv("GPUS")),  # Number of GPUs required.
        "num_cpus": int(os.getenv("CPUS")),  # Number of CPU cores required.
        "memory": 5 * 1024,  # Minimum memory required in Mb (required)
    },
)
