import json
from pathlib import Path

from loguru import logger

try:
    import boto3
except ModuleNotFoundError:
    logger.warning("Couldn't load AWS or SageMaker imports. Run 'pip install --with aws' to support AWS.")

from dotenv import load_dotenv
import os

# Load from .env file in current or parent directory
load_dotenv()

# Access environment variables
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")


def create_sagemaker_user(username: str):
    assert AWS_REGION, "AWS_REGION is not set."
    assert AWS_ACCESS_KEY, "AWS_ACCESS_KEY is not set."
    assert AWS_SECRET_KEY, "AWS_SECRET_KEY is not set."

    # Create IAM client
    iam = boto3.client(
        "iam",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )

    # Create user
    iam.create_user(UserName=username)

    # Attach necessary policies
    policies = [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AWSCloudFormationFullAccess",
        "arn:aws:iam::aws:policy/IAMFullAccess",
        "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    ]

    for policy in policies:
        iam.attach_user_policy(UserName=username, PolicyArn=policy)

    # Create access key
    response = iam.create_access_key(UserName=username)
    access_key = response["AccessKey"]

    logger.info(f"User '{username}' successfully created.")
    logger.info("Access Key ID and Secret Access Key successfully created.")

    return {"AccessKeyId": access_key["AccessKeyId"], "SecretAccessKey": access_key["SecretAccessKey"]}


if __name__ == "__main__":
    new_user = create_sagemaker_user("sagemaker-deployer")
    

    with Path("sagemaker_user_credentials.json").open("w") as f:
        json.dump(new_user, f)

logger.info("Credentials saved to 'sagemaker_user_credentials.json'")
