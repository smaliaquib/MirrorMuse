import os

import opik
from loguru import logger
from opik.configurator.configure import OpikConfigurator

from dotenv import load_dotenv
import os

# Load from .env file in current or parent directory
from dotenv import load_dotenv
load_dotenv()


def configure_opik() -> None:
    if os.getenv("COMET_API_KEY") and os.getenv("COMET_PROJECT"):
        try:
            print(os.getenv("COMET_API_KEY"))
            client = OpikConfigurator(api_key=os.getenv("COMET_API_KEY"))
            default_workspace = client._get_default_workspace()
        except Exception:
            logger.warning("Default workspace not found. Setting workspace to None and enabling interactive mode.")
            default_workspace = None

        os.environ["OPIK_PROJECT_NAME"] = os.getenv("COMET_PROJECT")

        opik.configure(api_key=os.getenv("COMET_API_KEY"), workspace=default_workspace, use_local=False, force=True)
        logger.info("Opik configured successfully.")
    else:
        logger.warning(
            "COMET_API_KEY and COMET_PROJECT are not set. Set them to enable prompt monitoring with Opik (powered by Comet ML)."
        )
