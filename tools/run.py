from datetime import datetime as dt
from pathlib import Path

import click
from loguru import logger

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pipelines.digital_data_etl import digital_data_etl
from pipelines.feature_engineering import feature_engineering
# from pipelines.export_artifact_to_json import export_artifact_to_json

@click.command(
    help="""
        LLM Engineering project CLI v0.0.1. 

        Main entry point for the pipeline execution. 
        This entrypoint is where everything comes together.

        Run the ZenML LLM Engineering project pipelines with various options.

        Run a pipeline with the required parameters. This executes
        all steps in the pipeline in the correct order using the orchestrator
        stack component that is configured in your active ZenML stack.

        Examples:

            \b
            # Run the pipeline with default options
            python run.py
                        
            \b
            # Run the pipeline without cache
            python run.py --no-cache
            
            \b
            # Run only the ETL pipeline
            python run.py --only-etl

    """
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--run-etl",
    is_flag=True,
    default=False,
    help="Whether to run the ETL pipeline.",
)
@click.option(
    "--etl-config-filename",
    default="digital_data_etl.yaml",
    help="Filename of the ETL config file.",
)
@click.option(
    "--run-feature-engineering",
    is_flag=True,
    default=False,
    help="Whether to run the FE pipeline.",
)
def main(
    no_cache: bool = False,
    run_etl: bool = False,
    etl_config_filename: str = "digital_data_etl.yaml",
    run_feature_engineering: bool = False,
    run_evaluation: bool = False,
) -> None:
    assert (
        run_etl
        or run_feature_engineering
        or run_evaluation
    ), "Please specify an action to run."

    # if export_settings:
    #     logger.info("Exporting settings to ZenML secrets.")
    #     tmp_settings.export()

    pipeline_args = {
        "enable_cache": not no_cache,
    }
    root_dir = Path(__file__).resolve().parent.parent

    if run_etl:
        run_args_etl = {}
        pipeline_args["config_path"] = root_dir / "configs" / etl_config_filename
        assert pipeline_args[
            "config_path"
        ].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = (
            f"digital_data_etl_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        digital_data_etl.with_options(**pipeline_args)(**run_args_etl)

    if run_feature_engineering:
        run_args_fe = {}
        pipeline_args["config_path"] = root_dir / "configs" / "feature_engineering.yaml"
        pipeline_args["run_name"] = (
            f"feature_engineering_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        feature_engineering.with_options(**pipeline_args)(**run_args_fe)

if __name__ == "__main__":
    main()
