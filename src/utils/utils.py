import logging
import sys

import yaml

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_yaml_args(config_file):
    """
    Given a yaml file, open it and parse it into nested dictionariers and lists.
    """
    logger.info(f"Using config: {config_file}")
    with open(config_file) as f:
        config = yaml.safe_load(f)
        logger.info(config)
    return config
