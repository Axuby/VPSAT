import yaml

def load_config(config_path):
    """
    Loads a YAML configuration file.
    :param config_path: Path to the config file.
    :return: Config dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config