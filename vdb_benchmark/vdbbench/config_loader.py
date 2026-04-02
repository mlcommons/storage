#!/usr/bin/env python3
import yaml
import os

def load_config(config_file=None):
    """
    Load configuration from a YAML file.
    
    Args:
        config_file (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary or empty dict if file not found
    """
    if not config_file:
        return {}

    path_exists = os.path.exists(config_file)
    configs_path_exists = os.path.exists(os.path.join("configs", config_file))
    if path_exists or configs_path_exists:
        config_file = config_file if path_exists else os.path.join("configs", config_file)
    else:
        print(f"ERROR: Configuration file not found: {config_file}")
        return {}
        
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Loaded vdbbench configuration from {config_file}")
            return config
    except Exception as e:
        print("ERROR - Error loading configuration file: {str(e)}")
        return {}


def merge_config_with_args(config, args):
    """
    Merge configuration from YAML with command line arguments.
    Command line arguments take precedence over YAML configuration.
    
    Args:
        config (dict): Configuration dictionary from YAML
        args (Namespace): Parsed command line arguments
        
    Returns:
        Namespace: Updated arguments with values from config where not specified in args
    """
    # Convert args to a dictionary
    args_dict = vars(args)

    # For each key in config, if the corresponding arg is None or has a default value,
    # update it with the value from config
    for section, params in config.items():
        for key, value in params.items():
            if key in args_dict and (args_dict[key] is None or
                                     (hasattr(args, 'is_default') and
                                      key in args.is_default and
                                      args.is_default[key])):
                args_dict[key] = value
    
    return args
