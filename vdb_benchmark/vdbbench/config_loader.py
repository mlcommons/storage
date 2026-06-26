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
        print(f"ERROR - Error loading configuration file: {str(e)}")
        return {}


def _flatten_section(section, section_name=None):
    """
    Flatten a single configuration *section* into a one-level mapping of
    leaf parameter name -> value.

    The vdbbench YAML configs group parameters into top-level sections
    (database, dataset, index, workflow, benchmark, ...) and may additionally
    nest index tuning parameters under an ``index_params:`` block, e.g.::

        index:
          index_type: HNSW
          index_params:
            M: 32
            ef_construction: 100

    Argparse exposes those tuning parameters as top-level flags (``--M``,
    ``--ef-construction``). Without flattening, the inner ``index_params``
    dict is seen as a single unknown key and silently dropped, so the values
    never reach the parsed args.

    Flattening is performed **per section**, not globally, to avoid collapsing
    same-named leaf keys that legitimately live in different sections (for
    example ``dataset.batch_size`` and ``benchmark.batch_size``). Collisions
    are only possible within a single section (between a section-level key and
    a key nested under ``index_params:``); those are unexpected, so we raise
    rather than silently pick a winner.

    Args:
        section (dict): One top-level section of the config.
        section_name (str): Name of the section, used for error messages.

    Returns:
        dict: Flat mapping of leaf parameter name -> value for this section.
    """
    flat = {}

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict):
                    _walk(value)
                else:
                    if key in flat and flat[key] != value:
                        raise ValueError(
                            f"Conflicting values for '{key}' within config "
                            f"section '{section_name}': {flat[key]!r} vs "
                            f"{value!r}. Rename one of the keys."
                        )
                    flat[key] = value

    _walk(section)
    return flat


def _flatten_config(config):
    """
    Flatten a (possibly nested) configuration dict into a single-level
    mapping of leaf parameter name -> value, flattening each top-level
    section independently.

    Flattening section-by-section (rather than over the whole document at
    once) prevents a leaf key in one section from silently overwriting a
    same-named key in another section. If the *same* leaf name appears in two
    different sections, the first occurrence (in document order) is kept and a
    warning is printed, because such cross-section duplication is almost always
    a mistake for the keys vdbbench consumes.

    Args:
        config (dict): Configuration dictionary loaded from YAML.

    Returns:
        dict: Flat mapping of parameter name -> value.
    """
    flat = {}

    if not isinstance(config, dict):
        return flat

    for section_name, section in config.items():
        if isinstance(section, dict):
            section_flat = _flatten_section(section, section_name)
        else:
            # Top-level scalar (rare); keep as-is.
            section_flat = {section_name: section}

        for key, value in section_flat.items():
            if key in flat and flat[key] != value:
                # The same leaf name appears in two sections (e.g.
                # dataset.batch_size vs benchmark.batch_size). These denote
                # different things for different consumers, so we must NOT let
                # a later section silently clobber an earlier one. Keep the
                # first occurrence (document order) and warn.
                print(
                    f"WARNING - configuration key '{key}' appears in multiple "
                    f"sections; keeping value {flat[key]!r} from the earlier "
                    f"section and ignoring {value!r} from section "
                    f"'{section_name}'."
                )
                continue
            flat[key] = value

    return flat


def merge_config_with_args(config, args):
    """
    Merge configuration from YAML into parsed command line arguments.

    Precedence (highest first):
        1. Arguments explicitly provided on the command line.
        2. Values from the YAML configuration file.
        3. Argparse defaults.

    This means a value present in the config file is always applied unless
    the user also passed that flag explicitly on the command line, even when
    the config value happens to equal the argparse default.

    Explicit-CLI detection relies on ``args.is_default`` when present (a dict
    of ``{arg_name: True_if_still_default}`` maintained by the callers). When
    that map is unavailable, config values fill only args that are currently
    ``None``.

    Args:
        config (dict): Configuration dictionary from YAML (may be nested).
        args (Namespace): Parsed command line arguments.

    Returns:
        Namespace: Updated arguments with values applied from config.
    """
    if not config:
        return args

    args_dict = vars(args)
    is_default = getattr(args, 'is_default', None)
    flat_config = _flatten_config(config)

    for key, value in flat_config.items():
        if key not in args_dict:
            # Config keys that are not exposed as CLI args (e.g. database
            # connection tuning like max_receive_message_length) are ignored
            # here; callers consume those directly from the config dict.
            continue

        if is_default is not None:
            # Apply config unless the user explicitly set this flag on the CLI.
            # An arg is "explicitly set" when is_default reports False for it.
            cli_explicit = (key in is_default) and (not is_default[key])
            if not cli_explicit:
                args_dict[key] = value
        else:
            # No explicit-vs-default tracking available: only fill holes.
            if args_dict[key] is None:
                args_dict[key] = value

    return args

