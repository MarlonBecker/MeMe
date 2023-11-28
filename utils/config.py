import argparse as ar
from typing import Any, Dict
import yaml
import os
import shutil

def parseArguments(args, data) -> None:
    """ 
    parse additional arguments from command line which might overwrite arguments defined in config files. unknown arguments will cause an error.
    """
    i = 0
    while i < len(args):
        arg = args[i]
        next_arg = args[i+1] if i != len(args)-1 else None
        if not arg.startswith("--"):
            raise RuntimeError("Invalid command line argument list. Arguments should start with '--'.")
        arg = arg[2:]
        #if argument has no value it is assumed to be bolean and is set to True
        if next_arg is None or next_arg.startswith("--"):
            val = "true" # string to stay consistent with yaml style in input files
            i += 1
        else:
            val = next_arg
            i += 2

        # parse value as yaml string
        parsed_value = yaml.safe_load(val)

        # parse key and set value
        key_error_found = False
        key_list = arg.split("/")
        try:
            data_ = data
            for key in key_list[:-1]:
                data_ = data_[key]
        except KeyError:
            key_error_found = True
        if key_list[-1] in data_:
            data_[key_list[-1]] = parsed_value
        else:
            key_error_found = True
        if key_error_found:
            raise RuntimeError(f"No valid entry found for key '{arg}'. Note: Command line parsing can only be used to overwrite arguments which were defined in config files before. Syntax should be 'keys/in/config'.")

def get_config() -> dict:
    """
    used to parse command line arguments as well as to read and store config files.
    config files should be .yaml files.
    command line arguments are overwriting definitions from config files. syntax: --path/in/yaml value. where value is supposed to be in yaml format too.
    access arguments via Config["foo"]["bar"] or get full config as dictionary from Config.getFullConfig()
    """

    #only read input file name(s)
    fileNameParser = ar.ArgumentParser(description='Framework for reinforcement learning in inverse nanophotonic design.',add_help=False)
    fileNameParser.add_argument("--device_config", help="optional additional config file (yaml) (e.g. for device parameters)", metavar="FILE", default = None)
    fileNameParser.add_argument("--config", help="main config file (yaml). all parameters can be defined here (inc device parameters)", metavar="FILE", required = True)
    args, remaining_argv = fileNameParser.parse_known_args()

    # read config/device_config files
    print("Loading config from:", args.config)
    with open(args.config, 'r') as stream:
        data = yaml.load(stream, Loader=yaml.Loader)
    device_config = {} 
    if args.device_config is not None:
        print("Loading device_config from:", args.device_config)
        with open(args.device_config, 'r') as stream:
            device_config = yaml.load(stream, Loader=yaml.Loader)

    # unite config files and check for duplications
    for key, value in device_config.items():
        if key in data:
            # if both values are lists: unite
            if isinstance(data[key], list) and isinstance(value, list):
                data[key] += value
            else:
                raise RuntimeError(f"Key '{key}' defined in both input files and cant be united (at least one is not a list).")
        else:
            data[key] = value

    # parse command line arguments
    parseArguments(remaining_argv, data)

    ## create logDir 
    if os.path.isdir(data["logDir"]):
        mode = data["existingFileMode"]
        if mode == "error":
            raise RuntimeError(f"Folder '{data['logDir']}' exists. Use '--existingFileMode truncate' or '--existingFileMode continue'.")
        elif mode == "truncate":
            shutil.rmtree(data["logDir"])
        elif mode == "continue":
            if not os.path.isdir(os.path.join(data["logDir"], "checkpoint")):
                raise RuntimeError(f"'existingFileMode == continue' but no folder named 'checkpoint' found in data dir. To save checkpoints set 'checkpointInterval'.")
        else:
            raise RuntimeError(f"'existingFileMode' has to be one of truncate/error/continue but is '{mode}'.")
    if not os.path.isdir(data["logDir"]):
        # store copy of full config in log dir (including parameters updated via command line)
        os.makedirs(data['logDir'], exist_ok=False)
        with open(os.path.join(data["logDir"], "config.yaml"), "x") as file:
            yaml.dump(data, file)

    return data
