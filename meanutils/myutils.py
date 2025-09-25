import torch
import numpy as np
import yaml
import os


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def read_scp(scp_file_path, only_get_last_value=False):
    assert os.path.exists(scp_file_path) and os.path.isfile(scp_file_path)
    scp_dict = {}
    keys = []
    with open(scp_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"invalid line: {line}")
            
            key = parts[0]
            if only_get_last_value:
                value = parts[-1]
            else:
                value = ' '.join(parts[1:])
            scp_dict[key] = value
            keys.append(key)
    
    return scp_dict, keys


def write_scp(scp_file_path, key, value, start=False):
    if start:
        with open(scp_file_path, "w") as f:
            f.write(f"{key} {value}\n")
    else:
        with open(scp_file_path, "a") as f:
            f.write(f"{key} {value}\n")