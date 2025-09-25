import torch
import fire
import numpy as np
from tqdm import tqdm


def print_checkpoint(checkpoint_path, show_key=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        print(checkpoint.keys())
        state_dict = checkpoint['state_dict']
    else:
        assert isinstance(checkpoint, dict)
        state_dict = checkpoint
    
    print("state_dict:")
    for key in state_dict.keys():
        print(key, state_dict[key].shape, state_dict[key].dtype, state_dict[key].is_complex())
        if show_key is not None and show_key in key:
            print(key, state_dict[key])

    if "ema_state_dict" in checkpoint:
        ema_state_dict = checkpoint['ema_state_dict']
        print("ema_state_dict:")
        for key in ema_state_dict.keys():
            print(key, ema_state_dict[key].shape, ema_state_dict[key].dtype, ema_state_dict[key].is_complex())


def detect_nan_and_inf(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        print(checkpoint.keys())
        state_dict = checkpoint['state_dict']
    else:
        assert isinstance(checkpoint, dict)
        state_dict = checkpoint

    detected = False

    for key, value in state_dict.items():
        if torch.isnan(value).any():
            print(f"NaN detected in {key}")
            detected = True
        if torch.isinf(value).any():
            print(f"Inf detected in {key}")
            detected = True

    if "ema_state_dict" in checkpoint:
        ema_state_dict = checkpoint['ema_state_dict']
        print("ema_state_dict:")
        for key, value in ema_state_dict.items():
            if torch.isnan(value).any():
                print(f"NaN detected in {key}")
                detected = True
            if torch.isinf(value).any():
                print(f"Inf detected in {key}")
                detected = True
    
    if not detected:
        print("No NaN or Inf detected in the checkpoint!")


def compare_state_dicts(state_dict1, state_dict2, visible=False):
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1

    if only_in_1:
        if visible:
            print("Keys only in first state dict:")
            for k in only_in_1:
                print(k, state_dict1[k].shape, state_dict1[k].dtype, state_dict1[k].is_complex())
    if only_in_2:
        if visible:
            print("Keys only in second state dict:")
            for k in only_in_2:
                print(k, state_dict2[k].shape, state_dict2[k].dtype, state_dict2[k].is_complex())

    common_keys = keys1 & keys2

    compare_result = True
    diff_shape_cnt = 0
    not_allclose_cnt = 0
    allclose_cnt = 0

    for key in common_keys:
        t1 = state_dict1[key]
        t2 = state_dict2[key]
        if t1.shape != t2.shape or t1.dtype != t2.dtype:
            diff_shape_cnt += 1
            compare_result = False
            if visible:
                print(f"Key {key} with different shape / type: {t1.shape}/{t1.dtype} vs {t2.shape}/{t2.dtype}")  
        else:
            allclose = torch.allclose(t1, t2)
            not_allclose_cnt += (0 if allclose else 1)
            allclose_cnt += (1 if allclose else 0)
            compare_result = compare_result and allclose
            if visible:
                print(f"Key {key} is allclose: {allclose}")

    print(f"Keys only in first: {len(only_in_1)}", flush=True)
    print(f"Keys only in second: {len(only_in_2)}", flush=True)
    print(f"Common keys with different shape/type: {diff_shape_cnt}", flush=True)
    print(f"Common keys not allclose: {not_allclose_cnt}", flush=True)
    print(f"Common keys allclose: {allclose_cnt}", flush=True)
    return compare_result


def compare_checkpoints(ckpt_path1, ckpt_path2):
    state_dict1 = load_state_dict_from_ckpt(ckpt_path1)
    state_dict2 = load_state_dict_from_ckpt(ckpt_path2)
    compare_result = compare_state_dicts(state_dict1, state_dict2, visible=False)
    return compare_result


def change_prefix(state_dict, old_prefix, new_prefix):
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace(old_prefix, new_prefix)
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def replace_with_ema_state_dict(state_dict, ema_state_dict):
    for key in ema_state_dict.keys():
        if key in state_dict:
            state_dict[key] = ema_state_dict[key]
    return state_dict


def save_state_dict(state_dict, save_path):
    torch.save(state_dict, save_path)
    print(f"State dict saved to {save_path}")


def load_state_dict_from_ckpt(ckpt_path, device='cpu'):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    else:
        assert isinstance(checkpoint, dict)
        return checkpoint

if __name__ == "__main__":
    fire.Fire(print_checkpoint)