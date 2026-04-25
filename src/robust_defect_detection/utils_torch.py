import collections
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def check_module(x):
    if not isinstance(x, nn.Module):
        raise ValueError("Only accept nn.Module input.")


def seed_everything(seed=42, verbose=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if verbose:
        print(f"Seed set to: {seed}")


def freeze_model(model):
    check_module(model)
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    check_module(model)
    for param in model.parameters():
        param.requires_grad = True


def is_all_frozen(model):
    check_module(model)
    return all(not param.requires_grad for param in model.parameters())


def is_any_frozen(model):
    check_module(model)
    return any(not param.requires_grad for param in model.parameters())


def get_grad_required_state(model):
    check_module(model)
    state = collections.OrderedDict()

    def write(module_state, prefix):
        for key, value in module_state.items():
            state[prefix + key] = value

    def dfs(module, prefix):
        if is_all_frozen(module):
            return
        if not is_any_frozen(module):
            write(module.state_dict(), prefix)
            return
        for name, child in module._modules.items():
            dfs(child, prefix + name + ".")

    dfs(model, "")
    return state


def load_grad_required_state(model, state, verbose=True, return_details=False):
    check_module(model)
    state = state.copy()

    def xprint(msg):
        if verbose:
            print(msg)

    model_keys = list(model.state_dict().keys())
    state_keys = list(state.keys())
    if model_keys and state_keys:
        model_has_module_prefix = all(key.startswith("module.") for key in model_keys)
        state_has_module_prefix = all(key.startswith("module.") for key in state_keys)

        if state_has_module_prefix and not model_has_module_prefix:
            state = collections.OrderedDict(
                (key[len("module."):], value) for key, value in state.items()
            )
        elif model_has_module_prefix and not state_has_module_prefix:
            state = collections.OrderedDict(
                (f"module.{key}", value) for key, value in state.items()
            )

    def write(module, prefix):
        names = [key for key in state.keys() if key.startswith(prefix)]
        trimmed = collections.OrderedDict()
        prefix_len = len(prefix)
        for name in names:
            trimmed[name[prefix_len:]] = state.pop(name)
        module.load_state_dict(trimmed, strict=True)

    def dfs(module, prefix):
        if is_all_frozen(module):
            return
        if not is_any_frozen(module):
            write(module, prefix)
            return
        for name, child in module._modules.items():
            dfs(child, prefix + name + ".")

    dfs(model, "")

    if state:
        for name in state.keys():
            xprint(f"<{name} do not match in model>")
    elif verbose:
        xprint("<All keys matched successfully>")

    if return_details:
        return model, state
    return model


class CustomizedLRScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        start_scale=0.0,
        warmup_epoch=-1,
        final_scale=0.0,
        total_epoch=-1,
        mode=None,
    ):
        if last_epoch > total_epoch:
            raise ValueError
        if start_scale > 1:
            raise ValueError
        if mode not in ["cosine", "linear", "exp", None]:
            raise ValueError
        self.mode = mode
        self.start_scale = start_scale
        self.final_scale = final_scale
        self.warmup_epoch = warmup_epoch
        self.total_epoch = total_epoch
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_scale(self, epoch):
        if epoch < 0 or self.mode is None:
            return 1.0
        if self.mode == "exp":
            gamma = math.log(self.final_scale + 1e-7) / self.total_epoch
            return math.exp(gamma * epoch)
        if self.mode == "linear":
            return 1 - (1 - self.final_scale) / self.total_epoch * epoch
        x = epoch / self.total_epoch * math.pi
        x = math.cos(x) + 1
        x = x / 2 * (1 - self.final_scale)
        return x + self.final_scale

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch:
            target_scale = self.get_scale(self.warmup_epoch)
            ratio = 1.0 * self.last_epoch / self.warmup_epoch
            scale = self.start_scale + (target_scale - self.start_scale) * ratio
        else:
            scale = self.get_scale(self.last_epoch)
        return [base_lr * scale for base_lr in self.base_lrs]
