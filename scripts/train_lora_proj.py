"""LoRA + lightweight projector fine-tuning entry point (Method B).

Same training pipeline as ``train_mydata.py`` (pair-margin loss + mydata
triplet sampling). The difference is purely in the YAML config:

    model.lora:                       # injects LoRALinear into attn.qkv + attn.proj
        rank: <int>
        blocks: [<int>, ...]
    model.projector-type: linear      # one Linear(in→out) + L2-normalise per layer
                                        (no GELU / hidden / LayerNorm)
    model.proj-out-dim: <int>

Run::

    python scripts/train_lora_proj.py scripts/configs/train_lora_proj.yml
"""

import json
import os
import sys

_SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

import train_mydata as _t


if __name__ == "__main__":
    args = _t.parse_args()
    env = args["environment"]
    _t._dry = bool(env["dry"])
    _t._seed = int(env["seed"])
    _t._verbose = bool(env["verbose"])
    _t._device = env["device"]
    _t._gpu_ids = env.get("gpu-ids")
    if _t._gpu_ids is None:
        _t._gpu_ids = [0] if str(_t._device).startswith("cuda") else []
    if str(_t._device).startswith("cuda") and len(_t._gpu_ids) == 1:
        _t._device = f"cuda:{_t._gpu_ids[0]}"

    output_path = os.path.abspath(args["wandb"]["output-path"])
    args["wandb"]["output-path"] = output_path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)
    with open(os.path.join(output_path, "args.json"), "w") as fd:
        json.dump(args, fd, indent=4)

    _t.main(args)
