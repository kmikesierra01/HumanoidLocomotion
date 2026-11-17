"""Export an rsl_rl checkpoint (model_<iter>.pt) to ONNX without Isaac, same graph as Isaac Lab's exporter
(actor MLP + ELU, no obs normalizer, input "obs" batch 1, output "actions", opset 11).
play.py would overwrite the run's exported/policy.onnx; this writes wherever you say.

Usage: python export_ckpt.py <model_X.pt> <out.onnx>
"""
import sys

import torch


def actor_from_checkpoint(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    linear = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.")})  # 0, 2, 4, 6 (ELU in between)
    layers = []
    for i in linear:
        w = sd[f"actor.{i}.weight"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.load_state_dict({"weight": w, "bias": sd[f"actor.{i}.bias"]})
        layers += [lin, torch.nn.ELU()]
    return torch.nn.Sequential(*layers[:-1]).eval()


def export(ckpt, out):
    actor = actor_from_checkpoint(ckpt)
    obs = torch.zeros(1, actor[0].in_features)
    torch.onnx.export(actor, obs, out, export_params=True, opset_version=11, input_names=["obs"],
                      output_names=["actions"], dynamic_axes={})


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
