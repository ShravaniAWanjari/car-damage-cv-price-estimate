"""Convert an untrusted Ultralytics ``.pt`` checkpoint to safetensors + JSON
*without executing anything from the file*.

Why: ``torch.load`` / ``YOLO("x.pt")`` unpickle arbitrary Python. Community
checkpoints are untrusted input. This script reads the pickle with a strict
allow-list unpickler: tensors, storages and plain containers are rebuilt by
our own code; every other global (nn.Module classes, ultralytics classes,
``getattr`` ...) becomes an inert stub that only records its attributes.
We then read parameters/buffers, ``names`` and the architecture YAML back out
of those stubs.

    python scripts/safe_convert.py in.pt out_dir

Writes out_dir/model.safetensors and out_dir/meta.json
(task, names, arch yaml, imgsz, source sha256).
"""
from __future__ import annotations

import collections
import hashlib
import json
import pickle
import sys
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file

_STORAGE_DTYPES = {
    "FloatStorage": torch.float32, "HalfStorage": torch.float16,
    "DoubleStorage": torch.float64, "LongStorage": torch.int64,
    "IntStorage": torch.int32, "ShortStorage": torch.int16,
    "CharStorage": torch.int8, "ByteStorage": torch.uint8,
    "BoolStorage": torch.bool, "BFloat16Storage": torch.bfloat16,
}


class _StorageType:
    def __init__(self, dtype):
        self.dtype = dtype


class _Storage:
    def __init__(self, dtype, raw):
        self.dtype, self.raw = dtype, raw


class Stub:
    """Inert stand-in for any class we do not explicitly allow."""
    _g = ("?", "?")

    def __init__(self, *args, **kwargs):
        self._args = args
        self._state = {}

    def __setstate__(self, state):
        self._state = state if isinstance(state, dict) else {"_": state}

    def __call__(self, *args, **kwargs):
        return Stub()


def _rebuild_tensor(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    flat = torch.frombuffer(bytearray(storage.raw), dtype=storage.dtype)
    return torch.as_strided(flat, tuple(size), tuple(stride), offset)


def _rebuild_parameter(data, requires_grad=False, hooks=None, *_):
    return data


class _SafeUnpickler(pickle.Unpickler):
    def __init__(self, fh, zf, prefix):
        super().__init__(fh)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(_STORAGE_DTYPES[name])
        if module == "torch" and name == "Size":
            return tuple
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module in ("builtins", "__builtin__") and name in ("set", "frozenset"):
            return set if name == "set" else frozenset
        return type(name, (Stub,), {"_g": (module, name)})  # inert

    def persistent_load(self, pid):
        kind, stype, key, _loc, _numel = pid
        assert kind == "storage"
        raw = self.zf.read(f"{self.prefix}/data/{key}")
        return _Storage(stype.dtype, raw)


def _collect(stub, prefix, out):
    st = stub._state
    for k, v in (st.get("_parameters") or {}).items():
        if isinstance(v, torch.Tensor):
            out[prefix + k] = v
    for k, v in (st.get("_buffers") or {}).items():
        if isinstance(v, torch.Tensor):
            out[prefix + k] = v
    for k, v in (st.get("_modules") or {}).items():
        if isinstance(v, Stub):
            _collect(v, f"{prefix}{k}.", out)


def convert(src: Path, dst: Path) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    with zipfile.ZipFile(src) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl.rsplit("/", 1)[0]
        with zf.open(pkl) as fh:
            ckpt = _SafeUnpickler(fh, zf, prefix).load()
    net = ckpt.get("ema") or ckpt.get("model")
    if not isinstance(net, Stub):
        raise ValueError("no model found in checkpoint")
    sd: dict = {}
    _collect(net, "", sd)
    state = net._state
    heads = {s._g[1] for s in _walk(net)}
    task = "segment" if "Segment" in heads else "pose" if "Pose" in heads else "detect"
    train_args = ckpt.get("train_args") or {}
    meta = {
        "task": task,
        "names": {int(k): v for k, v in dict(state["names"]).items()} if "names" in state else None,
        "yaml": state.get("yaml"),
        "imgsz": train_args.get("imgsz"),
        "source_sha256": sha,
        "source_name": src.name,
        "arch_class": net._g[1],
    }
    # float16 checkpoints are common; run in fp32 for CPU/MPS parity, GPUs can cast back.
    sd = {k: (v.float() if v.is_floating_point() else v).contiguous() for k, v in sd.items()}
    save_file(sd, str(dst / "model.safetensors"))
    (dst / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def _walk(stub):
    yield stub
    for v in (stub._state.get("_modules") or {}).values():
        if isinstance(v, Stub):
            yield from _walk(v)


if __name__ == "__main__":
    m = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps({k: m[k] for k in ("task", "names", "imgsz", "arch_class", "source_sha256")}, default=str))
