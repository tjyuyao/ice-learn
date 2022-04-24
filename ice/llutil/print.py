import abc
import sys
import types
from argparse import Namespace

import numpy as np
from ice.llutil.collections import ConfigDict
from tqdm import tqdm
from varname import argname
from collections.abc import Mapping


default_printoptions_set = False


def set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    sci_mode=None,
):
    import torch

    torch.set_printoptions(
        precision=precision,
        threshold=threshold,
        edgeitems=edgeitems,
        linewidth=linewidth,
        sci_mode=sci_mode,
    )
    np.set_printoptions(
        precision=precision,
        threshold=threshold,
        edgeitems=edgeitems,
        linewidth=linewidth,
        suppress=not sci_mode,
    )


def _brief_ndarray(data, threshold: int):
    if isinstance(data, np.ndarray) and data.size > threshold:
        nanstr = ", hasnan=True" if np.isnan(data).any() else ""
        brief = ", ".join([f"{x:.4g}" for x in data.flatten()[:threshold].tolist()])
        brief = f"[{brief}, ... ]"
        brief = f"array(shape={data.shape}, dtype={data.dtype}, min={np.min(data):.4g}, max={np.max(data):4g}, data={brief}{nanstr})"
    else:
        brief = data
    return brief


def _brief_tensor(data, threshold: int):
    import torch

    if isinstance(data, torch.Tensor) and data.numel() > threshold:
        nanstr = ", hasnan=True" if data.isnan().any() else ""
        brief = ", ".join([f"{x:.4g}" for x in data.flatten()[:threshold].tolist()])
        brief = f"[{brief}, ... ]"
        dtype = f"{data.dtype}"
        dtype = dtype[6:] if dtype.startswith("torch.") else dtype
        brief = f'tensor(shape={tuple(data.shape)}, dtype={dtype}, min={torch.min(data):.4g}, max={torch.max(data):.4g}, data={brief}{nanstr}, device="{data.device}")'
    else:
        brief = data
    return brief


def format_size(size):
    """Format a byte count as a human readable file size.

    >>> format_size(0)
    '0 bytes'
    >>> format_size(1)
    '1 byte'
    >>> format_size(5)
    '5 bytes'
    >>> format_size(1024)
    '1 KiB'
    """
    size_sign = "" if size > 0 else "-"
    size = abs(size)
    size_units = [(1024, "KiB"), (1024**2, "MiB"), (1024**3, "GiB")]
    for divider, unit in reversed(size_units):
        if size >= divider:
            size = float(size) / divider
            break
    else:
        unit = "byte" if size == 1 else "bytes"
    return f"{size_sign}{size:.2f} {unit}"


def _print(
    data,
    prefix="",
    uri=None,
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    sci_mode=None,
):
    import torch

    global default_printoptions_set
    if not default_printoptions_set:
        set_printoptions(threshold=4, linewidth=120, precision=2, sci_mode=False)

    if threshold or threshold or edgeitems or linewidth or sci_mode:
        set_printoptions(
            precision=precision,
            threshold=threshold,
            edgeitems=edgeitems,
            linewidth=linewidth,
            sci_mode=sci_mode,
        )

    threshold = threshold or torch._tensor_str.PRINT_OPTS.threshold

    if uri is None:
        try:
            uri = argname("data")
        except:
            uri = ""

    if isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            _print(data[i], prefix=prefix, uri=f"{uri}[{i}]")
        return
    elif isinstance(data, ConfigDict):
        for k, v in data.items():
            _print(v, prefix=prefix, uri=f"{uri}.{k}")
        return
    elif isinstance(data, Mapping):
        for k, v in data.items():
            if (
                k in ["In"]
                or isinstance(v, (types.ModuleType, types.FunctionType))
                or (
                    isinstance(k, str)
                    and k.startswith("_")
                    and not isinstance(v, (torch.Tensor, np.ndarray, float))
                )
                or str(type(v)).lower().find("ipython") != -1
            ):
                continue
            if k in ["duration_change", "duratioin_peak"]:
                v = format_size(v)
            _print(v, prefix=prefix, uri=f"{uri}[{repr(k)}]")
        return
    elif isinstance(data, np.ndarray):
        brief = _brief_ndarray(data, threshold=threshold)
    elif isinstance(data, torch.Tensor):
        brief = _brief_tensor(data, threshold=threshold)
    elif isinstance(data, Namespace):
        for k, v in vars(data).items():
            _print(v, prefix=prefix, uri=f"{uri}.{k}")
        return
    else:
        brief = data
    if uri != "" and not uri.endswith(": "):
        uri = uri + ": "
    tqdm.write(f"{prefix}{uri}{brief}\n", end="")
