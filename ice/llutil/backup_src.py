import dis
from collections import defaultdict

import importlib
import pathlib
import os
import shutil

_extended_sys_path = []


def _common_root(path_list):
    path_list = [path.split(os.path.sep) for path in path_list]
    common_tokens = []
    for tokens in zip(*path_list):
        for i in range(len(tokens) - 1):
            if tokens[i] != tokens[1+i]: break
        else:
            common_tokens.append(tokens[0])
            continue
        break
    return os.path.sep.join(common_tokens)


def parse_dependency(py_path:str, root_dir):

    if not py_path.endswith(".py"): return []

    # parse entrypoint script for imported files
    with open(py_path, "r") as f:
        statements = f.read()

    instructions = dis.get_instructions(statements)
    imports = [__ for __ in instructions if 'IMPORT' in __.opname]

    grouped = defaultdict(list)
    for instr in imports:
        grouped[instr.opname].append(instr.argval)

    pkgs = grouped["IMPORT_NAME"]

    deps = []
    for pkg in pkgs:
        try:
            dep = importlib.util.find_spec(pkg).origin
            pathlib.Path(dep).relative_to(root_dir)
        except: continue
        if dep not in deps:
            deps.append(dep)
            deps += parse_dependency(dep, root_dir)
    return deps


def _backup_source_files_to(entrypoint:str, extra_backup_paths, target_dir:str):

    entrypoint = os.path.realpath(entrypoint)
    
    root_dir = _common_root(_extended_sys_path + [os.path.dirname(entrypoint)])

    deps = parse_dependency(entrypoint, root_dir)
    deps.append(entrypoint)

    for absdep in deps:
        reldep = str(pathlib.Path(absdep).relative_to(root_dir))
        tardep = os.path.join(target_dir, reldep)
        tardir = os.path.dirname(tardep)
        os.makedirs(tardir, exist_ok=True)
        shutil.copy2(absdep, tardep)

    for ebp in extra_backup_paths:
        try:
            relebp = str(pathlib.Path(ebp).relative_to(root_dir))
            tarebp = os.path.join(target_dir, relebp)
        except: continue

        if os.path.isdir(ebp):
            shutil.copytree(ebp, tarebp)
        elif os.path.isfile(ebp):
            tardir = os.path.dirname(tarebp)
            os.makedirs(tardir, exist_ok=True)
            shutil.copy2(ebp, tarebp)
