import dis
from collections import defaultdict

import importlib
import pathlib
import os
import shutil

def _backup_source_files_to(entrypoint:str, target_dir:str):
    
    entrypoint = os.path.realpath(entrypoint)
    
    # parse entrypoint script for imported files
    with open(entrypoint, "r") as f:
        statements = f.read()

    instructions = dis.get_instructions(statements)
    imports = [__ for __ in instructions if 'IMPORT' in __.opname]

    grouped = defaultdict(list)
    for instr in imports:
        grouped[instr.opname].append(instr.argval)

    pkgs = grouped["IMPORT_NAME"]
    
    dirname = os.path.dirname(entrypoint)

    pkgpaths = [importlib.util.find_spec(pkg).origin for pkg in pkgs]
    selected_files = set()
    selected_files.add(entrypoint)
    for path in pkgpaths:
        try:
            path = pathlib.Path(path).relative_to(dirname)
            while True:
                parent = path.parent
                if str(parent) == '.':
                    selected_files.add(str(path))
                    break
                path = parent             
        except ValueError: pass

    file_or_folder:str
    for file_or_folder in selected_files:
        if file_or_folder.endswith(".py"):
            shutil.copy2(file_or_folder, target_dir)
        else:
            shutil.copytree(file_or_folder, os.path.join(target_dir, file_or_folder))