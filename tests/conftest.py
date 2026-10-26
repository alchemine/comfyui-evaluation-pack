"""The ComfyUI modules the pack imports are stubbed so it imports outside
ComfyUI, and `rtmlib` so the tests need neither it nor onnxruntime; nothing
else is faked.
"""

import sys
import types
import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

PACK_DIR = Path(__file__).resolve().parent.parent
PACK_NAME = "evaluation_pack"


@pytest.fixture(scope="session")
def comfy_stubs():
    model_management = types.ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cpu")
    comfy = types.ModuleType("comfy")
    comfy.model_management = model_management
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = ""
    folder_paths.get_folder_paths = lambda kind: []
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management
    sys.modules["folder_paths"] = folder_paths
    rtmlib = types.ModuleType("rtmlib")
    rtmlib.Wholebody = None
    sys.modules["rtmlib"] = rtmlib


@pytest.fixture(scope="session")
def pack(comfy_stubs):
    """Imports a node module by name, the way ComfyUI would."""
    package = types.ModuleType(PACK_NAME)
    package.__path__ = [str(PACK_DIR)]
    sys.modules.setdefault(PACK_NAME, package)
    return lambda name: importlib.import_module(f"{PACK_NAME}.nodes.{name}")


@pytest.fixture(scope="session")
def mappings(comfy_stubs):
    """The pack's `__init__.py`, loaded the way ComfyUI loads a custom node."""
    spec = importlib.util.spec_from_file_location(
        PACK_NAME, PACK_DIR / "__init__.py", submodule_search_locations=[str(PACK_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACK_NAME] = module
    spec.loader.exec_module(module)
    return module.NODE_CLASS_MAPPINGS
