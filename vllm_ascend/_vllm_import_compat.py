#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import types

# Keep in sync with vLLM 0.22 ``vllm/__init__.py`` MODULE_ATTRS.
_VLLM_MODULE_ATTRS: dict[str, str] = {
    "AsyncEngineArgs": ".engine.arg_utils:AsyncEngineArgs",
    "EngineArgs": ".engine.arg_utils:EngineArgs",
    "AsyncLLMEngine": ".engine.async_llm_engine:AsyncLLMEngine",
    "LLMEngine": ".engine.llm_engine:LLMEngine",
    "LLM": ".entrypoints.llm:LLM",
    "initialize_ray_cluster": ".v1.executor.ray_utils:initialize_ray_cluster",
    "PromptType": ".inputs:PromptType",
    "TextPrompt": ".inputs:TextPrompt",
    "TokensPrompt": ".inputs:TokensPrompt",
    "ModelRegistry": ".model_executor.models:ModelRegistry",
    "SamplingParams": ".sampling_params:SamplingParams",
    "PoolingParams": ".pooling_params:PoolingParams",
    "ClassificationOutput": ".outputs:ClassificationOutput",
    "ClassificationRequestOutput": ".outputs:ClassificationRequestOutput",
    "CompletionOutput": ".outputs:CompletionOutput",
    "EmbeddingOutput": ".outputs:EmbeddingOutput",
    "EmbeddingRequestOutput": ".outputs:EmbeddingRequestOutput",
    "PoolingOutput": ".outputs:PoolingOutput",
    "PoolingRequestOutput": ".outputs:PoolingRequestOutput",
    "RequestOutput": ".outputs:RequestOutput",
    "ScoringOutput": ".outputs:ScoringOutput",
    "ScoringRequestOutput": ".outputs:ScoringRequestOutput",
}

# Safe to resolve while ``engine.arg_utils`` is still importing.
_EARLY_EXPORTS = frozenset(
    {
        "PromptType",
        "TextPrompt",
        "TokensPrompt",
        "SamplingParams",
        "PoolingParams",
        "ClassificationOutput",
        "ClassificationRequestOutput",
        "CompletionOutput",
        "EmbeddingOutput",
        "EmbeddingRequestOutput",
        "PoolingOutput",
        "PoolingRequestOutput",
        "RequestOutput",
        "ScoringOutput",
        "ScoringRequestOutput",
        "ModelRegistry",
        "LLM",
        "initialize_ray_cluster",
    }
)

_LOGITS_PROCESSOR_MODULES = frozenset(
    {
        "vllm.v1.sample.logits_processor.interface",
        "vllm.v1.sample.logits_processor.builtin",
    }
)

_LAZY_EXPORTS_INSTALLED = False
_HOOK_INSTALLED = False


def _resolve_vllm_export(
    vllm_pkg: types.ModuleType,
    name: str,
    module_attrs: dict[str, str],
) -> object:
    spec = module_attrs.get(name)
    if spec is None:
        raise AttributeError(f"module {vllm_pkg.__name__!r} has no attribute {name!r}")
    module_name, attr_name = spec.split(":")
    module = importlib.import_module(module_name, vllm_pkg.__name__)
    return getattr(module, attr_name)


def _ensure_vllm_lazy_exports() -> None:
    """Install a resilient ``__getattr__`` when vLLM lazy exports break mid-import."""
    global _LAZY_EXPORTS_INSTALLED
    if _LAZY_EXPORTS_INSTALLED:
        return

    vllm_pkg = sys.modules.get("vllm")
    if vllm_pkg is None:
        return

    module_attrs = getattr(vllm_pkg, "MODULE_ATTRS", None) or _VLLM_MODULE_ATTRS
    original_getattr = getattr(vllm_pkg, "__getattr__", None)

    def __getattr__(name: str) -> object:
        if name in vllm_pkg.__dict__:
            return vllm_pkg.__dict__[name]

        if name in ("__version__", "__version_tuple__"):
            version_mod = importlib.import_module("vllm.version")
            obj = getattr(version_mod, name)
            setattr(vllm_pkg, name, obj)
            return obj

        if name in module_attrs:
            obj = _resolve_vllm_export(vllm_pkg, name, module_attrs)
            setattr(vllm_pkg, name, obj)
            return obj

        if original_getattr is not None and original_getattr is not __getattr__:
            return original_getattr(name)

        raise AttributeError(f"module {vllm_pkg.__name__!r} has no attribute {name!r}")

    vllm_pkg.__getattr__ = __getattr__
    _LAZY_EXPORTS_INSTALLED = True

    arg_utils = sys.modules.get("vllm.engine.arg_utils")
    arg_utils_loading = arg_utils is not None and not hasattr(arg_utils, "EngineArgs")
    exports = _EARLY_EXPORTS if arg_utils_loading else module_attrs.keys()
    for name in exports:
        if name in vllm_pkg.__dict__:
            continue
        try:
            setattr(vllm_pkg, name, _resolve_vllm_export(vllm_pkg, name, module_attrs))
        except Exception:
            # Defer exports that are not safe during a partial import.
            continue


class _VllmLogitsProcessorImportHook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _LOGITS_PROCESSOR_MODULES:
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            return None
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _VllmLogitsProcessorSourceLoader(fullname, spec.origin),
            origin=spec.origin,
        )


class _VllmLogitsProcessorSourceLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, origin: str) -> None:
        self.fullname = fullname
        self.origin = origin

    def create_module(self, spec):
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        with open(self.origin, encoding="utf-8") as f:
            source = f.read()
        patched_source = source.replace(
            "from vllm import SamplingParams",
            "from vllm.sampling_params import SamplingParams",
        )
        module.__file__ = self.origin
        module.__package__ = self.fullname.rpartition(".")[0]
        exec(compile(patched_source, self.origin, "exec"), module.__dict__)


def install_vllm_logits_processor_import_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    sys.meta_path.insert(0, _VllmLogitsProcessorImportHook())
    _HOOK_INSTALLED = True


def apply_vllm_import_compat() -> None:
    install_vllm_logits_processor_import_hook()
    _ensure_vllm_lazy_exports()
