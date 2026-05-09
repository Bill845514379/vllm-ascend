#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
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
#

import os
from typing import Any, Callable, Dict

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: Dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER":
    lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER":
    lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION":
    lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH":
    lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH":
    lambda: os.environ.get("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION":
    lambda: os.getenv("VLLM_VERSION", None),
    # Whether to enable the model execute time observe profile. Disable it when
    # running vllm ascend in production environment.
    "VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE", '0'))
                 ),
    # Whether to enable the profiler in NPUModelRunner.
    "VLLM_ASCEND_MODEL_RUNNER_PROFILER_ENABLE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_MODEL_RUNNER_PROFILER_ENABLE",
                               '1'))),
    "VLLM_ASCEND_MODEL_RUNNER_PROFILER_WAIT":
    lambda: int(os.getenv("VLLM_ASCEND_MODEL_RUNNER_PROFILER_WAIT", 2)),
    "VLLM_ASCEND_MODEL_RUNNER_PROFILER_WARMUP":
    lambda: int(os.getenv("VLLM_ASCEND_MODEL_RUNNER_PROFILER_WARMUP", 1)),
    "VLLM_ASCEND_MODEL_RUNNER_PROFILER_ACTIVE":
    lambda: int(os.getenv("VLLM_ASCEND_MODEL_RUNNER_PROFILER_ACTIVE", 10)),
    "VLLM_ASCEND_MODEL_RUNNER_PROFILER_REPEAT":
    lambda: int(os.getenv("VLLM_ASCEND_MODEL_RUNNER_PROFILER_REPEAT", 1)),
    "VLLM_ASCEND_MODEL_RUNNER_PROFILER_SKIP_FIRST":
    lambda: int(
        os.getenv("VLLM_ASCEND_MODEL_RUNNER_PROFILER_SKIP_FIRST", 1500)),
    "VLLM_ASCEND_MODEL_RUNNER_PROFILER_DIR":
    lambda: os.getenv("VLLM_ASCEND_MODEL_RUNNER_PROFILER_DIR") or os.getenv(
        "VLLM_TORCH_PROFILER_DIR") or "/tmp/profile/attn",
    # Whether to enable the profiler in NPUFFNModelRunner.
    "VLLM_ASCEND_FFN_PROFILER_ENABLE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_FFN_PROFILER_ENABLE", '1'))),
    "VLLM_ASCEND_FFN_PROFILER_WAIT":
    lambda: int(os.getenv("VLLM_ASCEND_FFN_PROFILER_WAIT", 2)),
    "VLLM_ASCEND_FFN_PROFILER_WARMUP":
    lambda: int(os.getenv("VLLM_ASCEND_FFN_PROFILER_WARMUP", 1)),
    "VLLM_ASCEND_FFN_PROFILER_ACTIVE":
    lambda: int(os.getenv("VLLM_ASCEND_FFN_PROFILER_ACTIVE", 20)),
    "VLLM_ASCEND_FFN_PROFILER_REPEAT":
    lambda: int(os.getenv("VLLM_ASCEND_FFN_PROFILER_REPEAT", 1)),
    "VLLM_ASCEND_FFN_PROFILER_SKIP_FIRST":
    lambda: int(os.getenv("VLLM_ASCEND_FFN_PROFILER_SKIP_FIRST", 1500)),
    "VLLM_ASCEND_FFN_PROFILER_DIR":
    lambda: os.getenv("VLLM_ASCEND_FFN_PROFILER_DIR") or os.getenv(
        "VLLM_TORCH_PROFILER_DIR") or "/tmp/profile/ffn",
    # Some models are optimized by vllm ascend. While in some case, e.g. rlhf
    # training, the optimized model may not be suitable. In this case, set this
    # value to False to disable the optimized model.
    "USE_OPTIMIZED_MODEL":
    lambda: bool(int(os.getenv('USE_OPTIMIZED_MODEL', '1'))),
    # Whether to enable MatmulAllReduce fusion kernel when tensor parallel is enabled.
    # this feature is supported in A2, and eager mode will get better performance.
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", '0'))),
    # Whether to enable FlashComm optimization when tensor parallel is enabled.
    # This feature will get better performance when concurrency is large.
    "VLLM_ASCEND_ENABLE_FLASHCOMM1":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", '0'))),
    # Whether to enable FLASHCOMM2. Setting it to 0 disables the feature, while setting it to 1 or above enables it.
    # The specific value set will be used as the O-matrix TP group size for flashcomm2.
    # For a detailed introduction to the parameters and the differences and applicable scenarios
    # between this feature and FLASHCOMM1, please refer to the feature guide in the documentation.
    "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE":
    lambda: int(os.getenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", 0)),
    # Whether to enable MLP weight prefetch, only used in small concurrency.
    "VLLM_ASCEND_ENABLE_PREFETCH_MLP":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_PREFETCH_MLP", '0'))),
    # buffer size for gate up prefetch
    "VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE":
    lambda: int(
        os.getenv("VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE", 18 * 1024 * 1024)),
    # buffer size for down proj prefetch
    "VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE":
    lambda: int(
        os.getenv("VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE", 18 * 1024 * 1024)),
    # Whether to enable msMonitor tool to monitor the performance of vllm-ascend.
    "MSMONITOR_USE_DAEMON":
    lambda: bool(int(os.getenv("MSMONITOR_USE_DAEMON", '0'))),
    "VLLM_ASCEND_ENABLE_MLAPO":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLAPO", '0'))),
    # Whether to enable weight cast format to FRACTAL_NZ.
    # 0: close nz;
    # 1: only quant case enable nz;
    # 2: enable nz as long as possible.
    "VLLM_ASCEND_ENABLE_NZ":
    lambda: int(os.getenv("VLLM_ASCEND_ENABLE_NZ", 1)),
    # Decide whether we should enable CP parallelism.
    "VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL", '0'))),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB":
    lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Whether to enable fused mc2(`dispatch_gmm_combine_decode`/`dispatch_ffn_combine` operator)
    # 0, or not set: default ALLTOALL and MC2 will be used.
    # 1: ALLTOALL and MC2 might be replaced by `dispatch_ffn_combine` operator.
    # `dispatch_ffn_combine` can be used only for moe layer with W8A8, EP<=32, non-mtp, non-dynamic-eplb.
    # 2: MC2 might be replaced by `dispatch_gmm_combine_decode` operator.
    # `dispatch_gmm_combine_decode` can be used only for **decode node** moe layer
    # with W8A8. And MTP layer must be W8A8.
    "VLLM_ASCEND_ENABLE_FUSED_MC2":
    lambda: int(os.getenv("VLLM_ASCEND_ENABLE_FUSED_MC2", '0')),
    # Whether to anbale balance scheduling
    "VLLM_ASCEND_BALANCE_SCHEDULING":
    lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", '0'))),
    # AFD FFN diagnostics: log enforce_eager / cudagraph flags once at runner init.
    "VLLM_ASCEND_FFN_DIAG_LOG_INIT":
    lambda: bool(int(os.getenv("VLLM_ASCEND_FFN_DIAG_LOG_INIT", '0'))),
    # AFD FFN diagnostics: torch.npu.synchronize() after each full layer (all ubatches)
    # to localize NPU failures that first appear at the worker sync point.
    "VLLM_ASCEND_FFN_DIAG_SYNC_PER_LAYER":
    lambda: bool(int(os.getenv("VLLM_ASCEND_FFN_DIAG_SYNC_PER_LAYER", '0'))),
    # Verbose trace for ACL graph + MTP issues: metadata keys, replay vs fallback,
    # per-layer recv/compute shapes (default off).
    "VLLM_ASCEND_FFN_GRAPH_MTP_TRACE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_FFN_GRAPH_MTP_TRACE", '0'))),
    # Semicolon-separated repr(dp_key) strings; matching keys skip NPUGraph replay and
    # run eager _ffn_forward (see NPUFFNModelRunner._get_dp_metadata_key). Example for
    # single-stage 96 tokens: ((0, (96, 96)),)
    "VLLM_ASCEND_FFN_GRAPH_REPLAY_DENYLIST":
    lambda: frozenset(
        x.strip() for x in os.getenv(
            "VLLM_ASCEND_FFN_GRAPH_REPLAY_DENYLIST", "").split(";") if x.strip()),
    # If true, torch.npu.synchronize() before graph.replay() to isolate stream/backlog
    # issues (default off).
    "VLLM_ASCEND_FFN_GRAPH_REPLAY_PRE_SYNC":
    lambda: bool(int(os.getenv("VLLM_ASCEND_FFN_GRAPH_REPLAY_PRE_SYNC", '0'))),
    # After CAM recv, before MoE: log topk_ids min/max (CPU) and mask stats (no
    # torch.npu.synchronize — that can trigger 107027 with NPUGraph). Default off.
    "VLLM_ASCEND_FFN_POST_RECV_SYNC_DIAG":
    lambda: bool(int(os.getenv("VLLM_ASCEND_FFN_POST_RECV_SYNC_DIAG", '0'))),
    # AFD camp2p: log Attention send tensors and a2e outputs incl. x_active_mask.
    # Recv path logs [AFD-A2E-RECV-HOST] first (shapes/metadata only, no D2H) so traces stay
    # readable after aicore faults (507015). Optional [AFD-A2E-RECV] adds minmax/mask stats via D2H.
    # No torch.npu.synchronize in these helpers. Default off.
    "VLLM_ASCEND_AFD_WIRE_DIAG":
    lambda: bool(int(os.getenv("VLLM_ASCEND_AFD_WIRE_DIAG", '0'))),
    # When WIRE_DIAG is on: allow NPU→CPU minmax/mask reads for [AFD-A2E-RECV]. Set 0 to keep only
    # [AFD-A2E-RECV-HOST], or when NPUGraph capture still errors (107027) if is_current_stream_capturing
    # is missing on your CANN build. Default on.
    "VLLM_ASCEND_AFD_WIRE_DIAG_D2H":
    lambda: bool(int(os.getenv("VLLM_ASCEND_AFD_WIRE_DIAG_D2H", '1'))),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
