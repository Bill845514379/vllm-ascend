# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import gc
from typing import TYPE_CHECKING, Any, Optional
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch_npu
import torch.nn as nn
from tqdm import tqdm
import re
from vllm.config import VllmConfig
from vllm.distributed.afd_transfer.afd_connector.factory import (
    AFDConnectorFactory)
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (get_dp_group, get_ep_group,
                                             get_tensor_model_parallel_rank,
                                             get_tensor_model_parallel_world_size,
                                             get_world_group, is_global_first_rank)
from vllm.forward_context import set_forward_context, BatchDescriptor, AFDMetadata
from vllm.logger import init_logger
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.model_executor.model_loader import get_model_loader
from vllm_ascend.worker.model_runner_v1 import (NPUModelRunner, graph_capture,
                                              _torch_cuda_wrapper,
                                              _replace_gpu_model_runner_function_wrapper)
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.distributed.metadata import (M2NAFDConnectorMetadata, CAMM2NAFDConnectorMetadata, CAMP2PAFDConnectorMetadata)
import vllm_ascend.envs as envs_ascend
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (CompilationMode, CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.v1.worker.gpu_ffn_model_runner import GPUFFNModelRunner
from vllm.platforms import current_platform
import vllm.envs as envs


if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

logger = init_logger(__name__)


def _ffn_graph_mtp_trace_enabled() -> bool:
    return envs_ascend.VLLM_ASCEND_FFN_GRAPH_MTP_TRACE


def _ffn_graph_replay_denied(dp_key: tuple) -> bool:
    denied = envs_ascend.VLLM_ASCEND_FFN_GRAPH_REPLAY_DENYLIST
    if not denied:
        return False
    return repr(dp_key) in denied


def _maybe_warn_moe_topk_oob(
    topk_ids: torch.Tensor | None,
    moe_expert_num: int,
    *,
    where: str,
    layer_idx: int,
    ubatch_idx: int,
) -> None:
    """D2H min/max; warn if expert indices are outside global routed range."""
    if (not envs_ascend.VLLM_ASCEND_FFN_MOE_OOB_WARN
            or topk_ids is None or topk_ids.numel() == 0):
        return
    try:
        ep_rank = get_ep_group().rank_in_group
        ep_world = get_ep_group().world_size
    except Exception:
        ep_rank, ep_world = -1, -1
    try:
        tmin = int(topk_ids.min().detach().cpu().item())
        tmax = int(topk_ids.max().detach().cpu().item())
    except Exception as ex:
        logger.warning(
            "[FFN-MOE-OOB] %s layer=%s ubatch=%s ep=%s/%s cannot read topk_ids: %s",
            where, layer_idx, ubatch_idx, ep_rank, ep_world, ex)
        return
    if tmin < 0 or tmax >= moe_expert_num:
        logger.warning(
            "[FFN-MOE-OOB] %s layer=%s ubatch=%s ep=%s/%s topk_ids in [%s,%s] "
            "not inside [0, %s) (n_routed_experts=%s). Likely corrupt recv / "
            "EP desync before MoE.",
            where, layer_idx, ubatch_idx, ep_rank, ep_world, tmin, tmax,
            moe_expert_num, moe_expert_num)


def _log_ffn_post_recv_moe_sync_diag(
    *,
    logger,
    layer_idx: int,
    num_layers: int,
    ubatch_idx: int,
    dp_key: tuple,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor | None,
    x_active_mask: torch.Tensor | None,
    moe_expert_num: int | None = None,
) -> None:
    try:
        ep_rank = get_ep_group().rank_in_group
        ep_world = get_ep_group().world_size
    except Exception:
        ep_rank, ep_world = -1, -1
    tmin = tmax = None
    if topk_ids is not None and topk_ids.numel() > 0:
        try:
            tmin = int(topk_ids.min().detach().cpu().item())
            tmax = int(topk_ids.max().detach().cpu().item())
        except Exception as ex:
            tmin = tmax = str(ex)
    msum = mdtype = None
    if x_active_mask is not None:
        mdtype = str(x_active_mask.dtype)
        try:
            msum = int(x_active_mask.sum().detach().cpu().item())
        except Exception as ex:
            msum = str(ex)
    tid_ptr = int(topk_ids.data_ptr()) if topk_ids is not None else 0
    logger.info(
        "[FFN-POST-RECV-SYNC-DIAG] layer=%s/%s ubatch=%s ep=%s/%s dp_key=%s "
        "hid=%s topk_ids=%s dtype=%s ptr=0x%x post_sync_minmax=(%s,%s) "
        "mask_sum=%s mask_dtype=%s n_routed=%s",
        layer_idx,
        num_layers - 1,
        ubatch_idx,
        ep_rank,
        ep_world,
        dp_key,
        tuple(hidden_states.shape),
        tuple(topk_ids.shape) if topk_ids is not None else None,
        str(topk_ids.dtype) if topk_ids is not None else None,
        tid_ptr,
        tmin,
        tmax,
        msum,
        mdtype,
        moe_expert_num,
    )
    if moe_expert_num is not None and isinstance(tmin, int) and isinstance(
            tmax, int):
        if tmin < 0 or tmax >= moe_expert_num:
            logger.warning(
                "[FFN-POST-RECV-SYNC-DIAG] OOB topk_ids layer=%s ubatch=%s ep=%s/%s "
                "range=[%s,%s] vs n_routed=%s",
                layer_idx, ubatch_idx, ep_rank, ep_world, tmin, tmax,
                moe_expert_num,
            )


def _summarize_dp_metadata_list(dp_metadata_list: dict | None) -> str:
    if not dp_metadata_list:
        return "{}"
    parts = []
    for k in sorted(dp_metadata_list.keys()):
        m = dp_metadata_list[k]
        try:
            nta = m.num_tokens_across_dp_cpu.tolist()
            mx = int(m.max_tokens_across_dp_cpu.item())
        except Exception as ex:
            parts.append(f"{k}:<err {ex}>")
            continue
        parts.append(f"{k}:nta={nta},max={mx}")
    return "{" + "; ".join(parts) + "}"


class NPUFFNModelRunner(NPUModelRunner,GPUFFNModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config=vllm_config,
                         device=device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.dtype = self.model_config.dtype
        self.load_config = vllm_config.load_config
        self.first_k_dense_replace = self.model_config.hf_config.first_k_dense_replace
        self.num_hidden_layers = self.model_config.hf_config.num_hidden_layers

        self.afd_config = vllm_config.afd_config
        if not self.afd_config or not self.afd_config.is_ffn_server:
            raise ValueError(
                "AFD config must be provided with afd_role='ffn' for FFN server"
            )
        self.connector_name = self.afd_config.afd_connector
        
        # Initialize ACL graph support
        self.aclgraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        # Storage for captured graphs - 使用 dp_metadata_key 作为 key
        # key 格式: ((stage_idx, tuple(num_tokens_across_dp_cpu)), ...)
        self._acl_graphs: dict[tuple, dict] = {}
        self.graph_pool = None
        if self.use_aclgraph:
            self.graph_pool = current_platform.get_global_graph_pool()

        # True only while executing inside ``torch.npu.graph`` (capture). Used to
        # avoid device sync / barriers that Ascend rejects with 107027
        # ("stream is captured").
        self._inside_ffn_npu_graph_capture = False

        assert self.afd_config.is_ffn_server
        self.connector = AFDConnectorFactory.create_connector(
            get_world_group().rank,
            get_world_group().local_rank, self.vllm_config)
        
        self.connector.init_afd_connector()
        self.attn_size = self.connector.attn_size
        self.ffn_size = self.connector.ffn_size

        self.ffn_multistream_capable = self.afd_config.is_ffn_multistream
        num_ubatches_cfg = self.parallel_config.num_ubatches if self.parallel_config.num_ubatches else 1
        self.ffn_comm_stream = torch.npu.Stream() if self.ffn_multistream_capable else None
        self.ffn_comm_events = [torch.npu.Event() for _ in range(num_ubatches_cfg)] if self.ffn_multistream_capable else []
        print(f'attn_size = {self.attn_size},ffn_size = {self.ffn_size}')
        if getattr(self.model_config.hf_config, "text_config",
                   None) is not None:
            self.num_layers = (
                self.model_config.hf_config.text_config.num_hidden_layers)
        else:
            self.num_layers = self.model_config.hf_config.num_hidden_layers
        self.dummy_run_call_cnt = 0
        self.replay_cnt = 0
        self.topk = self.model_config.hf_config.num_experts_per_tok
        self.n_routed_experts = self.model_config.hf_config.n_routed_experts
        self.hidden_size = self.model_config.hf_config.hidden_size
        print(f'self.topk is {self.topk}')
        self.decode_max_num_token = self.scheduler_config.max_num_seqs * \
                        self.uniform_decode_query_len

        # Initialize cudagraph keys for FFN server as initialize_kv_cache is not called
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            self.vllm_config.compilation_config.cudagraph_mode,
            self.uniform_decode_query_len
        )

        if envs_ascend.VLLM_ASCEND_FFN_DIAG_LOG_INIT:
            cc = self.vllm_config.compilation_config
            logger.info(
                "[FFN-DIAG] NPUFFNModelRunner init: enforce_eager=%s "
                "compilation_mode=%s cudagraph_mode=%s capture_sizes=%s "
                "use_aclgraph=%s num_ubatches_cfg=%s",
                self.model_config.enforce_eager,
                cc.mode,
                cc.cudagraph_mode,
                list(cc.cudagraph_capture_sizes or []),
                self.use_aclgraph,
                num_ubatches_cfg,
            )

        self.prof = None
        if envs_ascend.VLLM_ASCEND_FFN_PROFILER_ENABLE:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=torch_npu.profiler.ExportType.Text,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            )
            logger.info(
                "NPUFFNModelRunner profiler enabled. Traces will be saved to: %s",
                envs_ascend.VLLM_ASCEND_FFN_PROFILER_DIR)
            self.prof = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU
                ],
                schedule=torch_npu.profiler.schedule(
                    wait=envs_ascend.VLLM_ASCEND_FFN_PROFILER_WAIT,
                    warmup=envs_ascend.VLLM_ASCEND_FFN_PROFILER_WARMUP,
                    active=envs_ascend.VLLM_ASCEND_FFN_PROFILER_ACTIVE,
                    repeat=envs_ascend.VLLM_ASCEND_FFN_PROFILER_REPEAT,
                    skip_first=envs_ascend.VLLM_ASCEND_FFN_PROFILER_SKIP_FIRST),
                # 初步采集最好不要使用下面两个选项， with_stack 会大幅增加采集时间及采集的数据大小，深入分析CPU测瓶颈时再打开
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    envs_ascend.VLLM_ASCEND_FFN_PROFILER_DIR))

    def get_model(self) -> nn.Module:
        return self.model

    def initialize_afd_connector(self) -> None:
        self.connector.init_afd_connector()

    def _get_current_layer_idx(self) -> int:
        return (self._counter // self._current_num_ubatches) % self.num_layers
        
    @torch.inference_mode()
    def execute_model(self, scheduler_output=None, intermediate_tensors=None,
                     dp_metadata_list: dict | None = None,
                     afd_num_actual_tokens: int | None = None,
                     attn_aclgraph_runtime_mode: int | None = None):
        """Execute FFN computation for a single request

        Args:
            scheduler_output: 调度器输出（FFN侧通常为None）
            intermediate_tensors: 中间张量（FFN侧通常为None）
            dp_metadata_list: dp_metadata列表，包含每个stage的token数量信息
        """
        if self.prof is not None:
            self.prof.step()
        try:
            # 从dp_metadata_list中获取is_ubatch
            is_ubatch = dp_metadata_list is not None and len(dp_metadata_list) > 1

            if self.use_aclgraph:
                # 图是在 Attention FULL 下与 CAM/a2e 一起抓的；若本步 Attention 走
                # eager (NONE) 或 PIECEWISE 而 FFN 仍 replay，极易 a2e 与 MoE 不同步
                # → 损坏的 mask/topk → 507015。
                replay_blocked_by_attn_mode = (
                    attn_aclgraph_runtime_mode is not None
                    and int(attn_aclgraph_runtime_mode)
                    != int(CUDAGraphMode.FULL.value))
                # 用 dp_metadata_key 查找 graph
                dp_metadata_key = self._get_dp_metadata_key(
                    dp_metadata_list, afd_num_actual_tokens)
                acl_graph_info = self._acl_graphs.get(dp_metadata_key)
                if _ffn_graph_mtp_trace_enabled():
                    logger.info(
                        "[FFN-GRAPH-MTP-TRACE] execute_model: dp_key=%s hit=%s "
                        "num_stages=%s captured_keys=%s replay_cnt=%s "
                        "attn_acl_mode=%s replay_blocked=%s",
                        dp_metadata_key,
                        acl_graph_info is not None,
                        len(dp_metadata_list) if dp_metadata_list else 0,
                        list(self._acl_graphs.keys()),
                        self.replay_cnt,
                        attn_aclgraph_runtime_mode,
                        replay_blocked_by_attn_mode,
                    )
                if acl_graph_info is not None:
                    if replay_blocked_by_attn_mode:
                        logger.warning(
                            "Skipping NPUGraph replay: Attention "
                            "cudagraph_runtime_mode=%s (not FULL); running eager "
                            "_ffn_forward for CAM/a2e alignment.",
                            attn_aclgraph_runtime_mode,
                        )
                        self._ffn_forward(
                            aclgraph_runtime_mode=CUDAGraphMode.NONE,
                            dp_metadata_list=dp_metadata_list,
                            afd_num_actual_tokens=afd_num_actual_tokens,
                        )
                    elif _ffn_graph_replay_denied(dp_metadata_key):
                        logger.warning(
                            "Skipping NPUGraph replay for dp_key=%s (listed in "
                            "VLLM_ASCEND_FFN_GRAPH_REPLAY_DENYLIST); running eager "
                            "_ffn_forward.",
                            dp_metadata_key,
                        )
                        self._ffn_forward(
                            aclgraph_runtime_mode=CUDAGraphMode.NONE,
                            dp_metadata_list=dp_metadata_list,
                            afd_num_actual_tokens=afd_num_actual_tokens,
                        )
                    else:
                        if envs_ascend.VLLM_ASCEND_FFN_GRAPH_REPLAY_PRE_SYNC:
                            torch.npu.current_stream().synchronize()
                            logger.info(
                                "[FFN-GRAPH] replay pre-sync OK dp_key=%s "
                                "replay_cnt=%s",
                                dp_metadata_key,
                                self.replay_cnt,
                            )
                        graph = acl_graph_info['graph']
                        with self._ffn_ascend_forward_ctx(
                                dp_metadata_list,
                                CUDAGraphMode.FULL,
                                afd_num_actual_tokens,
                        ):
                            graph.replay()
                        self.replay_cnt += 1
                        logger.debug(
                            "ffn replay, replay_cnt is %s, dp_metadata_key=%s",
                            self.replay_cnt,
                            dp_metadata_key,
                        )
                else:
                    # fallback to eager mode
                    logger.warning(
                        "No acl graph found for dp_metadata_key=%s, fallback to "
                        "eager (captured_keys=%s)",
                        dp_metadata_key,
                        list(self._acl_graphs.keys()),
                    )
                    self._ffn_forward(
                        aclgraph_runtime_mode=CUDAGraphMode.NONE,
                        dp_metadata_list=dp_metadata_list,
                        afd_num_actual_tokens=afd_num_actual_tokens,
                    )
            else:
                # eager mode for non-ubatch or no aclgraph
                logger.debug(f"ffn_forward, is_ubatch is {is_ubatch}")
                self._ffn_forward(
                    aclgraph_runtime_mode=CUDAGraphMode.NONE,
                    dp_metadata_list=dp_metadata_list,
                    afd_num_actual_tokens=afd_num_actual_tokens,
                )

        except Exception as e:
            raise ValueError(
                f"Error computing FFN: {e}"
            ) from e
        return None  # FFN server doesn't return ModelRunnerOutput

    def capture_model(self,
                      dp_metadata_list: Optional[dict] = None,
                      is_warmup: bool = False,
                      is_attn_graph_capturing: bool = True,
                      afd_num_actual_tokens: int | None = None,
                      attn_aclgraph_runtime_mode: int | None = None) -> int:
        """Capture ACL graphs for FFN operations.

        Args:
            dp_metadata_list: 从Attention侧接收的dp_metadata列表
            is_warmup: 是否为warmup模式（只执行forward，不capture graph）
            is_attn_graph_capturing: Attention侧是否正在capture（用于同步）
        """
        if not self.use_aclgraph:
            return 0

        logger.debug("Starting ACL graph capture for FFN operations, "
                     "is_warmup=%s", is_warmup)
        start_time = time.perf_counter()
        start_free_npu_memory = torch.npu.mem_get_info()[0]

        set_cudagraph_capturing_enabled(True)
        try:
            if is_warmup:
                # Warmup模式：只执行forward，不capture graph
                self._warmup_model(
                    dp_metadata_list=dp_metadata_list,
                    afd_num_actual_tokens=afd_num_actual_tokens,
                )
                logger.info("FFN warmup completed, dp_metadata_list=%s", dp_metadata_list)
            else:
                # 正式Capture模式：根据dp_metadata_list捕获单个graph
                self._capture_model(
                    dp_metadata_list=dp_metadata_list,
                    afd_num_actual_tokens=afd_num_actual_tokens,
                )
        finally:
            set_cudagraph_capturing_enabled(False)

        end_time = time.perf_counter()
        end_free_npu_memory = torch.npu.mem_get_info()[0]
        elapsed_time = end_time - start_time
        npu_graph_size = start_free_npu_memory - end_free_npu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, npu_graph_size / (1 << 30))

        return npu_graph_size

    def _get_dp_metadata_key(
        self,
        dp_metadata_list: dict,
        afd_num_actual_tokens: int | None = None,
    ) -> tuple:
        """Extract a hashable key from dp_metadata_list for CUDA graph lookup.

        与 GPU 版本的 _make_graph_key 保持一致。
        The key is a tuple of (stage_idx, tuple(num_tokens_across_dp_cpu))
        for each stage, sorted by stage_idx.

        When ``afd_num_actual_tokens`` is passed it is only used for tracing /
        diagnostics; the NPUGraph pool key must stay **layout-only** (padded DP
        token counts). Spec/MTP steps share the same padded shape and the same
        captured graph; per-step validity comes from AFD ``x_active_mask`` inside
        the captured recv+MoE region. Extending the key with an ``int`` caused
        uncaptured keys at runtime → eager fallback or stale graph mismatches.

        Args:
            dp_metadata_list: {stage_idx: DPMetadata}
            afd_num_actual_tokens: 本步真实 token 数（仅用于 TRACE/DENY 对比，不参与 key）

        Returns:
            tuple: ((stage_idx, tuple(num_tokens_across_dp_cpu)), ...)
        """
        if dp_metadata_list is None:
            base: tuple = ()
        else:
            base = tuple(
                (stage_idx, tuple(meta.num_tokens_across_dp_cpu.tolist()))
                for stage_idx, meta in sorted(dp_metadata_list.items())
            )
        return base

    def _warmup_model(self, dp_metadata_list: dict = None,
                      afd_num_actual_tokens: int | None = None) -> None:
        """执行warmup，只运行forward不capture graph

        Args:
            is_ubatch: 是否为ubatch模式
            dp_metadata_list: 从Attention侧接收的dp_metadata列表
        """
        # Warmup只执行eager模式的forward，根据dp_metadata_list确定num_tokens
        dp_metadata_key = self._get_dp_metadata_key(
            dp_metadata_list, afd_num_actual_tokens)

        self._dummy_run(aclgraph_runtime_mode=CUDAGraphMode.NONE,
                        uniform_decode=True,
                        dp_metadata_list=dp_metadata_list,
                        dp_metadata_key=dp_metadata_key,
                        afd_num_actual_tokens=afd_num_actual_tokens)
        logger.debug("FFN warmup for dp_metadata_key=%s", dp_metadata_key)

    def _capture_model(self, dp_metadata_list: dict = None,
                       afd_num_actual_tokens: int | None = None):
        """内部capture实现 - 根据dp_metadata_list捕获单个graph

        Args:
            is_ubatch: 是否为ubatch模式
            dp_metadata_list: 从Attention侧接收的dp_metadata列表
        """
        if dp_metadata_list is None:
            logger.warning("dp_metadata_list is None, skip capture")
            return

        # 生成 dp_metadata key
        dp_metadata_key = self._get_dp_metadata_key(
            dp_metadata_list, afd_num_actual_tokens)
        if _ffn_graph_mtp_trace_enabled():
            logger.info(
                "[FFN-GRAPH-MTP-TRACE] _capture_model: dp_key=%s dp_summary=%s "
                "existing_graph_keys=%s",
                dp_metadata_key,
                _summarize_dp_metadata_list(dp_metadata_list),
                list(self._acl_graphs.keys()),
            )

        @contextmanager
        def freeze_gc():
            # Optimize garbage collection during CUDA graph capture.
            gc.collect()
            should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
            if should_freeze:
                gc.freeze()
            try:
                yield
            finally:
                if should_freeze:
                    gc.unfreeze()
                    gc.collect()

        with freeze_gc(), graph_capture(device=self.device):
            # 直接捕获单个graph，无warmup（warmup已通过Attention侧同步完成）
            self._capture_single_aclgraph(
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                uniform_decode=True,
                dp_metadata_key=dp_metadata_key,
                dp_metadata_list=dp_metadata_list,
                afd_num_actual_tokens=afd_num_actual_tokens,
            )

    def _capture_single_aclgraph(self,
                                  cudagraph_runtime_mode: CUDAGraphMode,
                                  uniform_decode: bool,
                                  dp_metadata_key: tuple = None,
                                  dp_metadata_list: dict = None,
                                  afd_num_actual_tokens: int | None = None):
        """捕获单个 ACL graph（无 warmup 循环）

        Args:
            num_tokens: token数量
            cudagraph_runtime_mode: graph模式
            uniform_decode: 是否为uniform decode
            is_ubatch: 是否为ubatch模式
            dp_metadata_key: dp_metadata的key，用于存储graph
        """
        assert cudagraph_runtime_mode != CUDAGraphMode.NONE
        logger.info("Capturing ACL graph for dp_metadata_key=%s", dp_metadata_key)

        # 直接capture，不做warmup（warmup由Attention侧控制）
        self._dummy_run(aclgraph_runtime_mode=cudagraph_runtime_mode,
                        uniform_decode=uniform_decode,
                        dp_metadata_list=dp_metadata_list,
                        dp_metadata_key=dp_metadata_key,
                        afd_num_actual_tokens=afd_num_actual_tokens)

    def _dummy_run(self,
                   aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
                   uniform_decode: bool = False,
                   dp_metadata_list: dict | None = None,
                   dp_metadata_key: tuple = None,
                   **kwargs):
        """执行dummy run用于warmup或capture

        Args:
            num_tokens: token数量
            aclgraph_runtime_mode: ACL graph运行模式
            force_attention: 是否强制attention
            uniform_decode: 是否为uniform decode
            dp_metadata_list: dp_metadata列表，包含每个stage的token数量信息
            dp_metadata_key: dp_metadata的key，用于存储graph（替代num_tokens作为key）
        """
        is_ubatch = dp_metadata_list is not None and len(dp_metadata_list) > 1
        print(f'is_ubatch in _dummy_run is {is_ubatch}')
        if _ffn_graph_mtp_trace_enabled():
            logger.info(
                "[FFN-GRAPH-MTP-TRACE] _dummy_run: dp_key=%s acl_mode=%s "
                "is_ubatch=%s dp_summary=%s num_layers=%s num_hidden_layers=%s",
                dp_metadata_key,
                aclgraph_runtime_mode,
                is_ubatch,
                _summarize_dp_metadata_list(dp_metadata_list),
                self.num_layers,
                self.num_hidden_layers,
            )

        # only support eager mode and piecewise graph now
        assert aclgraph_runtime_mode is None or aclgraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
        }
        # _ag_mode, batch_descriptor = \
        #     self.cudagraph_dispatcher.dispatch(num_tokens=num_tokens, uniform_decode=uniform_decode, has_lora=False)
        afd_nt = kwargs.get("afd_num_actual_tokens")
        if aclgraph_runtime_mode == CUDAGraphMode.FULL:
            # Create and capture the graph
            aclgraph = torch.npu.NPUGraph()
            self._inside_ffn_npu_graph_capture = True
            try:
                with torch.npu.graph(aclgraph, pool=self.graph_pool):
                    # compute_ffn_output
                    output = self._ffn_forward(
                        aclgraph_runtime_mode=aclgraph_runtime_mode,
                        dp_metadata_list=dp_metadata_list,
                        afd_num_actual_tokens=afd_nt,
                    )
            finally:
                self._inside_ffn_npu_graph_capture = False
            # Store the captured graph with dp_metadata_key as key
            self._acl_graphs[dp_metadata_key] = {
                'graph': aclgraph,
                'input_hidden_states': output,
                'output': output
            }
            print(f'self._acl_graphs key={dp_metadata_key}', flush=True)
        else:
            self._ffn_forward(
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                dp_metadata_list=dp_metadata_list,
                afd_num_actual_tokens=afd_nt,
            )
            print("finsh capture warm_up or prefile run",flush=True)
        print(f'self.dummy_run_call_cnt is {self.dummy_run_call_cnt}')
        self.dummy_run_call_cnt += 1

    # TODO: to adapt m2nAFDConnector for deepseek w9a8量化适配
    def _build_and_recv_m2n_afdconnector(
        self,
        m2n_afdconnector_data: M2NAFDConnectorMetadata,
        n_routed_experts: int,
        hidden_size: int,
        topk: int,
        expert_token_nums_type:int,
        attn_size: int,
        max_num_tokens: int,
        quant_mode: int = 0,
        expand_x_type: torch.dtype = torch.bfloat16,
    ):
        m2n_afdconnector_data.quant_mode = quant_mode
        m2n_afdconnector_data.expand_x_type = expand_x_type
        m2n_afdconnector_data.moe_expert_num = n_routed_experts
        m2n_afdconnector_data.h = hidden_size
        m2n_afdconnector_data.k = topk
        m2n_afdconnector_data.expert_token_nums_type = expert_token_nums_type
        m2n_afdconnector_data.aiv_num = 48
        m2n_afdconnector_data.batch_size = max_num_tokens * m2n_afdconnector_data.k * attn_size
        
        recv_output = self.connector.recv_attn_output(metadata=m2n_afdconnector_data)
        m2n_afdconnector_data.handle = recv_output.handle
        m2n_afdconnector_data.topk_weights = recv_output.topk_weights
        
        return recv_output
    
    def _build_ffn_num_tokens_across_dp(self, dp_metadata_list: dict) -> Optional[torch.Tensor]:
        """构造 FFN 侧的 num_tokens_across_dp tensor

        对于不对称 A/F 场景（A > F），需要将多个 A 的 token 数量合并到对应的 F。
        """
        # 从 dp_metadata_list 获取 dp_metadata
        dp_metadata = dp_metadata_list.get(0, None) if dp_metadata_list else None

        if dp_metadata is None:
            return None

        attn_num_tokens = dp_metadata.num_tokens_across_dp_cpu

        # For asymmetric A/F, scale token counts by ratio
        # because FFN receives concatenated tokens from multiple A's
        if hasattr(self.connector, 'ratio') and self.connector.ratio > 1:
            ffn_size = self.connector.ffn_size
            ratio = self.connector.ratio

            ffn_num_tokens_list = []
            for f_idx in range(ffn_size):
                start_a_idx = f_idx * ratio
                end_a_idx = start_a_idx + ratio
                ffn_num_tokens_list.append(
                    attn_num_tokens[start_a_idx:end_a_idx].sum().item()
                )

            ffn_num_tokens_across_dp_cpu = torch.tensor(
                ffn_num_tokens_list,
                dtype=attn_num_tokens.dtype,
                device=attn_num_tokens.device,
            )
            return ffn_num_tokens_across_dp_cpu
        else:
            return attn_num_tokens

    @contextmanager
    def _ffn_ascend_forward_ctx(
        self,
        dp_metadata_list: dict | None,
        aclgraph_runtime_mode: CUDAGraphMode,
        afd_num_actual_tokens: int | None = None,
    ):
        """Match capture-time ascend forward context for eager *and* NPUGraph replay.

        NPUGraph ``replay()`` does not re-run Python ``_ffn_forward``; without
        this, ``get_forward_context()`` (moe_comm_type, mc2 scratch, padded
        lengths) can differ from capture and break MTP + AFD + EP paths.
        """
        is_ubatch = dp_metadata_list is not None and len(dp_metadata_list) > 1
        num_ubatches = self.parallel_config.num_ubatches if is_ubatch else 1
        afd_metadata = AFDMetadata(
            afd_tokens_start_loc=[],
            afd_reqs_start_loc=[],
            afd_stage_idx=0,
            afd_connector=self.connector,
            afd_tokens_lens=[],
            num_of_stages=num_ubatches,
        )
        num_tokens_across_dp = self._build_ffn_num_tokens_across_dp(
            dp_metadata_list)
        local_num_tokens = 0
        if num_tokens_across_dp is not None and num_tokens_across_dp.numel() > 0:
            try:
                dp = get_dp_group()
                rid = dp.rank_in_group
                if rid is not None and 0 <= int(rid) < int(
                        num_tokens_across_dp.numel()):
                    local_num_tokens = int(
                        num_tokens_across_dp[int(rid)].item())
                else:
                    local_num_tokens = int(num_tokens_across_dp[0].item())
            except Exception:
                local_num_tokens = int(num_tokens_across_dp[0].item())
        _ffd_ctx = dict(
            attn_metadata=None,
            vllm_config=self.vllm_config,
            batch_descriptor=None,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            model_instance=self.model,
            afd_metadata=afd_metadata,
            num_tokens=local_num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
        )
        if afd_num_actual_tokens is not None:
            _ffd_ctx["num_actual_tokens"] = int(afd_num_actual_tokens)
        with set_ascend_forward_context(**_ffd_ctx):
            yield

    def _ffn_forward(self,
                     aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
                     dp_metadata_list: dict | None = None,
                     afd_num_actual_tokens: int | None = None):
        """Run FFN computation for graph capture or replay"""
        is_ubatch = dp_metadata_list is not None and len(dp_metadata_list) > 1
        num_ubatches = self.parallel_config.num_ubatches if is_ubatch else 1
        rank_ffn_output = None
        print(f"jcz _ffn_forward max_num_tokens:{self.max_num_tokens}")
        if _ffn_graph_mtp_trace_enabled():
            logger.info(
                "[FFN-GRAPH-MTP-TRACE] _ffn_forward: aclgraph_runtime_mode=%s "
                "is_ubatch=%s num_ubatches_cfg=%s num_ubatches_effective=%s "
                "max_num_tokens(cfg)=%s dp_summary=%s dp_key=%s",
                aclgraph_runtime_mode,
                is_ubatch,
                self.parallel_config.num_ubatches,
                num_ubatches,
                self.max_num_tokens,
                _summarize_dp_metadata_list(dp_metadata_list),
                self._get_dp_metadata_key(
                    dp_metadata_list, afd_num_actual_tokens),
            )

        ffn_multistream_enable = self.ffn_multistream_capable and num_ubatches > 1

        ffn_event_recorded = [False] * num_ubatches
        with self._ffn_ascend_forward_ctx(
                dp_metadata_list, aclgraph_runtime_mode, afd_num_actual_tokens):
            for layer_idx in range(0, self.num_layers):
                layer_multistream = ffn_multistream_enable and (layer_idx > 0)
                for ubatch_idx in range(num_ubatches):
                    if ffn_multistream_enable and ffn_event_recorded[ubatch_idx]:
                        self.ffn_comm_events[ubatch_idx].wait(torch.npu.current_stream())
                    # recv (a2f): runs on default stream
                    afd_connector_data = self.connector.create_recv_metadata(
                        dp_metadata_list=dp_metadata_list,
                        ubatch_idx=ubatch_idx,
                        layer_idx=layer_idx,
                        max_num_tokens=self.max_num_tokens)
                    recv_output = self.connector.recv_attn_output(metadata=afd_connector_data, ubatch_idx=ubatch_idx)
                    if hasattr(self.connector, "update_metadata") and afd_connector_data is not None:
                        self.connector.update_metadata(afd_connector_data, recv_output)
                    print(f'{self.connector_name} recv_attn_output success ,layer id is {layer_idx}, '
                        f'ubatch_idx is {ubatch_idx} recv_output:{recv_output.hidden_states.shape}', flush=True)

                    # EP ranks must enter MoE/CAM for the same (layer, ubatch) together.
                    # Logs showed EP0/EP1 progressing at different speeds → garbage topk on one
                    # rank (e.g. int32 expert ids ±1e9) → 507015 in dispatch / aclnnCast.
                    #
                    # During ``torch.npu.graph`` capture, ``torch.npu.synchronize()`` (device-wide)
                    # triggers Ascend 107027 (stream is captured). Use only the **current**
                    # stream there if needed; logs show EP barrier cannot be omitted during capture
                    # or ranks finish different layers concurrently and MoE/collectives corrupt.
                    try:
                        ep_g = get_ep_group()
                    except Exception:
                        ep_g = None
                    if (ep_g is not None and ep_g.world_size > 1
                            and dist.is_initialized()):
                        if self._inside_ffn_npu_graph_capture:
                            dist.barrier(group=ep_g.cpu_group)
                        else:
                            torch.npu.current_stream().synchronize()
                            dist.barrier(group=ep_g.cpu_group)

                    hidden_states = recv_output.hidden_states
                    dynamic_scales = recv_output.dynamic_scales
                    group_list = recv_output.group_list
                    topk_weights = recv_output.topk_weights
                    topk_ids = recv_output.topk_ids
                    router_logits = recv_output.router_logits
                    row_idx = recv_output.row_idx
                    x_active_mask = recv_output.x_active_mask
                    if _ffn_graph_mtp_trace_enabled():
                        msum = None
                        if x_active_mask is not None:
                            try:
                                msum = int(x_active_mask.sum().detach().cpu().item())
                            except Exception:
                                msum = "<?>"
                        tmin = tmax = None
                        if topk_ids is not None and topk_ids.numel() > 0:
                            try:
                                tmin = int(topk_ids.min().detach().cpu().item())
                                tmax = int(topk_ids.max().detach().cpu().item())
                            except Exception:
                                tmin = tmax = None
                        logger.info(
                            "[FFN-GRAPH-MTP-TRACE] after_recv: layer=%s/%s "
                            "ubatch=%s is_mtp_layer=%s hid=%s topk_w=%s topk_id=%s "
                            "topk_ids_minmax=%s router_logits=%s x_active_mask_sum=%s "
                            "group_list=%s",
                            layer_idx,
                            self.num_layers - 1,
                            ubatch_idx,
                            layer_idx >= self.num_hidden_layers,
                            tuple(hidden_states.shape),
                            tuple(topk_weights.shape) if topk_weights is not None else None,
                            tuple(topk_ids.shape) if topk_ids is not None else None,
                            (tmin, tmax),
                            tuple(router_logits.shape)
                            if router_logits is not None else None,
                            msum,
                            tuple(group_list.shape) if group_list is not None else None,
                        )

                    _maybe_warn_moe_topk_oob(
                        topk_ids,
                        self.n_routed_experts,
                        where="after_recv",
                        layer_idx=layer_idx,
                        ubatch_idx=ubatch_idx,
                    )

                    if (envs_ascend.VLLM_ASCEND_FFN_POST_RECV_SYNC_DIAG
                            and topk_ids is not None):
                        _log_ffn_post_recv_moe_sync_diag(
                            logger=logger,
                            layer_idx=layer_idx,
                            num_layers=self.num_layers,
                            ubatch_idx=ubatch_idx,
                            dp_key=self._get_dp_metadata_key(
                                dp_metadata_list, afd_num_actual_tokens),
                            hidden_states=hidden_states,
                            topk_ids=topk_ids,
                            x_active_mask=x_active_mask,
                            moe_expert_num=self.n_routed_experts,
                        )

                    # FFN compute: runs on default stream
                    rank_ffn_output = self._run_ffn_computation(
                        hidden_states=hidden_states,
                        layer_idx=layer_idx,
                        group_list=group_list,
                        dynamic_scales=dynamic_scales if self.connector.quant_mode == 1 else None,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        router_logits=router_logits,
                        row_idx=row_idx,
                        x_active_mask=x_active_mask,
                        cam_p2p_ep_name=recv_output.cam_p2p_ep_name or ""
                    )
                    # send (f2a): when multistream enabled, dispatched to comm_stream
                    self.connector.send_ffn_output(
                        rank_ffn_output, afd_connector_data,
                        ubatch_idx=ubatch_idx,
                        multistream_enable=layer_multistream,
                        comm_stream=self.ffn_comm_stream if layer_multistream else None,
                        comm_event=self.ffn_comm_events[ubatch_idx] if layer_multistream else None)
                    if layer_multistream:
                        ffn_event_recorded[ubatch_idx] = True
                    print(f'cam send_ffn_output success ,layer id is {layer_idx},ubatch_idx is {ubatch_idx}', flush=True)

                if (envs_ascend.VLLM_ASCEND_FFN_DIAG_SYNC_PER_LAYER
                        and not self._inside_ffn_npu_graph_capture):
                    torch.npu.current_stream().synchronize()
                    logger.info(
                        "[FFN-DIAG] per-layer sync OK layer_idx=%s/%s dp_key=%s "
                        "aclgraph_mode=%s",
                        layer_idx,
                        self.num_layers - 1,
                        self._get_dp_metadata_key(
                            dp_metadata_list, afd_num_actual_tokens),
                        aclgraph_runtime_mode,
                    )

            if ffn_multistream_enable:
                curr_stream = torch.npu.current_stream()
                for i, ev in enumerate(self.ffn_comm_events):
                    if ffn_event_recorded[i]:
                        ev.wait(curr_stream)
        return rank_ffn_output

    def _run_ffn_computation(self,
                             hidden_states: torch.Tensor,
                             layer_idx: Optional[int] = None,
                             capture_mode: bool = False,
                             router_logits: Optional[torch.Tensor] = None,
                             group_list: Optional[torch.Tensor] = None,
                             dynamic_scales: Optional[torch.Tensor] = None,
                             topk_weights: Optional[torch.Tensor] = None,
                             topk_ids: Optional[torch.Tensor] = None,
                             row_idx: Optional[torch.Tensor] = None,
                             x_active_mask: Optional[torch.Tensor] = None,
                             cam_p2p_ep_name: Optional[str] = ""):
        """Run FFN computation for graph capture or replay."""
        rank_ffn_output = self.model.compute_ffn_output(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            router_logits=router_logits,
            group_list=group_list,
            dynamic_scales=dynamic_scales,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            row_idx=row_idx,
            x_active_mask=x_active_mask,
            cam_p2p_ep_name=cam_p2p_ep_name)
        return rank_ffn_output

    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        """FFN servers don't use samplers."""
        pass

    def update_config(self, overrides: dict[str, Any]) -> None:
        """Update configuration for FFN model runner."""
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, \
                f"Config `{config_name}` not supported. " \
                f"Allowed configs: {allowed_config_names}"
            config = getattr(self, config_name)
            from vllm.config import update_config
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def reload_weights(self) -> None:
        """Reload model weights for FFN model runner."""
        assert getattr(self, "model", None) is not None, \
            "Cannot reload weights before model is loaded."
        model_loader = get_model_loader(self.load_config)
        logger.info("Reloading weights inplace...")
        model = self.get_model()
        model_loader.load_weights(model, model_config=self.model_config)


    def lora_config(self):
        """FFN servers don't support LoRA."""
        return None


    def is_pooling_model(self) -> bool:
        """FFN servers are not pooling models."""
        return False

    def _dummy_pooler_run(self, hidden_states: torch.Tensor):
        """FFN servers don't have poolers."""
        pass

    def get_supported_tasks(self):
        """Get supported tasks for FFN model runner."""
        return []

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        """Get number of input tokens for FFN model runner."""
        return num_scheduled_tokens

    def take_draft_token_ids(self, **kwargs):
        """FFN servers don't support draft tokens."""
        pass


    def eplb_state(self):
        """FFN servers don't have EPLB state."""
        return None

    def ensure_kv_transfer_shutdown(self):
        """FFN servers don't need KV transfer shutdown."""
        pass

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        """FFN servers don't support tensorized model saving."""
        raise NotImplementedError(
            "FFN servers don't support tensorized model saving")
