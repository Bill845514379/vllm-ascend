import os
from dataclasses import dataclass
from typing import Any, Optional

from vllm.distributed.afd_transfer.afd_connector import (AFDConnectorBase, AFDConnectorFactory,
                                                         AFDConnectorMetadata)
from vllm.logger import init_logger

__all__ = ["AFDConnectorBase", "AFDConnectorMetadata", "AFDConnectorFactory"]

import torch_npu
import torch
import pickle

from torch.distributed.distributed_c10d import _get_default_group
import re

import torch
from torch.distributed.distributed_c10d import _update_default_pg, _get_default_group

from vllm.distributed.parallel_state import (get_dp_group, init_afd_process_group,
                                              init_model_parallel_group)
from vllm_ascend.distributed.metadata import (CAMP2PAFDConnectorMetadata)
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ascend_config import get_ascend_config
from vllm.config import VllmConfig, CUDAGraphMode, CompilationMode
from vllm.distributed.afd_transfer.afd_connector.p2p_connector import DefaultProcessGroupSwitcher

from vllm.utils.torch_utils import direct_register_custom_op
from vllm.forward_context import ForwardContext, get_forward_context

from vllm_ascend.utils import npu_stream_switch_within_graph

logger = init_logger(__name__)

# vLLM's TorchCompileWithNoGuardsWrapper rejects @torch._dynamo.disable callees
# inside the compiled region; keep scalar tensor reads traceable where possible.
try:
    torch._dynamo.config.capture_scalar_outputs = True
except Exception:
    pass


def _in_torch_compile_trace() -> bool:
    try:
        if hasattr(torch.compiler, "is_compiling"):
            return bool(torch.compiler.is_compiling())
    except Exception:
        pass
    try:
        return bool(torch._dynamo.is_compiling())
    except Exception:
        return False


def _active_dp_metadata_from_forward_ctx(ctx: ForwardContext):
    if ctx.dp_metadata is not None:
        return ctx.dp_metadata
    am = ctx.afd_metadata
    if am is None:
        return None
    dpl = getattr(am, "dp_metadata_list", None)
    if not dpl:
        return None
    idx = ctx.ubatch_idx
    if idx < len(dpl):
        return dpl[idx]
    return dpl[0]


def _expected_local_attn_rows_for_a2e(dm) -> Optional[int]:
    """Rows this DP rank should send into a2e; must match dp_metadata treaty.

    Always use num_tokens_across_dp_cpu[dp_rank] only. Do not index
    dm.local_sizes by dp_rank when SP/chunking expands that list — lengths
    can match num_tokens_across_dp by coincidence but semantics differ.
    """
    try:
        dp_rank = get_dp_group().rank_in_group
    except Exception:
        return None
    nta = dm.num_tokens_across_dp_cpu
    if dp_rank < 0 or dp_rank >= nta.numel():
        return None
    return int(nta[dp_rank].item())


def _barrier_attention_dp_before_cam_send() -> None:
    """Match Attention DP ranks at the same layer before a2e / CAM collectives.

    If one rank finishes attention earlier and enters cam_send while the other
    is still in a prior layer, the paired FFN EP ranks can process different
    layers and MoE HCCL collectives hang (stuck waiting for a peer).
    """
    if _in_torch_compile_trace():
        return
    try:
        dp = get_dp_group()
    except Exception:
        return
    if dp is None or dp.world_size <= 1:
        return
    dp.barrier()


def _pad_attn_tensors_to_dp_metadata(
    hidden_states: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    topk_idx: Optional[torch.Tensor],
    compute_gate: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Align Attention send tensors with DPMetadata before a2e (torch_binding uses x.size(0))."""
    ctx = get_forward_context()
    if ctx is None:
        return hidden_states, topk_weights, topk_idx
    dm = _active_dp_metadata_from_forward_ctx(ctx)
    if dm is None:
        return hidden_states, topk_weights, topk_idx
    # Prefer int expected rows stashed on ForwardContext when the runner builds
    # the batch (see vllm forward_context / npu_ubatch_wrapper). Reading
    # num_tokens_across_dp_cpu inside torch.compile makes expected a
    # data-dependent scalar (Dynamo UserError on `if actual >= expected`).
    expected = getattr(ctx, "afd_expected_a2e_rows", None)
    if expected is None:
        expected = getattr(ctx, "num_tokens", None)
    if expected is None:
        expected = _expected_local_attn_rows_for_a2e(dm)
    if expected is None:
        return hidden_states, topk_weights, topk_idx
    row0 = hidden_states.shape[0]
    if row0 >= expected:
        return hidden_states, topk_weights, topk_idx
    if row0 <= 0:
        return hidden_states, topk_weights, topk_idx
    pad_rows = expected - row0
    # Ghost rows: duplicate last token hidden + routing; zero gate weights so
    # they contribute nothing (avoids mass-routing padded rows to expert 0).
    pad_hs = hidden_states[-1:].expand(pad_rows, hidden_states.shape[1]).clone()
    hidden_states = torch.cat([hidden_states, pad_hs], dim=0)
    if compute_gate == 1 and topk_idx is not None and topk_weights is not None:
        k = topk_idx.shape[1]
        pad_ids = topk_idx[-1:].expand(pad_rows, k).clone()
        pad_w = torch.zeros(
            (pad_rows, k), dtype=topk_weights.dtype, device=topk_weights.device)
        topk_idx = torch.cat([topk_idx, pad_ids], dim=0)
        topk_weights = torch.cat([topk_weights, pad_w], dim=0)
    return hidden_states, topk_weights, topk_idx


def _get_group_ep(ubatch_idx: int, hccl_comm_name: str, hccl_comm_name2: str, hccl_comm_name3: Optional[str]) -> str:
    groupEp = hccl_comm_name
    if ubatch_idx == 1:
        groupEp = hccl_comm_name2
    elif ubatch_idx == 2:
        assert hccl_comm_name3 is not None
        groupEp = hccl_comm_name3
    return groupEp


class CAMP2PAFDConnector(AFDConnectorBase):
    def __init__(self,
                 rank: int,
                 local_rank: int,
                 config: "VllmConfig"
                 ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self._initialized = False
        self.config = config
        self.hf_config = config.model_config.hf_config
        self.scheduler_config = config.scheduler_config
        decode_max_num_seqs = getattr(self.scheduler_config,
                                      'decode_max_num_seqs', 0)
        self.max_num_reqs = max(self.scheduler_config.max_num_seqs,
                                decode_max_num_seqs)
        self.attn_size = 0
        self.ffn_size = 0
        self.quant_mode = 0
        self.use_aclgraph = self._use_aclgraph()
        self.hccl_comm_name1 = ""
        self.dst_list = []
        ascend_config = get_ascend_config()
        self.mix_placement = getattr(ascend_config, "mix_placement", False)
        self.num_logical_experts = self.hf_config.n_routed_experts
        self.num_shared_experts = self.hf_config.n_shared_experts

    def _use_aclgraph(self) -> bool:
        return self.config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE and \
               self.config.compilation_config.mode == CompilationMode.VLLM_COMPILE and \
               not self.config.model_config.enforce_eager

    def close(self) -> None:
        """Close the connector and release resources."""
        # destroy process group
        pass

    def init_afd_connector(self) -> None:
        """Initialize the AFD connector."""
        afd_size = self.config.afd_config.afd_extra_config.get("afd_size")
        role = self.config.afd_config.afd_role
        self.attn_size, self.ffn_size = map(
            int,
            re.match(r"(\d+)\D+(\d+)", afd_size).groups())

        self.min_size = min(self.ffn_size, self.attn_size)
        self.ratio = self.attn_size // self.ffn_size  # attn_size / ffn_size, for asymmetric A/F
        world_rank = self.rank + self.ffn_size if role == "attention" else self.rank
        # p2p_rank: 所有FFN [0, ffn_size), 前min_size个Attention [ffn_size, ffn_size+min_size)
        self.p2p_rank = self.rank + self.min_size if role == "attention" else self.rank
        self.rank = world_rank

        self.afd_pg_list = []
        self.hccl_comm_name_list = []
        num_ubatches = self.config.parallel_config.num_ubatches if self.config.parallel_config.num_ubatches else 1
        for i in range(num_ubatches):
            group_name = "afd" + str(i) if i > 0 else "afd"
            afd_pg = init_afd_process_group(
                backend="hccl",
                init_method=(
                    f"tcp://{self.config.afd_config.afd_host}"
                    f":{self.config.afd_config.afd_port}"
                ),
                world_size=self.ffn_size + self.attn_size,
                rank=self.rank,
                group_name=group_name
            )
            self.afd_pg_list.append(afd_pg)
            self.hccl_comm_name_list.append(afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank))
        self.hccl_comm_name = self.hccl_comm_name_list[0]
        self.hccl_comm_name2 = self.hccl_comm_name_list[1] if num_ubatches > 1 else self.hccl_comm_name
        self.hccl_comm_name3 = self.hccl_comm_name_list[2] if num_ubatches > 2 else None

        if self.rank < self.ffn_size:
            self.afd_pg1 = init_afd_process_group(
                backend="hccl",
                init_method=(
                    f"tcp://{self.config.afd_config.afd_host}"
                    f":{self.config.afd_config.afd_port}"
                ),
                world_size=self.ffn_size,
                rank=self.rank,
                group_name="afd_moe"
            )
            self.hccl_comm_name1 = self.afd_pg1._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank)

        # 所有FFN和前min_size的Attention参与p2p通信
        # 所有FFN: world_rank in [0, ffn_size), 前min_size个Attention: world_rank in [ffn_size, ffn_size+min_size)
        import datetime
        timeout = datetime.timedelta(seconds=30000)
        if self.is_vaild_rank_for_inequal_AF(self.rank):
            self.p2p_pg = init_afd_process_group(
                backend="gloo",
                init_method=(
                    f"tcp://{self.config.afd_config.afd_host}"
                    f":{self.config.afd_config.afd_port}"
                ),
                world_size=self.ffn_size + self.min_size,
                rank=self.p2p_rank,
                group_name="p2p",
                timeout=timeout  # TODO(yxj):use timeout set
            )

        # 前min_size的Attention向多个FFN发送metadata（1对多映射）
        # attn_i 向所有 ffn_j (其中 j % min_size == i) 发送
        if self.is_attn_top_min_size_rank(self.rank):
            local_attn_rank = self.rank - self.ffn_size
            dst = local_attn_rank
            while dst < self.ffn_size:
                self.dst_list.append(dst)
                dst += self.min_size

        if self.config.afd_config.is_attention_server:
            self.aiv_num = self.config.afd_config.attn_core_num if self.config.afd_config.is_attn_multistream else 8
        else:
            self.aiv_num = self.config.afd_config.ffn_core_num if self.config.afd_config.is_ffn_multistream else 8

        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if the connector is initialized and ready to use.

        Returns:
            bool: True if the connector is initialized, False otherwise.
        """
        return self._initialized

    def configure_metadata(self, metadata: "AFDConnectorMetadata", **kwargs) -> None:
        if metadata.connector_data is None:
            metadata.connector_data = CAMP2PAFDConnectorMetadata()

        config = kwargs.get('config')
        batch_size = kwargs.get('batch_size')
        if self.mix_placement:
            k = self.hf_config.num_experts_per_tok + self.num_shared_experts
        else:
            k = self.hf_config.num_experts_per_tok
        if config:
            metadata.connector_data.moe_expert_num = config.n_routed_experts
            # TODO: quant_mode and aiv_num read from config
            metadata.connector_data.quant_mode = 0
            metadata.connector_data.aiv_num = self.aiv_num
            metadata.connector_data.scale = None
            metadata.connector_data.batch_size = batch_size
            metadata.connector_data.h = config.hidden_size
            metadata.connector_data.k = k

    def select_experts(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            top_k: int,
            use_grouped_topk: bool,
            renormalize: bool,
            topk_group: Optional[int] = None,
            num_expert_group: Optional[int] = None,
            custom_routing_function: Optional[Any] = None,
            routed_scaling_factor=1.0,
            e_score_correction_bias: Optional[torch.Tensor] = None,
            mix_placement: Optional[bool] = False,
            num_logical_experts: int = -1,
            num_shared_experts: int = 0,
            global_num_experts: int = -1,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.cam_select_experts(
            hidden_states,
            router_logits,
            top_k,
            use_grouped_topk,
            renormalize,
            topk_group,
            num_expert_group,
            float(routed_scaling_factor),
            e_score_correction_bias,
            bool(mix_placement),
            num_logical_experts,
            num_shared_experts,
            global_num_experts
        )

    def compute_moe(self, experts, hidden_states, **kwargs):
        group_list = kwargs.get('group_list')
        dynamic_scales = kwargs.get('dynamic_scales')
        topk_weights = kwargs.get('topk_weights')
        topk_ids = kwargs.get('topk_ids')
        x_active_mask = kwargs.get('x_active_mask')
        cam_p2p_ep_name = kwargs.get('cam_p2p_ep_name')

        return experts.afd_m2n_ffn_compute(
            layer=experts,
            hidden_states=hidden_states,
            group_list=group_list,
            dynamic_scale=dynamic_scales,
            connector_name="camp2pconnector",
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            x_active_mask=x_active_mask,
            cam_p2p_ep_name=cam_p2p_ep_name,
            layer_idx=kwargs.get("layer_idx"),
        )

    # ATTN发给MOE（ATTN发送）
    # TODO:metadata的获取，最好从框架侧去拿
    def send_attn_output(self,
                         hidden_states: torch.Tensor,
                         metadata: AFDConnectorMetadata,
                         **kwargs) -> Any:
        # Get from kwargs
        topk_weights = kwargs.get('topk_weights')
        topk_idx = kwargs.get('topk_ids')

        if metadata.connector_data:
            get_forward_context().cam_afdconnector_data = metadata.connector_data

        if self.mix_placement:
            k = self.hf_config.num_experts_per_tok + self.num_shared_experts
            moe_expert_num = self.hf_config.n_routed_experts + self.num_shared_experts
        else:
            k = self.hf_config.num_experts_per_tok
            moe_expert_num = self.hf_config.n_routed_experts

        multistream_enable = False if metadata.layer_idx < self.hf_config.first_k_dense_replace else self.config.afd_config.is_attn_multistream
        if metadata.layer_idx < self.hf_config.first_k_dense_replace:
            compute_gate = 0
        else:
            compute_gate = 1 if getattr(self.config.afd_config, 'compute_gate_on_attention', True) else 0
        # Padding is only for a2e wire format; return unpadded tensors so residual /
        # next-layer hidden shapes stay consistent (see maybe_chunk_residual).
        hs_ret, tw_ret, tid_ret = hidden_states, topk_weights, topk_idx
        hs_send, tw_send, tid_send = _pad_attn_tensors_to_dp_metadata(
            hidden_states, topk_weights, topk_idx, compute_gate)
        _barrier_attention_dp_before_cam_send()
        torch.ops.vllm.cam_send_attn_output(hs_send, tw_send, tid_send,
                                            self.hccl_comm_name,
                                            self.hccl_comm_name2,
                                            self.hccl_comm_name3,
                                            self.rank,
                                            self.ffn_size,
                                            self.attn_size,
                                            moe_expert_num,
                                            self.max_num_reqs,
                                            self.hf_config.hidden_size,
                                            k,
                                            multistream_enable,
                                            self.aiv_num,
                                            compute_gate)
        return hs_ret, None

    # MOE发给ATTN（ATTN接收）
    def recv_ffn_output(self,
                        hidden_states: Optional[torch.Tensor] = None,
                        metadata: Optional["AFDConnectorMetadata"] = None) -> torch.Tensor:
        return torch.ops.vllm.cam_recv_ffn_output(hidden_states,
                                                  self.hccl_comm_name,
                                                  self.hccl_comm_name2,
                                                  self.hccl_comm_name3,
                                                  self.rank,
                                                  self.ffn_size,
                                                  self.attn_size,
                                                  self.config.afd_config.is_attn_multistream)

    # MOE发给ATTN(MOE发送)
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: CAMP2PAFDConnectorMetadata, **kwargs):
        ubatch_idx = kwargs.get('ubatch_idx', 0)
        multistream_enable = kwargs.get('multistream_enable', False)
        comm_stream = kwargs.get('comm_stream', None)
        comm_event = kwargs.get('comm_event', None)

        batch_size = metadata.batch_size
        h = metadata.h
        k = metadata.k
        moe_expert_num = metadata.moe_expert_num
        shared_expert_num = metadata.shared_expert_num
        aiv_num = metadata.aiv_num
        handle = metadata.handle

        groupEp = _get_group_ep(ubatch_idx, self.hccl_comm_name, self.hccl_comm_name2, self.hccl_comm_name3)

        curr_stream = torch.npu.current_stream()
        with npu_stream_switch_within_graph(curr_stream, comm_stream, multistream_enable):
            torch.ops._C_ascend.e2a(expand_x=ffn_output, atten_batch_size=handle[4],
                                      batch_size=batch_size, hidden_size=h, topk=k,
                                      expert_rank_size=self.ffn_size, attention_rank_size=self.attn_size,
                                      rank=self.rank, group_ep=groupEp,
                                      aiv_num=aiv_num)
            if multistream_enable and comm_event is not None:
                comm_event.record(comm_stream)

        return

    # ATTN发给MOE(MOE接收)
    def recv_attn_output(self, metadata: Optional[Any] = None, **kwargs) -> Any:
        ubatch_idx = kwargs.get('ubatch_idx', 0)
        afdmetadata = None

        batch_size = metadata.batch_size
        h = metadata.h
        k = metadata.k
        aiv_num = metadata.aiv_num
        
        if hasattr(metadata, 'layer_idx') and metadata.layer_idx < self.hf_config.first_k_dense_replace:
            compute_gate = 0
        else:
            compute_gate = 1 if getattr(self.config.afd_config, 'compute_gate_on_attention', True) else 0

        groupEp = _get_group_ep(ubatch_idx, self.hccl_comm_name, self.hccl_comm_name2, self.hccl_comm_name3)
        outputs = torch.ops._C_ascend.a2e(x=torch.tensor([], dtype=torch.bfloat16, device='npu'),
                                                expert_ids=torch.tensor([], dtype=torch.int32, device='npu'),
                                                scales=torch.tensor([], dtype=torch.float, device='npu'),
                                                batch_size=batch_size, hidden_size=h, topk=k,
                                                expert_rank_size=self.ffn_size, attention_rank_size=self.attn_size,
                                                rank=self.rank, group_ep=groupEp,
                                                aiv_num=aiv_num,
                                                compute_gate=compute_gate)

        # outputs: [hidden_states1, simulateExpertIds, simulateExpertScales, attenBatchSize, xActiveMaskOut]
        from vllm.distributed.afd_transfer.afd_connector.metadata import AFDRecvOutput
        out_topk_ids = outputs[1] if compute_gate == 1 else None
        out_topk_w = outputs[2] if compute_gate == 1 else None
        out_mask = outputs[4]
        return AFDRecvOutput(
            hidden_states=outputs[0],
            metadata=afdmetadata,
            topk_ids=out_topk_ids,  # simulateExpertIdss
            topk_weights=out_topk_w,  # simulateExpertScales
            atten_batch_size=outputs[3],
            x_active_mask=out_mask,
            cam_p2p_ep_name=self.hccl_comm_name1
        )

    def is_vaild_rank_for_inequal_AF(self, rank):
        # Only support ffn rank < attn rank
        return ((rank >= self.ffn_size and rank < self.ffn_size + self.min_size) or rank < self.ffn_size)

    def is_attn_top_min_size_rank(self, rank):
        # Only support ffn rank < attn rank
        return (rank >= self.ffn_size and rank < self.ffn_size + self.min_size)

    def send_is_ubatch(self, data):
        for dst in self.dst_list:
            object_bytes = pickle.dumps(data)
            object_tensor_cpu = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)

            object_tensor_npu = torch.empty(object_tensor_cpu.shape,
                                            dtype=torch.uint8,
                                            device="cpu")
            object_tensor_npu.copy_(object_tensor_cpu)

            size_tensor = torch.tensor([object_tensor_cpu.numel()],
                                       dtype=torch.long,
                                       device="cpu")

            torch.distributed.send(size_tensor, dst=dst, group=self.p2p_pg)
            torch.distributed.send(object_tensor_npu, dst=dst, group=self.p2p_pg)

    def recv_is_ubatch(self):
        src = self.p2p_rank % self.min_size + self.ffn_size

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
        rank_size = torch.distributed.recv(size_tensor, src=src, group=self.p2p_pg)
        object_tensor_npu = torch.empty(size_tensor.item(), dtype=torch.uint8, device="cpu")
        rank_object = torch.distributed.recv(object_tensor_npu, src=src, group=self.p2p_pg)

        assert rank_object == rank_size, "Received object sender rank does not match the size sender rank."

        object_tensor_cpu = object_tensor_npu.cpu()
        data = pickle.loads(object_tensor_cpu.numpy().tobytes())
        return data

    def create_recv_metadata(self, **kwargs):
        # 从 kwargs 获取 dp_metadata_list 和 ubatch_idx
        dp_metadata_list = kwargs.get('dp_metadata_list')
        ubatch_idx = kwargs.get('ubatch_idx', 0)
        layer_idx = kwargs.get('layer_idx', 0)

        # 从 dp_metadata_list 和 ubatch_idx 获取 max_num_tokens
        if dp_metadata_list is not None and ubatch_idx in dp_metadata_list:
            dp_metadata = dp_metadata_list[ubatch_idx]
            num_tokens_across_dp = dp_metadata.num_tokens_across_dp_cpu.tolist()

            # 计算 max_num_tokens：根据 A > F 且 A 是 F 的倍数的场景
            # 例如 4A2F：第一个F取前两个之和，第二个F取后两个之和
            # TODO(jcz): 需要补不对称场景的计算逻辑
            if self.attn_size >= self.ffn_size and self.attn_size % self.ffn_size == 0:
                # 每个 FFN 处理 group_size 个 Attention 的数据
                group_size = self.attn_size // self.ffn_size
                start_idx = self.rank * group_size
                end_idx = start_idx + group_size
                n_dp_slots = len(num_tokens_across_dp)
                if end_idx > n_dp_slots:
                    logger.warning(
                        "[AFD CAMP2P create_recv_metadata] num_tokens slice OOB: "
                        "layer_idx=%s ubatch_idx=%s rank=%s start_idx=%s end_idx=%s "
                        "len(num_tokens_across_dp)=%s num_tokens_across_dp=%s "
                        "attn_size=%s ffn_size=%s group_size=%s dp_metadata_list_keys=%s",
                        layer_idx,
                        ubatch_idx,
                        self.rank,
                        start_idx,
                        end_idx,
                        n_dp_slots,
                        num_tokens_across_dp,
                        self.attn_size,
                        self.ffn_size,
                        group_size,
                        list(dp_metadata_list.keys())
                        if dp_metadata_list is not None else None,
                    )
                slice_vals = num_tokens_across_dp[start_idx:end_idx]
                max_num_tokens = sum(slice_vals)
                if max_num_tokens == 0:
                    logger.warning(
                        "[AFD CAMP2P create_recv_metadata] max_num_tokens==0 from "
                        "num_tokens slice: layer_idx=%s ubatch_idx=%s rank=%s "
                        "start_idx=%s end_idx=%s slice=%s len(dp)=%s end_oob=%s dp_keys=%s",
                        layer_idx,
                        ubatch_idx,
                        self.rank,
                        start_idx,
                        end_idx,
                        slice_vals,
                        n_dp_slots,
                        end_idx > n_dp_slots,
                        list(dp_metadata_list.keys())
                        if dp_metadata_list is not None else None,
                    )
                diag = os.getenv("VLLM_ASCEND_AFD_MOE_INDEX_DIAG", "0") == "1"
                if diag and 2 <= layer_idx <= 5:
                    logger.info(
                        "[AFD CAMP2P create_recv_metadata] layer_idx=%s ubatch_idx=%s "
                        "rank=%s group_size=%s start_idx=%s end_idx=%s "
                        "max_num_tokens=%s num_tokens_across_dp=%s",
                        layer_idx,
                        ubatch_idx,
                        self.rank,
                        group_size,
                        start_idx,
                        end_idx,
                        max_num_tokens,
                        num_tokens_across_dp,
                    )
                print(f"rank {self.rank} get max_num_tokens {max_num_tokens} from dp_metadata_list with group_size {group_size}")
            else:
                max_num_tokens = kwargs.get('max_num_tokens', 0)
                print(f"rank {self.rank} get max_num_tokens {max_num_tokens} from kwargs due to attn_size {self.attn_size} and ffn_size {self.ffn_size}")
        else:
            max_num_tokens = kwargs.get('max_num_tokens', 0)
            if dp_metadata_list is not None:
                logger.warning(
                    "[AFD CAMP2P create_recv_metadata] ubatch_idx not in dp_metadata_list "
                    "(falling back to kwargs max_num_tokens=%s): layer_idx=%s ubatch_idx=%s "
                    "rank=%s keys=%s",
                    max_num_tokens,
                    layer_idx,
                    ubatch_idx,
                    self.rank,
                    list(dp_metadata_list.keys()),
                )

        hf_config = self.config.model_config.hf_config

        if self.mix_placement:
            k = self.hf_config.num_experts_per_tok + self.num_shared_experts
        else:
            k = self.hf_config.num_experts_per_tok

        metadata = CAMP2PAFDConnectorMetadata(
            moe_expert_num=hf_config.n_routed_experts,
            shared_expert_num=0,
            scale=None,
            handle=None,
            quant_mode=0,
            aiv_num=self.aiv_num,
            batch_size=max_num_tokens,
            h=hf_config.hidden_size,
            k=k
        )
        metadata.layer_idx = layer_idx
        return metadata

    def update_metadata(self, metadata, recv_output):
        metadata.handle = [
            recv_output.topk_ids,
            recv_output.topk_weights,
            recv_output.expand_idx,
            recv_output.ep_recv_counts,
            recv_output.atten_batch_size
        ]

    def send_dp_metadata_list(
        self,
        data,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ):
        """发送dp_metadata_list给对应的FFN rank

        Args:
            data: dp_metadata_list字典
            is_graph_capturing: 是否处于graph capture阶段
            is_warmup: 是否处于warmup阶段
        """
        send_data = (data, is_graph_capturing, is_warmup)

        for dst in self.dst_list:
            object_bytes = pickle.dumps(send_data)
            object_tensor_cpu = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)

            object_tensor_npu = torch.empty(object_tensor_cpu.shape,
                                            dtype=torch.uint8,
                                            device="cpu")
            object_tensor_npu.copy_(object_tensor_cpu)

            size_tensor = torch.tensor([object_tensor_cpu.numel()],
                                       dtype=torch.long,
                                       device="cpu")

            torch.distributed.send(size_tensor, dst=dst, group=self.p2p_pg)
            torch.distributed.send(object_tensor_npu, dst=dst, group=self.p2p_pg)

    def recv_dp_metadata_list(self):
        """接收dp_metadata_list

        Returns:
            tuple: (data, is_graph_capturing, is_warmup)
        """
        src = self.p2p_rank % self.min_size + self.ffn_size

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
        rank_size = torch.distributed.recv(size_tensor, src=src, group=self.p2p_pg)

        object_tensor_npu = torch.empty(size_tensor.item(), dtype=torch.uint8, device="cpu")
        rank_object = torch.distributed.recv(object_tensor_npu, src=src, group=self.p2p_pg)

        assert rank_object == rank_size, \
            "Received object sender rank does not match the size sender rank."

        object_tensor_cpu = object_tensor_npu.cpu()
        obj = pickle.loads(object_tensor_cpu.numpy().tobytes())

        if len(obj) == 3:
            data, is_graph_capturing, is_warmup = obj
        else:
            # 兼容旧格式
            data, is_graph_capturing = obj
            is_warmup = False

        return data, is_graph_capturing, is_warmup

    def update_state_from_dp_metadata(
        self,
        dp_metadata_list: dict,
        is_graph_capturing: bool = False,
    ):
        """更新connector状态

        Args:
            dp_metadata_list: dp_metadata_list字典
            is_graph_capturing: 是否处于graph capture阶段
        """
        self.dp_metadata_list = dp_metadata_list
        self.is_graph_capturing = is_graph_capturing


def cam_select_experts_impl(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int],
        num_expert_group: Optional[int],
        routed_scaling_factor: float,
        e_score_correction_bias: Optional[torch.Tensor],
        mix_placement: bool,
        num_logical_experts: int,
        num_shared_experts: int,
        global_num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=None,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        mix_placement=mix_placement,
        num_logical_experts=num_logical_experts,
        num_shared_experts=num_shared_experts,
        global_num_experts=global_num_experts,
    )


def cam_select_experts_fake_impl(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int],
        num_expert_group: Optional[int],
        routed_scaling_factor: float,
        e_score_correction_bias: Optional[torch.Tensor],
        mix_placement: bool,
        num_logical_experts: int,
        num_shared_experts: int,
        global_num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = router_logits.shape[0]
    out_k = top_k
    if mix_placement:
        out_k += num_shared_experts
    
    topk_weights = torch.empty((num_tokens, out_k), dtype=hidden_states.dtype, device=hidden_states.device)
    topk_ids = torch.empty((num_tokens, out_k), dtype=torch.int32, device=hidden_states.device)
    return topk_weights, topk_ids


def cam_send_attn_output_impl(hidden_states: torch.Tensor,
                              topk_weights: Optional[torch.Tensor],
                              topk_idx: Optional[torch.Tensor],
                              hccl_comm_name: str,
                              hccl_comm_name2: str,
                              hccl_comm_name3: Optional[str],
                              rank: int,
                              ffn_size: int,
                              attn_size: int,
                              moe_expert_num: int,
                              batch_size: int,
                              h: int,
                              k: int,
                              multistream_enable: bool,
                              aiv_num: int,
                              compute_gate: int = 1) -> torch.Tensor:
    ubatch_idx = get_forward_context().ubatch_idx
    comm_stream = get_forward_context().afd_comm_stream
    comm_event = get_forward_context().afd_comm_event
    if get_forward_context().cam_afdconnector_data is None:
        cam_afdconnector_data = CAMP2PAFDConnectorMetadata(
            moe_expert_num=moe_expert_num,
            shared_expert_num=0,
            scale=None,
            handle=None,
            quant_mode=0,
            aiv_num=aiv_num,
            batch_size=batch_size,
            h=h,
            k=k
        )
        get_forward_context().cam_afdconnector_data = cam_afdconnector_data

    cam_metadata = get_forward_context().cam_afdconnector_data
    batch_size = cam_metadata.batch_size
    h = cam_metadata.h
    k = cam_metadata.k
    aiv_num = cam_metadata.aiv_num

    groupEp = _get_group_ep(ubatch_idx, hccl_comm_name, hccl_comm_name2, hccl_comm_name3)

    curr_stream = torch.npu.current_stream()
    with npu_stream_switch_within_graph(curr_stream, comm_stream, multistream_enable):
        handle_out = torch.ops._C_ascend.a2e(x=hidden_states, expert_ids=topk_idx,
                                                scales=topk_weights,
                                                batch_size=batch_size, hidden_size=h, topk=k,
                                                expert_rank_size=ffn_size, attention_rank_size=attn_size,
                                                rank=rank, group_ep=groupEp,
                                                aiv_num=aiv_num,
                                                compute_gate=compute_gate)

        hidden_states1, simulateExpertIds, simulateExpertScales, attenBatchSize, xActiveMaskOut = handle_out[0:5]
        handle = [hidden_states1, simulateExpertIds, simulateExpertScales, attenBatchSize]
        cam_metadata.handle = handle
        get_forward_context().cam_afdconnector_data = cam_metadata
        if multistream_enable:
            comm_event.record(comm_stream)
    return hidden_states


def cam_send_attn_output_fake_impl(hidden_states: torch.Tensor,
                                   topk_weights: Optional[torch.Tensor],
                                   topk_idx: Optional[torch.Tensor],
                                   hccl_comm_name: str,
                                   hccl_comm_name2: str,
                                   hccl_comm_name3: Optional[str],
                                   rank: int,
                                   ffn_size: int,
                                   attn_size: int,
                                   moe_expert_num: int,
                                   batch_size: int,
                                   h: int,
                                   k: int,
                                   multistream_enable: bool,
                                   aiv_num: int,
                                   compute_gate: int = 1) -> torch.Tensor:
    return hidden_states


def cam_recv_ffn_output_impl(hidden_states: torch.Tensor,
                             hccl_comm_name: str,
                             hccl_comm_name2: str,
                             hccl_comm_name3: Optional[str],
                             rank: int,
                             ffn_size: int,
                             attn_size: int,
                             multistream_enable: bool) -> torch.Tensor:
    cam_metadata = get_forward_context().cam_afdconnector_data
    assert cam_metadata is not None, "cam_metadata is None"
    ubatch_idx = get_forward_context().ubatch_idx
    comm_event = get_forward_context().afd_comm_event
    batch_size = cam_metadata.batch_size
    h = cam_metadata.h
    k = cam_metadata.k
    aiv_num = cam_metadata.aiv_num
    handle = cam_metadata.handle

    groupEp = _get_group_ep(ubatch_idx, hccl_comm_name, hccl_comm_name2, hccl_comm_name3)

    if multistream_enable:
        curr_stream = torch.npu.current_stream()
        comm_event.wait(curr_stream)
    output2 = torch.ops._C_ascend.e2a(expand_x=hidden_states, atten_batch_size=handle[3],
                                            batch_size=batch_size, hidden_size=h, topk=k,
                                            expert_rank_size=ffn_size, attention_rank_size=attn_size,
                                            rank=rank, group_ep=groupEp,
                                            aiv_num=aiv_num)
    return output2


def cam_recv_ffn_output_fake_impl(hidden_states: torch.Tensor,
                                  hccl_comm_name: str,
                                  hccl_comm_name2: str,
                                  hccl_comm_name3: Optional[str],
                                  rank: int,
                                  ffn_size: int,
                                  attn_size: int,
                                  multistream_enable: bool) -> torch.Tensor:
    return hidden_states


direct_register_custom_op(op_name="cam_select_experts",
                          op_func=cam_select_experts_impl,
                          fake_impl=cam_select_experts_fake_impl,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="cam_send_attn_output",
                          op_func=cam_send_attn_output_impl,
                          fake_impl=cam_send_attn_output_fake_impl,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="cam_recv_ffn_output",
                          op_func=cam_recv_ffn_output_impl,
                          fake_impl=cam_recv_ffn_output_fake_impl,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")