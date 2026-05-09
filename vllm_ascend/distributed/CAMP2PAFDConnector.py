from dataclasses import dataclass
from typing import Any, Optional

from vllm.distributed.afd_transfer.afd_connector import (AFDConnectorBase, AFDConnectorFactory,
                                                         AFDConnectorMetadata)

__all__ = ["AFDConnectorBase", "AFDConnectorMetadata", "AFDConnectorFactory"]

import torch_npu
import torch
import pickle

from torch.distributed.distributed_c10d import _get_default_group
import re

import torch
from torch.distributed.distributed_c10d import _update_default_pg, _get_default_group

from vllm.distributed.parallel_state import (get_ep_group,
                                              init_afd_process_group,
                                              init_model_parallel_group)
from vllm_ascend.distributed.metadata import (CAMP2PAFDConnectorMetadata)
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ascend_config import get_ascend_config
from vllm.config import VllmConfig, CUDAGraphMode, CompilationMode
from vllm.distributed.afd_transfer.afd_connector.p2p_connector import DefaultProcessGroupSwitcher

from vllm.utils.torch_utils import direct_register_custom_op
from vllm.forward_context import ForwardContext, get_forward_context

from vllm_ascend.utils import npu_stream_switch_within_graph
import vllm_ascend.envs as envs_ascend
from vllm.logger import init_logger

logger = init_logger(__name__)


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


def _npu_stream_in_graph_capture() -> bool:
    """True if the current NPU stream is in ACL/NPUGraph capture (D2H/sync is illegal → 107027)."""
    for name in (
            "is_current_stream_capturing",
            "_is_in_graph_capture",
            "is_in_graph_capture",
    ):
        try:
            fn = getattr(torch.npu, name, None)
            if callable(fn) and bool(fn()):
                return True
        except Exception:
            continue
    return False


def _afd_a2e_wire_diag_allow_d2h() -> bool:
    if not envs_ascend.VLLM_ASCEND_AFD_WIRE_DIAG_D2H:
        return False
    if _npu_stream_in_graph_capture():
        return False
    return True


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


def _get_group_ep(ubatch_idx: int, hccl_comm_name: str, hccl_comm_name2: str, hccl_comm_name3: Optional[str]) -> str:
    groupEp = hccl_comm_name
    if ubatch_idx == 1:
        groupEp = hccl_comm_name2
    elif ubatch_idx == 2:
        assert hccl_comm_name3 is not None
        groupEp = hccl_comm_name3
    return groupEp


def _summarize_dp_metadata_for_wire_diag(ctx: ForwardContext | None) -> str:
    if ctx is None:
        return "ctx=none"
    dm = _active_dp_metadata_from_forward_ctx(ctx)
    if dm is None:
        return "dp_meta=none"
    try:
        nta = dm.num_tokens_across_dp_cpu.tolist()
        mx = int(dm.max_tokens_across_dp_cpu.item())
        exp = getattr(ctx, "afd_expected_a2e_rows", None)
        nt = getattr(ctx, "num_tokens", None)
        ubx = getattr(ctx, "ubatch_idx", None)
        return (
            f"ubatch_idx={ubx} nta={nta} max={mx} "
            f"afd_expected_a2e_rows={exp} num_tokens={nt}"
        )
    except Exception as ex:
        return f"dp_meta_err={ex}"


def _tensor_shape(t: Any) -> Any:
    if t is None:
        return None
    try:
        return tuple(t.shape)
    except Exception:
        return "?"


def _tensor_dtype_str(t: Any) -> Any:
    if t is None:
        return None
    try:
        return str(t.dtype)
    except Exception:
        return "?"


def _tensor_data_ptr(t: Any) -> int:
    if t is None:
        return 0
    try:
        return int(t.data_ptr())
    except Exception:
        return 0


def _log_afd_wire_send_diag(
        *,
        attn_world_rank: int,
        p2p_rank: int,
        layer_idx: int,
        compute_gate: int,
        hidden_size: int,
        topk: int,
        moe_expert_num: int,
        max_num_reqs: int,
        aiv_num: int,
        multistream_enable: bool,
        group_ep: str,
        wire_ubatch_idx: int,
        afd_ffn_size: int,
        afd_attn_size: int,
        afd_min_size: int,
        hs_ret: torch.Tensor,
        hs_send: torch.Tensor,
        tw_send: Optional[torch.Tensor],
        tid_send: Optional[torch.Tensor],
        connector_batch_size: Optional[int] = None,
) -> None:
    if not envs_ascend.VLLM_ASCEND_AFD_WIRE_DIAG:
        return
    if _in_torch_compile_trace():
        return
    ctx = get_forward_context()
    dp_s = _summarize_dp_metadata_for_wire_diag(ctx)
    logger.info(
        "[AFD-WIRE-SEND-HOST] attn_world_rank=%s p2p_rank=%s wire_ubatch=%s layer=%s compute_gate=%s "
        "meta.h=%s meta.k=%s moe_expert_num=%s max_num_reqs=%s aiv_num=%s multistream=%s "
        "group_ep=%s afd_ffn_size=%s afd_attn_size=%s afd_min_size=%s connector_batch_size=%s %s "
        "hs_ret=%s hs_send=%s tw_send=%s tid_send=%s "
        "ptrs=[hs_ret=0x%x,hs_send=0x%x,tw=0x%x,tid=0x%x]",
        attn_world_rank,
        p2p_rank,
        wire_ubatch_idx,
        layer_idx,
        compute_gate,
        hidden_size,
        topk,
        moe_expert_num,
        max_num_reqs,
        aiv_num,
        multistream_enable,
        group_ep,
        afd_ffn_size,
        afd_attn_size,
        afd_min_size,
        connector_batch_size,
        dp_s,
        tuple(hs_ret.shape),
        tuple(hs_send.shape),
        tuple(tw_send.shape) if tw_send is not None else None,
        tuple(tid_send.shape) if tid_send is not None else None,
        _tensor_data_ptr(hs_ret),
        _tensor_data_ptr(hs_send),
        _tensor_data_ptr(tw_send),
        _tensor_data_ptr(tid_send),
    )


def _log_afd_a2e_recv_diag_host(
        *,
        ffn_rank: int,
        metadata: Any,
        compute_gate: int,
        outputs: list,
        ubatch_idx: int,
        group_ep: str,
) -> None:
    """Shapes / metadata only — no NPU→CPU reads (survives 507015 blowups in later D2H)."""
    if not envs_ascend.VLLM_ASCEND_AFD_WIRE_DIAG:
        return
    if _in_torch_compile_trace():
        return
    lyr = getattr(metadata, "layer_idx", "?")
    bs = getattr(metadata, "batch_size", "?")
    mh = getattr(metadata, "h", "?")
    mk = getattr(metadata, "k", "?")
    ep_ig = None
    try:
        ep_ig = get_ep_group().rank_in_group
    except Exception:
        pass
    ctx = get_forward_context()
    dp_s = _summarize_dp_metadata_for_wire_diag(ctx)
    o0 = outputs[0] if len(outputs) > 0 else None
    o1 = outputs[1] if len(outputs) > 1 else None
    o2 = outputs[2] if len(outputs) > 2 else None
    o3 = outputs[3] if len(outputs) > 3 else None
    o4 = outputs[4] if len(outputs) > 4 else None
    if compute_gate != 1:
        o1 = o2 = None
    logger.info(
        "[AFD-A2E-RECV-HOST] ffn_rank=%s ep_ig=%s ubatch=%s layer=%s meta.batch_size=%s meta.h=%s "
        "meta.k=%s compute_gate=%s group_ep=%s %s "
        "out0_hid=%s out1_topk=%s out2_tw=%s out3_atten_bs=%s out4_mask=%s "
        "dtypes=[%s,%s,%s,%s,%s] ptrs=[0x%x,0x%x,0x%x,0x%x,0x%x]",
        ffn_rank,
        ep_ig,
        ubatch_idx,
        lyr,
        bs,
        mh,
        mk,
        compute_gate,
        group_ep,
        dp_s,
        _tensor_shape(o0),
        _tensor_shape(o1),
        _tensor_shape(o2),
        _tensor_shape(o3),
        _tensor_shape(o4),
        _tensor_dtype_str(o0),
        _tensor_dtype_str(o1),
        _tensor_dtype_str(o2),
        _tensor_dtype_str(o3),
        _tensor_dtype_str(o4),
        _tensor_data_ptr(o0),
        _tensor_data_ptr(o1),
        _tensor_data_ptr(o2),
        _tensor_data_ptr(o3),
        _tensor_data_ptr(o4),
    )


def _log_afd_a2e_recv_diag_d2h(
        *,
        ffn_rank: int,
        metadata: Any,
        compute_gate: int,
        outputs: list,
        ubatch_idx: int = 0,
) -> None:
    out_h = outputs[0]
    out_tid = outputs[1] if compute_gate == 1 else None
    out_tw = outputs[2] if compute_gate == 1 else None
    out_abs = outputs[3]
    out_mask = outputs[4]
    lyr = getattr(metadata, "layer_idx", "?")
    bs = getattr(metadata, "batch_size", "?")
    tid_mm = None
    if out_tid is not None and out_tid.numel() > 0:
        try:
            tid_mm = (
                int(out_tid.min().detach().cpu().item()),
                int(out_tid.max().detach().cpu().item()),
            )
        except Exception as ex:
            tid_mm = f"<d2h_failed:{ex!r}>"
    mask_sum = mask_nz = None
    mask_err = None
    mask_checks = ""
    mask_numel = hid_rows = -1
    mask_head_i32 = None
    mask_mm_i32 = None
    ep_ig = None
    try:
        ep_ig = get_ep_group().rank_in_group
    except Exception:
        pass
    if out_mask is not None:
        try:
            mask_numel = int(out_mask.numel())
            hid_rows = int(out_h.shape[0]) if out_h.dim() >= 1 else -1
        except Exception:
            mask_numel = hid_rows = -1
        try:
            mask_sum = int(out_mask.to(torch.int64).sum().detach().cpu().item())
            mask_nz = int(torch.count_nonzero(out_mask).detach().cpu().item())
        except Exception as ex:
            mask_err = str(ex)
        issues = []
        if mask_sum is not None:
            if mask_sum < 0:
                issues.append("neg_sum")
            if mask_numel >= 0 and mask_sum > mask_numel:
                issues.append("sum_gt_numel")
        if (
            mask_sum is not None
            and mask_nz is not None
            and mask_sum != mask_nz
        ):
            issues.append("sum_ne_count_nonzero")
        if hid_rows >= 0 and mask_numel >= 0 and mask_numel != hid_rows:
            issues.append(f"numel_ne_hid_rows({mask_numel}!={hid_rows})")
        mask_checks = ",".join(issues) if issues else "ok"
        try:
            flat = out_mask.flatten()
            k = min(8, flat.numel())
            if k > 0:
                f32 = flat[:k].to(torch.int32)
                mask_head_i32 = f32.detach().cpu().tolist()
                mask_mm_i32 = (
                    int(flat.to(torch.int32).min().detach().cpu().item()),
                    int(flat.to(torch.int32).max().detach().cpu().item()),
                )
        except Exception as ex:
            mask_head_i32 = f"<d2h_failed:{ex!r}>"
    abs_info = None
    if out_abs is not None:
        try:
            if out_abs.numel() <= 8:
                abs_info = out_abs.detach().cpu().tolist()
            else:
                abs_info = f"shape={tuple(out_abs.shape)}"
        except Exception as ex:
            abs_info = f"<d2h_failed:{ex!r}>"
    logger.info(
        "[AFD-A2E-RECV] ffn_rank=%s ep_ig=%s ubatch=%s layer=%s metadata.batch_size=%s compute_gate=%s "
        "hid=%s topk_ids=%s topk_minmax=%s tw=%s atten_batch_size=%s "
        "x_active_mask=%s dtype=%s ptr=0x%x mask_numel=%s hid_rows=%s checks=%s "
        "sum_i64=%s count_nonzero=%s mask_i32_minmax=%s head_i32=%s err=%s",
        ffn_rank,
        ep_ig,
        ubatch_idx,
        lyr,
        bs,
        compute_gate,
        tuple(out_h.shape),
        tuple(out_tid.shape) if out_tid is not None else None,
        tid_mm,
        tuple(out_tw.shape) if out_tw is not None else None,
        abs_info,
        tuple(out_mask.shape) if out_mask is not None else None,
        str(out_mask.dtype) if out_mask is not None else None,
        int(out_mask.data_ptr()) if out_mask is not None else 0,
        mask_numel,
        hid_rows,
        mask_checks,
        mask_sum,
        mask_nz,
        mask_mm_i32,
        mask_head_i32,
        mask_err,
    )


def _log_afd_a2e_recv_diag(
        *,
        ffn_rank: int,
        metadata: Any,
        compute_gate: int,
        outputs: list,
        ubatch_idx: int = 0,
        group_ep: str = "",
) -> None:
    if not envs_ascend.VLLM_ASCEND_AFD_WIRE_DIAG:
        return
    if _in_torch_compile_trace():
        return
    _log_afd_a2e_recv_diag_host(
        ffn_rank=ffn_rank,
        metadata=metadata,
        compute_gate=compute_gate,
        outputs=outputs,
        ubatch_idx=ubatch_idx,
        group_ep=group_ep,
    )
    if not _afd_a2e_wire_diag_allow_d2h():
        logger.info(
            "[AFD-A2E-RECV-D2H-SKIP] ffn_rank=%s ubatch=%s layer=%s "
            "reason=d2h_disabled_or_graph_capture(%s,%s)",
            ffn_rank,
            ubatch_idx,
            getattr(metadata, "layer_idx", "?"),
            not envs_ascend.VLLM_ASCEND_AFD_WIRE_DIAG_D2H,
            _npu_stream_in_graph_capture(),
        )
        return
    try:
        _log_afd_a2e_recv_diag_d2h(
            ffn_rank=ffn_rank,
            metadata=metadata,
            compute_gate=compute_gate,
            outputs=outputs,
            ubatch_idx=ubatch_idx,
        )
    except Exception as ex:
        logger.info(
            "[AFD-A2E-RECV-D2H-SKIP] ffn_rank=%s ubatch=%s layer=%s reason=%r",
            ffn_rank,
            ubatch_idx,
            getattr(metadata, "layer_idx", "?"),
            ex,
        )


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
        if metadata.connector_data is not None:
            metadata.connector_data.batch_size = int(hidden_states.shape[0])
            get_forward_context().cam_afdconnector_data = metadata.connector_data
        ubatch_idx = int(kwargs.get("ubatch_idx", 0))
        group_ep = _get_group_ep(
            ubatch_idx,
            self.hccl_comm_name,
            self.hccl_comm_name2,
            self.hccl_comm_name3,
        )
        _log_afd_wire_send_diag(
            attn_world_rank=self.rank,
            p2p_rank=self.p2p_rank,
            layer_idx=metadata.layer_idx,
            compute_gate=compute_gate,
            hidden_size=int(self.hf_config.hidden_size),
            topk=int(k),
            moe_expert_num=int(moe_expert_num),
            max_num_reqs=int(self.max_num_reqs),
            aiv_num=int(self.aiv_num),
            multistream_enable=bool(multistream_enable),
            group_ep=group_ep,
            wire_ubatch_idx=ubatch_idx,
            afd_ffn_size=int(self.ffn_size),
            afd_attn_size=int(self.attn_size),
            afd_min_size=int(self.min_size),
            hs_ret=hidden_states,
            hs_send=hidden_states,
            tw_send=topk_weights,
            tid_send=topk_idx,
            connector_batch_size=int(hidden_states.shape[0]),
        )
        torch.ops.vllm.cam_send_attn_output(hidden_states, topk_weights, topk_idx,
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
        return hidden_states, None

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
        _log_afd_a2e_recv_diag(
            ffn_rank=self.rank,
            metadata=metadata,
            compute_gate=compute_gate,
            outputs=outputs,
            ubatch_idx=ubatch_idx,
            group_ep=groupEp,
        )
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
                max_num_tokens = sum(num_tokens_across_dp[start_idx:end_idx])
                print(f"rank {self.rank} get max_num_tokens {max_num_tokens} from dp_metadata_list with group_size {group_size}")
            else:
                max_num_tokens = kwargs.get('max_num_tokens', 0)
                print(f"rank {self.rank} get max_num_tokens {max_num_tokens} from kwargs due to attn_size {self.attn_size} and ffn_size {self.ffn_size}")
        else:
            max_num_tokens = kwargs.get('max_num_tokens', 0)

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
        """发送dp_metadata_list给对应的FFN rank"""
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
            tuple: (data, is_graph_capturing, is_warmup, afd_num_actual_tokens,
                attn_aclgraph_runtime_mode)
                后两项可为 None（旧 pickle 格式）。
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

        afd_num_actual_tokens = None
        attn_aclgraph_runtime_mode = None
        if len(obj) == 5:
            data, is_graph_capturing, is_warmup, afd_num_actual_tokens, attn_aclgraph_runtime_mode = obj
        elif len(obj) == 4:
            data, is_graph_capturing, is_warmup, afd_num_actual_tokens = obj
        elif len(obj) == 3:
            data, is_graph_capturing, is_warmup = obj
        else:
            # 兼容旧格式
            data, is_graph_capturing = obj
            is_warmup = False

        return (
            data,
            is_graph_capturing,
            is_warmup,
            afd_num_actual_tokens,
            attn_aclgraph_runtime_mode,
        )

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
    if not _npu_stream_in_graph_capture():
        torch.npu.synchronize()
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