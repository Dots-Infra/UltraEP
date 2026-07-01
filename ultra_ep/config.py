from dataclasses import dataclass
from .util import read_bool_env, read_float_env, read_int_env, read_str_env


_WEIGHT_SYNC_PLAN_MODE_IDS = {
    "direct": 0,
    "adaptive": 1,
    "forcerelay": 2,
}


def _normalize_weight_sync_plan_mode(value: str) -> str:
    normalized = value.lower().replace("_", "")
    if normalized not in _WEIGHT_SYNC_PLAN_MODE_IDS:
        raise ValueError(
            "ULTRA_EP_WEIGHT_SYNC_PLAN_MODE must be one of: direct, adaptive, force_relay"
        )
    return normalized


@dataclass(frozen=True)
class UltraEPTuning:
    balance_threshold: float
    quota_locality_aware: bool
    quota_min_tokens_per_replica: int
    quota_allow_zero_master_quota: bool
    grad_reduce_num_sms: int
    grad_reduce_deterministic: bool
    quota_oracle_eps: float
    quota_kernel_stage: int
    quota_reroute_interleave: bool
    weight_sync_plan_mode: str
    weight_sync_plan_mode_id: int
    weight_sync_relay_min_replicas: int
    weight_sync_relay_max_relays: int
    weight_sync_relay_min_fanout_gain: int


def load_tuning_from_env() -> UltraEPTuning:
    weight_sync_plan_mode = _normalize_weight_sync_plan_mode(
        read_str_env("ULTRA_EP_WEIGHT_SYNC_PLAN_MODE", "adaptive")
    )
    grad_reduce_num_sms = read_int_env("ULTRA_EP_GRAD_REDUCE_NUM_SMS", 42)
    if grad_reduce_num_sms <= 0:
        raise ValueError("ULTRA_EP_GRAD_REDUCE_NUM_SMS must be positive")
    if grad_reduce_num_sms % 2 != 0:
        raise ValueError("ULTRA_EP_GRAD_REDUCE_NUM_SMS must be even")
    grad_reduce_deterministic = read_bool_env(
        "ULTRA_EP_GRAD_REDUCE_DETERMINISTIC", True
    )

    quota_kernel_stage = read_int_env("ULTRA_EP_QUOTA_KERNEL_STAGE", 1)
    if quota_kernel_stage not in (0, 1):
        raise ValueError("ULTRA_EP_QUOTA_KERNEL_STAGE supports only 0 or 1")

    return UltraEPTuning(
        balance_threshold=read_float_env("ULTRA_EP_BALANCE_THRESHOLD", 1.0),
        quota_locality_aware=read_bool_env("ULTRA_EP_QUOTA_LOCALITY_AWARE", True),
        quota_min_tokens_per_replica=read_int_env(
            "ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA", 1024
        ),
        quota_allow_zero_master_quota=read_bool_env(
            "ULTRA_EP_QUOTA_ALLOW_ZERO_MASTER_QUOTA", False
        ),
        grad_reduce_num_sms=grad_reduce_num_sms,
        grad_reduce_deterministic=grad_reduce_deterministic,
        quota_oracle_eps=read_float_env("ULTRA_EP_QUOTA_ORACLE_EPS", 0.01),
        quota_kernel_stage=quota_kernel_stage,
        quota_reroute_interleave=read_bool_env(
            "ULTRA_EP_QUOTA_REROUTE_INTERLEAVE", True
        ),
        weight_sync_plan_mode=weight_sync_plan_mode,
        weight_sync_plan_mode_id=_WEIGHT_SYNC_PLAN_MODE_IDS[weight_sync_plan_mode],
        weight_sync_relay_min_replicas=read_int_env(
            "ULTRA_EP_WEIGHT_SYNC_RELAY_MIN_REPLICAS", 4
        ),
        weight_sync_relay_max_relays=read_int_env(
            "ULTRA_EP_WEIGHT_SYNC_RELAY_MAX_RELAYS", 8
        ),
        weight_sync_relay_min_fanout_gain=read_int_env(
            "ULTRA_EP_WEIGHT_SYNC_RELAY_MIN_FANOUT_GAIN", 2
        ),
    )
