import os
from dataclasses import dataclass


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_WEIGHT_SYNC_PLAN_MODE_IDS = {
    "direct": 0,
    "adaptive": 1,
    "forcerelay": 2,
}


def _read_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _read_bool_env(name: str, default: bool) -> bool:
    value = _read_env(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of: 0/1, true/false, yes/no, on/off")


def _read_int_env(name: str, default: int) -> int:
    value = _read_env(name)
    return default if value is None else int(value)


def _read_float_env(name: str, default: float) -> float:
    value = _read_env(name)
    return default if value is None else float(value)


def _read_str_env(name: str, default: str) -> str:
    value = _read_env(name)
    return default if value is None else value


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
    quota_oracle_eps: float
    quota_kernel_stage: int
    quota_reroute_interleave: bool
    weight_sync_plan_mode: str
    weight_sync_plan_mode_id: int
    weight_sync_relay_min_replicas: int
    weight_sync_relay_max_relays: int
    weight_sync_relay_min_fanout_gain: int
    log_expert_loads: bool
    loads_save_dir: str


def load_tuning_from_env() -> UltraEPTuning:
    weight_sync_plan_mode = _normalize_weight_sync_plan_mode(
        _read_str_env("ULTRA_EP_WEIGHT_SYNC_PLAN_MODE", "adaptive")
    )
    grad_reduce_num_sms = _read_int_env("ULTRA_EP_GRAD_REDUCE_NUM_SMS", 24)
    if grad_reduce_num_sms <= 0:
        raise ValueError("ULTRA_EP_GRAD_REDUCE_NUM_SMS must be positive")
    if grad_reduce_num_sms % 2 != 0:
        raise ValueError("ULTRA_EP_GRAD_REDUCE_NUM_SMS must be even")

    quota_kernel_stage = _read_int_env("ULTRA_EP_QUOTA_KERNEL_STAGE", 1)
    if quota_kernel_stage not in (0, 1):
        raise ValueError("ULTRA_EP_QUOTA_KERNEL_STAGE supports only 0 or 1")

    return UltraEPTuning(
        balance_threshold=_read_float_env("ULTRA_EP_BALANCE_THRESHOLD", 1.0),
        quota_locality_aware=_read_bool_env("ULTRA_EP_QUOTA_LOCALITY_AWARE", True),
        quota_min_tokens_per_replica=_read_int_env(
            "ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA", 1024
        ),
        quota_allow_zero_master_quota=_read_bool_env(
            "ULTRA_EP_QUOTA_ALLOW_ZERO_MASTER_QUOTA", False
        ),
        grad_reduce_num_sms=grad_reduce_num_sms,
        quota_oracle_eps=_read_float_env("ULTRA_EP_QUOTA_ORACLE_EPS", 0.01),
        quota_kernel_stage=quota_kernel_stage,
        quota_reroute_interleave=_read_bool_env(
            "ULTRA_EP_QUOTA_REROUTE_INTERLEAVE", True
        ),
        weight_sync_plan_mode=weight_sync_plan_mode,
        weight_sync_plan_mode_id=_WEIGHT_SYNC_PLAN_MODE_IDS[weight_sync_plan_mode],
        weight_sync_relay_min_replicas=_read_int_env(
            "ULTRA_EP_WEIGHT_SYNC_RELAY_MIN_REPLICAS", 6
        ),
        weight_sync_relay_max_relays=_read_int_env(
            "ULTRA_EP_WEIGHT_SYNC_RELAY_MAX_RELAYS", 8
        ),
        weight_sync_relay_min_fanout_gain=_read_int_env(
            "ULTRA_EP_WEIGHT_SYNC_RELAY_MIN_FANOUT_GAIN", 2
        ),
        log_expert_loads=_read_bool_env("ULTRA_EP_LOG_EXPERT_LOADS", False),
        loads_save_dir=_read_str_env(
            "ULTRA_EP_LOADS_SAVE_DIR", "/var/log/ultra_ep_loads"
        ),
    )
