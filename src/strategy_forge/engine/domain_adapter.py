"""DomainAdapter — 统一领域适配器：所有领域专属配置从 YAML 加载，零硬编码分支。

架构：
  每个领域对应 data/domain_adapters/xxx.yaml 一份配置，运行时动态加载。
  DomainAdapter 封装：元信息、参数、基础类型映射、tier 规则、Prompt、别名。

用法：
  adapter = load_adapter("geo_strategy")           # 加载指定域
  adapter = get_adapter("geo_strategy")            # 缓存版
  adapters = list_adapters()                       # 列出所有可用域
  adapter = load_adapter("universal_neutral")      # 未知域兜底
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ── 子数据类 ──

@dataclass
class DomainMeta:
    domain_id: str = ""
    display_name: str = ""
    detect_keywords: list[dict] = field(default_factory=list)
    methodology_mode: str = "neutral"  # geo | narrative | neutral


@dataclass
class SamplingConfig:
    head_size: int = 2000
    tail_size: int = 2000
    lit_mode: bool = False


@dataclass
class TokenLimit:
    base: int = 200
    per: int = 100
    cap: int = 4000


@dataclass
class TokenConfig:
    l1_normalize: TokenLimit = field(default_factory=lambda: TokenLimit(base=200, per=150, cap=12000))
    l1_shard: TokenLimit = field(default_factory=lambda: TokenLimit(base=200, per=200, cap=6000))
    l1_refine: TokenLimit = field(default_factory=lambda: TokenLimit(base=200, per=120, cap=8000))
    l2_classify: TokenLimit = field(default_factory=lambda: TokenLimit(base=150, per=100, cap=4000))
    l3_cross: TokenLimit = field(default_factory=lambda: TokenLimit(base=300, per=60, cap=3000))

    def get_limit(self, key: str, n: int) -> int:
        """min(cap, base + n * per)"""
        tk = getattr(self, key, None)
        if tk is None:
            return 4000
        return min(tk.cap, tk.base + n * tk.per)


@dataclass
class CacheConfig:
    l1_enabled: bool = True
    l1_max_entries: int = 16
    l3_enabled: bool = True
    l3_max_entries: int = 64
    ttl_sec: int = 1800


@dataclass
class ParamConfig:
    shard_threshold: int = 12000
    shard_size: int = 7500
    shard_overlap: int = 800
    jaccard_threshold: float = 0.45
    max_sample_chars: int = 8000
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    token: TokenConfig = field(default_factory=TokenConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    fallback_thresholds: dict = field(default_factory=dict)


@dataclass
class TierRule:
    force_tier1_base_types: list[str] = field(default_factory=list)
    force_tier2_base_types: list[str] = field(default_factory=list)
    force_tier3_base_types: list[str] = field(default_factory=list)
    extra_rules: str = ""


@dataclass
class FallbackRules:
    org_overlap_threshold: int = 3
    org_members_map: dict = field(default_factory=dict)
    person_country_map: dict = field(default_factory=dict)


@dataclass
class LayerConfig:
    skip_layer3: bool = False
    quantified_supported: bool = True
    min_kept_for_check: int = 3
    warn_threshold: int = 8
    desc_truncate: int = 80
    sample_chars_l3: int = 5000
    monarchs_force_tier1: bool = False
    warlord_force_tier1: bool = False
    strategic_geo_tier2: bool = False
    log_file: str = ""
    fallback_rules: FallbackRules = field(default_factory=FallbackRules)


@dataclass
class Prompts:
    l2_entity_rules: str = ""
    l2_tier_table: str = ""
    l3_system_prompt: str = ""
    l3_redundancy_rules: str = ""
    l3_downgrade_rules: str = ""
    agent_domain_role: str = ""
    intel_extra_rules: str = ""
    strategic_context: str = ""


@dataclass
class Aliases:
    org_members: dict[str, frozenset[str]] = field(default_factory=dict)
    person_country: dict[str, str] = field(default_factory=dict)
    force_keep: set[str] = field(default_factory=set)
    entity_aliases: dict[str, set[str]] = field(default_factory=dict)
    strategic_regions: set[str] = field(default_factory=set)


@dataclass
class DomainAdapter:
    meta: DomainMeta = field(default_factory=DomainMeta)
    params: ParamConfig = field(default_factory=ParamConfig)
    base_type_mapping: dict[str, list[str]] = field(default_factory=dict)
    tier_rule: TierRule = field(default_factory=TierRule)
    layer: LayerConfig = field(default_factory=LayerConfig)
    prompts: Prompts = field(default_factory=Prompts)
    aliases: Aliases = field(default_factory=Aliases)
    methodology: dict = field(default_factory=dict)


# ── 适配器加载与缓存 ──

_adapter_cache: dict[str, DomainAdapter] = {}

# 7 大基础类型常量
BASE_TYPES = ["Agent", "Subordinate", "Resource", "Geography", "Contract", "Event", "Concept"]


def _find_adapter_dir() -> Path:
    """查找 domain_adapters 目录。统一经 config.resolve_rule_dirs() 解析（P1#8）。"""
    from strategy_forge.core.config import resolve_rule_dirs

    builtin, _custom = resolve_rule_dirs()
    candidates = [
        builtin / "domain_adapters",
        builtin.parent / "domain_adapters",
    ]
    for c in candidates:
        if c.exists():
            return c

    rule_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domain_adapters"
    if not rule_dir.exists():
        rule_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "rule"
    return rule_dir


def list_adapters() -> list[str]:
    """列出所有可用的领域适配器。"""
    adapter_dir = _find_adapter_dir()
    if not adapter_dir.exists():
        return []
    adapters = []
    for f in sorted(adapter_dir.glob("*.yaml")):
        adapters.append(f.stem)
    for f in sorted(adapter_dir.glob("*.yml")):
        adapters.append(f.stem)
    return adapters


def _parse_token_config(raw: dict | None) -> TokenConfig:
    if not raw or not isinstance(raw, dict):
        return TokenConfig()
    tc = TokenConfig()
    for key in ("l1_normalize", "l1_shard", "l1_refine", "l2_classify", "l3_cross"):
        val = raw.get(key, {})
        if isinstance(val, dict):
            setattr(tc, key, TokenLimit(
                base=int(val.get("base", 200)),
                per=int(val.get("per", 100)),
                cap=int(val.get("cap", 4000)),
            ))
    return tc


def _parse_sampling_config(raw: dict | None) -> SamplingConfig:
    if not raw or not isinstance(raw, dict):
        return SamplingConfig()
    return SamplingConfig(
        head_size=int(raw.get("head_size", 2000)),
        tail_size=int(raw.get("tail_size", 2000)),
        lit_mode=bool(raw.get("lit_mode", False)),
    )


def _parse_cache_config(raw: dict | None) -> CacheConfig:
    if not raw or not isinstance(raw, dict):
        return CacheConfig()
    return CacheConfig(
        l1_enabled=bool(raw.get("l1_enabled", True)),
        l1_max_entries=int(raw.get("l1_max_entries", 16)),
        l3_enabled=bool(raw.get("l3_enabled", True)),
        l3_max_entries=int(raw.get("l3_max_entries", 64)),
        ttl_sec=int(raw.get("ttl_sec", 1800)),
    )


def load_adapter(domain_id: str) -> DomainAdapter:
    """从 YAML 文件加载一个领域适配器。若文件不存在，返回 universal_neutral。"""
    adapter_dir = _find_adapter_dir()

    yaml_path = adapter_dir / f"{domain_id}.yaml"
    if not yaml_path.exists():
        yaml_path = adapter_dir / f"{domain_id}.yml"

    if not yaml_path.exists():
        logger.warning("[DomainAdapter] 域 '%s' 适配器文件未找到，回退 universal_neutral", domain_id)
        if domain_id != "universal_neutral":
            return load_adapter("universal_neutral")
        return _build_default_adapter()

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception as e:
        logger.error("[DomainAdapter] 加载 %s 失败: %s", yaml_path, e)
        if domain_id != "universal_neutral":
            return load_adapter("universal_neutral")
        return _build_default_adapter()

    if not isinstance(raw, dict):
        return _build_default_adapter()

    return _parse_adapter(domain_id, raw)


def _parse_adapter(domain_id: str, raw: dict) -> DomainAdapter:
    raw_meta = raw.get("domain_meta", {}) or {}
    raw_params = raw.get("param_config", {}) or {}
    raw_tier = raw.get("tier_rule", {}) or {}
    raw_layer = raw.get("layer_config", {}) or {}
    raw_prompts = raw.get("prompts", {}) or {}
    raw_aliases = raw.get("aliases", {}) or {}

    meta = DomainMeta(
        domain_id=raw_meta.get("domain_id", domain_id),
        display_name=raw_meta.get("display_name", domain_id),
        detect_keywords=raw_meta.get("detect_keywords", []),
        methodology_mode=raw_meta.get("methodology_mode", "neutral"),
    )

    params = ParamConfig(
        shard_threshold=int(raw_params.get("shard_threshold", 12000)),
        shard_size=int(raw_params.get("shard_size", 7500)),
        shard_overlap=int(raw_params.get("shard_overlap", 800)),
        jaccard_threshold=float(raw_params.get("jaccard_threshold", 0.45)),
        max_sample_chars=int(raw_params.get("max_sample_chars", 8000)),
        sampling=_parse_sampling_config(raw_params.get("sampling")),
        token=_parse_token_config(raw_params.get("token")),
        cache=_parse_cache_config(raw_params.get("cache")),
        fallback_thresholds=raw_params.get("fallback_thresholds", {}) or {},
    )

    base_type_mapping = raw.get("base_type_mapping", {}) or {}

    tier_rule = TierRule(
        force_tier1_base_types=list(raw_tier.get("force_tier1_base_types", [])) if raw_tier else [],
        force_tier2_base_types=list(raw_tier.get("force_tier2_base_types", [])) if raw_tier else [],
        force_tier3_base_types=list(raw_tier.get("force_tier3_base_types", [])) if raw_tier else [],
        extra_rules=str(raw_tier.get("extra_rules", "")) if raw_tier else "",
    )

    raw_fallback = raw_layer.get("fallback_rules", {}) or {}
    fallback_rules = FallbackRules(
        org_overlap_threshold=int(raw_fallback.get("org_overlap_threshold", 3)),
        org_members_map=raw_fallback.get("org_members_map", {}),
        person_country_map=raw_fallback.get("person_country_map", {}),
    )

    layer = LayerConfig(
        skip_layer3=bool(raw_layer.get("skip_layer3", False)),
        quantified_supported=bool(raw_layer.get("quantified_supported", True)),
        min_kept_for_check=int(raw_layer.get("min_kept_for_check", 3)),
        warn_threshold=int(raw_layer.get("warn_threshold", 8)),
        desc_truncate=int(raw_layer.get("desc_truncate", 80)),
        sample_chars_l3=int(raw_layer.get("sample_chars_l3", 5000)),
        monarchs_force_tier1=bool(raw_layer.get("monarchs_force_tier1", False)),
        warlord_force_tier1=bool(raw_layer.get("warlord_force_tier1", False)),
        strategic_geo_tier2=bool(raw_layer.get("strategic_geo_tier2", False)),
        log_file=str(raw_layer.get("log_file", "")),
        fallback_rules=fallback_rules,
    )

    prompts = Prompts(
        l2_entity_rules=str(raw_prompts.get("l2_entity_rules", "")),
        l2_tier_table=str(raw_prompts.get("l2_tier_table", "")),
        l3_system_prompt=str(raw_prompts.get("l3_system_prompt", "")),
        l3_redundancy_rules=str(raw_prompts.get("l3_redundancy_rules", "")),
        l3_downgrade_rules=str(raw_prompts.get("l3_downgrade_rules", "")),
        agent_domain_role=str(raw_prompts.get("agent_domain_role", "")),
        intel_extra_rules=str(raw_prompts.get("intel_extra_rules", "")),
        strategic_context=str(raw_prompts.get("strategic_context", "")),
    )

    # 解析别名
    org_members: dict[str, frozenset[str]] = {}
    raw_org = raw_aliases.get("_org_members", {}) or {}
    for k, v in raw_org.items():
        if isinstance(v, list):
            org_members[str(k)] = frozenset(str(x) for x in v)

    person_country: dict[str, str] = {}
    raw_pc = raw_aliases.get("_person_country", {}) or {}
    for k, v in raw_pc.items():
        person_country[str(k)] = str(v)

    force_keep: set[str] = set()
    raw_fk = raw_aliases.get("_force_keep", [])
    if isinstance(raw_fk, list):
        force_keep = {str(x) for x in raw_fk}

    entity_aliases: dict[str, set[str]] = {}
    raw_ea = raw_aliases.get("entity_aliases", {}) or {}
    for k, v in raw_ea.items():
        if isinstance(v, list):
            entity_aliases[str(k)] = {str(x) for x in v}

    strategic_regions: set[str] = set()
    raw_sr = raw_aliases.get("_strategic_regions", [])
    if isinstance(raw_sr, list):
        strategic_regions = {str(x) for x in raw_sr}

    aliases = Aliases(
        org_members=org_members,
        person_country=person_country,
        force_keep=force_keep,
        entity_aliases=entity_aliases,
        strategic_regions=strategic_regions,
    )

    methodology = raw.get("methodology", {}) or {}

    return DomainAdapter(
        meta=meta,
        params=params,
        base_type_mapping=base_type_mapping,
        tier_rule=tier_rule,
        layer=layer,
        prompts=prompts,
        aliases=aliases,
        methodology=methodology,
    )


def get_adapter(domain_id: str) -> DomainAdapter:
    """缓存版加载器。同一进程内不重复解析 YAML。"""
    if domain_id in _adapter_cache:
        return _adapter_cache[domain_id]
    adapter = load_adapter(domain_id)
    _adapter_cache[domain_id] = adapter
    return adapter


def clear_cache() -> None:
    """清空适配器缓存（用于测试或热重载）。"""
    _adapter_cache.clear()


def _build_default_adapter() -> DomainAdapter:
    return DomainAdapter(
        meta=DomainMeta(
            domain_id="universal_neutral",
            display_name="通用中立领域",
            detect_keywords=[],
            methodology_mode="neutral",
        ),
        params=ParamConfig(
            shard_threshold=12000,
            shard_size=7500,
            shard_overlap=800,
            jaccard_threshold=0.45,
            max_sample_chars=8000,
            sampling=SamplingConfig(head_size=2000, tail_size=2000, lit_mode=False),
            token=TokenConfig(),
            cache=CacheConfig(),
        ),
        base_type_mapping={},
        tier_rule=TierRule(
            force_tier1_base_types=["Agent"],
            force_tier2_base_types=["Subordinate"],
            force_tier3_base_types=["Resource", "Geography", "Contract", "Event", "Concept"],
            extra_rules="仅依据实体独立决策能力分级：拥有完整独立行动与利益诉求为tier1；依附其他主体决策为tier2；纯资源/工具/概念为tier3。",
        ),
        layer=LayerConfig(skip_layer3=False),
        prompts=Prompts(
            l3_system_prompt="你是通用博弈实体冗余检测专家。检测实体间是否存在决策权重叠。不确定时保守保留。",
        ),
        aliases=Aliases(),
    )
