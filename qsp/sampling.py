"""抽样规则引擎 - 支持多种抽样策略和规则配置."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class SamplingMethod(str, Enum):
    """抽样方法枚举."""
    RANDOM = "random"
    STRATIFIED = "stratified"
    SYSTEMATIC = "systematic"
    CLUSTER = "cluster"
    PPS = "pps"
    SIMPLE = "simple"


class SampleSizeMode(str, Enum):
    """样本量计算模式."""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    STATISTICAL = "statistical"


@dataclass
class SamplingRule:
    """抽样规则配置."""
    method: SamplingMethod = SamplingMethod.RANDOM
    mode: SampleSizeMode = SampleSizeMode.PERCENTAGE
    sample_size: int = 0
    percentage: float = 0.10
    confidence_level: float = 0.95
    margin_of_error: float = 0.05
    expected_defect_rate: float = 0.05
    stratify_by: Optional[str] = None
    cluster_by: Optional[str] = None
    cluster_min_clusters: int = 2
    pps_size_col: Optional[str] = None
    pps_replace: bool = True
    pps_imbalance_threshold: float = 0.001
    pps_fallback: bool = True
    random_seed: Optional[int] = 42
    min_per_group: int = 1
    weights: Dict[str, float] = field(default_factory=dict)


def calculate_statistical_sample_size(
    population_size: int,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.05,
    expected_defect_rate: float = 0.05,
) -> int:
    """基于统计学公式计算样本量.

    Cochran 公式：n = (Z^2 * p * (1-p)) / E^2
    有限总体校正：n_corrected = n / (1 + (n-1)/N)
    """
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576, 0.999: 3.291}
    z = z_map.get(confidence_level, 1.96)
    p = expected_defect_rate
    e = margin_of_error

    n_0 = (z ** 2 * p * (1 - p)) / (e ** 2)

    if population_size > 0:
        n = n_0 / (1 + (n_0 - 1) / population_size)
    else:
        n = n_0

    return max(1, math.ceil(n))


def determine_sample_size(population: int, rule: SamplingRule) -> int:
    """根据规则确定目标样本量."""
    if rule.mode == SampleSizeMode.FIXED:
        size = rule.sample_size
    elif rule.mode == SampleSizeMode.PERCENTAGE:
        size = math.ceil(population * rule.percentage)
    else:
        size = calculate_statistical_sample_size(
            population,
            rule.confidence_level,
            rule.margin_of_error,
            rule.expected_defect_rate,
        )
    return max(1, min(size, population))


def simple_random_sample(df: pd.DataFrame, n: int, seed: Optional[int] = None) -> pd.DataFrame:
    """简单随机抽样."""
    return df.sample(n=n, random_state=seed)


def systematic_sample(df: pd.DataFrame, n: int, seed: Optional[int] = None) -> pd.DataFrame:
    """系统抽样（等距抽样）."""
    n_total = len(df)
    if n >= n_total:
        return df.copy()
    interval = n_total / n
    rng = random.Random(seed)
    start = rng.randint(0, int(interval) - 1) if interval > 1 else 0
    indices = [int(start + i * interval) for i in range(n)]
    indices = [i % n_total for i in indices]
    return df.iloc[indices].copy()


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    stratify_by: str,
    seed: Optional[int] = None,
    min_per_group: int = 1,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """分层抽样."""
    if stratify_by not in df.columns:
        raise ValueError(f"分层列 '{stratify_by}' 不存在于数据中")

    groups = list(df.groupby(stratify_by, dropna=False))
    if not groups:
        raise ValueError("没有可用的分层组")

    total_pop = len(df)
    group_counts: Dict[str, Tuple[pd.DataFrame, int]] = {}
    for group_name, group_df in groups:
        group_counts[str(group_name)] = (group_df, len(group_df))

    sizes: Dict[str, int] = {}
    remaining = n

    for g_name, (g_df, g_size) in group_counts.items():
        if weights and g_name in weights:
            w = weights[g_name]
            allocated = max(min_per_group, math.ceil(n * w))
        else:
            ratio = g_size / total_pop if total_pop > 0 else 0
            allocated = max(min_per_group, round(n * ratio))
        allocated = min(allocated, g_size)
        sizes[g_name] = allocated
        remaining -= allocated

    sorted_groups = sorted(group_counts.items(), key=lambda x: x[1][1], reverse=True)
    idx = 0
    while remaining > 0 and sorted_groups:
        g_name, (g_df, g_size) = sorted_groups[idx % len(sorted_groups)]
        if sizes[g_name] < g_size:
            sizes[g_name] += 1
            remaining -= 1
        idx += 1
        if idx > len(sorted_groups) * 10:
            break

    sampled_parts = []
    for g_name, (g_df, g_size) in group_counts.items():
        s = sizes.get(g_name, min_per_group)
        s = min(s, g_size)
        part = g_df.sample(n=s, random_state=seed)
        sampled_parts.append(part)

    result = pd.concat(sampled_parts, ignore_index=True)
    return result.head(n)


def cluster_sample(
    df: pd.DataFrame,
    n: int,
    cluster_by: str,
    seed: Optional[int] = None,
    min_clusters: int = 2,
) -> pd.DataFrame:
    """整群抽样 (Cluster Sampling).

    随机抽取若干个整群，整群内样本全部入样。
    若样本数不足则补加下一个群；若超过则从已选群中二次随机裁剪。
    """
    if cluster_by not in df.columns:
        raise ValueError(f"整群列 '{cluster_by}' 不存在于数据中")

    clusters = list(df[cluster_by].unique())
    if not clusters:
        raise ValueError("没有可用的群")
    if min_clusters > len(clusters):
        min_clusters = len(clusters)
    if min_clusters < 1:
        min_clusters = 1

    rng = random.Random(seed)
    shuffled = list(clusters)
    rng.shuffle(shuffled)

    selected_clusters: List[pd.DataFrame] = []
    current_count = 0
    for c in shuffled:
        cluster_df = df[df[cluster_by] == c]
        selected_clusters.append(cluster_df)
        current_count += len(cluster_df)
        if current_count >= n and len(selected_clusters) >= min_clusters:
            break

    if not selected_clusters:
        raise ValueError("无法选取整群后样本数为零")

    result = pd.concat(selected_clusters, ignore_index=True)

    if len(result) > n:
        remain = n
        parts = []
        rng2 = random.Random(seed)
        shuffled_parts = list(selected_clusters)
        rng2.shuffle(shuffled_parts)
        for i, part in enumerate(shuffled_parts):
            if i < min_clusters:
                slots_left = min_clusters - i
                keep = min(len(part), max(1, remain // slots_left))
                parts.append(part.sample(n=keep, random_state=seed))
                remain -= keep
            elif remain > 0:
                take = min(len(part), remain)
                parts.append(part.sample(n=take, random_state=seed))
                remain -= take
            else:
                break
        if remain > 0 and parts:
            for i in range(len(parts)):
                if remain <= 0:
                    break
                part = shuffled_parts[i % len(shuffled_parts)]
                current_idx = i % len(parts)
                current_part = parts[current_idx]
                full_part = shuffled_parts[current_idx]
                if len(current_part) < len(full_part):
                    extra = min(remain, len(full_part) - len(current_part))
                    leftover = full_part.drop(current_part.index)
                    add = leftover.sample(n=extra, random_state=seed)
                    parts[current_idx] = pd.concat([current_part, add], ignore_index=True)
                    remain -= extra
        result = pd.concat(parts, ignore_index=True)

    return result


class PPSSkewWarning:
    """PPS 极不均衡回退警告."""

    def __init__(self, reason: str, fallback: str, n_imbalanced: int,
                 n_total: int, threshold: float):
        self.reason = reason
        self.fallback = fallback
        self.n_imbalanced = n_imbalanced
        self.n_total = n_total
        self.threshold = threshold

    def __repr__(self) -> str:
        return (
            f"PPSSkewWarning(reason={self.reason!r}, fallback={self.fallback!r}, "
            f"n_imbalanced={self.n_imbalanced}/{self.n_total}, "
            f"threshold={self.threshold})"
        )


def _detect_pps_skew(
    sizes: "pd.Series[float]",
    n: int,
    threshold: float = 0.001,
) -> Optional[PPSSkewWarning]:
    """检测 PPS 规模分布是否极不均衡，返回警告或 None."""
    n_zero = int((sizes == 0).sum())
    total = sizes.sum()

    if n_zero > 0 and total <= 0:
        return PPSSkewWarning(
            reason="all_zero", fallback="none",
            n_imbalanced=n_zero, n_total=len(sizes), threshold=threshold,
        )

    if n_zero > 0:
        return PPSSkewWarning(
            reason="has_zero_weights",
            fallback="exclude_zero_then_pps",
            n_imbalanced=n_zero, n_total=len(sizes), threshold=threshold,
        )

    if total <= 0:
        return None

    probs = sizes / total

    max_prob = float(probs.max())
    if max_prob > 0.5 and n > 3:
        return PPSSkewWarning(
            reason="single_dominant_weight",
            fallback="cap_and_redistribute",
            n_imbalanced=1, n_total=len(sizes), threshold=threshold,
        )

    n_below = int((probs < threshold).sum())
    if n_below > len(sizes) * 0.3:
        return PPSSkewWarning(
            reason="too_many_tiny_weights",
            fallback="blend_equal_and_pps",
            n_imbalanced=n_below, n_total=len(sizes), threshold=threshold,
        )

    return None


def pps_sample(
    df: pd.DataFrame,
    n: int,
    size_col: str,
    seed: Optional[int] = None,
    replace: bool = True,
    imbalance_threshold: float = 0.001,
    fallback: bool = True,
) -> pd.DataFrame:
    """PPS 抽样 (Probability Proportional to Size).

    按与规模大小成比例的概率抽样。
    size_col 为规模指标列，值越大被抽中概率越高。
    replace=True 为有放回 PPS（汉森-赫维茨估计），replace=False 为无放回 PPS。

    当规模分布极不均衡时（零权重、过多极小权重、单权重主导），
    若 fallback=True 则自动回退到修正策略而非直接报错：
      - has_zero_weights → 排除零值后对剩余做 PPS
      - too_many_tiny_weights → 等概率与 PPS 各 50% 混合
      - single_dominant_weight → 截断最大权重并重新分配
    若 fallback=False 则抛出 ValueError。
    """
    if size_col not in df.columns:
        raise ValueError(f"规模列 '{size_col}' 不存在于数据中")

    sizes = pd.to_numeric(df[size_col], errors="coerce")
    if sizes.isna().any():
        raise ValueError(f"规模列 '{size_col}' 含有非数值或空值")
    if (sizes < 0).any():
        raise ValueError(f"规模列 '{size_col}' 含有负值，无法作为抽样概率权重")

    if n <= 0:
        raise ValueError("样本量 n 必须为正整数")
    if n > len(df):
        n = len(df)

    skew = _detect_pps_skew(sizes, n, threshold=imbalance_threshold)

    if skew is not None and skew.reason == "all_zero":
        raise ValueError(
            f"规模列 '{size_col}' 全部为零，无有效样本可做 PPS 抽样"
        )

    total = sizes.sum()
    if total <= 0:
        raise ValueError(f"规模列 '{size_col}' 总和非正，无法计算概率")

    if skew is not None:
        if not fallback:
            raise ValueError(
                f"PPS 规模分布极不均衡 ({skew.reason})，"
                f"影响 {skew.n_imbalanced}/{skew.n_total} 个样本；"
                f"设置 fallback=True 可自动回退"
            )

        if skew.reason == "has_zero_weights":
            valid_mask = sizes > 0
            valid_df = df[valid_mask].reset_index(drop=True)
            valid_sizes = sizes[valid_mask].reset_index(drop=True)
            if len(valid_df) == 0:
                raise ValueError(
                    f"规模列 '{size_col}' 全部为零，无有效样本可做 PPS 抽样"
                )
            if not replace and n > len(valid_df):
                raise ValueError(
                    f"排除零权重后有效样本 {len(valid_df)} < 请求样本量 {n}，"
                    f"无法完成无放回 PPS 抽样"
                )
            return _pps_core(valid_df, valid_sizes, n, seed, replace)

        elif skew.reason == "too_many_tiny_weights":
            return _pps_blend_equal_and_weighted(df, sizes, n, seed, replace)

        elif skew.reason == "single_dominant_weight":
            return _pps_cap_and_redistribute(df, sizes, n, seed, replace)

    return _pps_core(df, sizes, n, seed, replace)


def _pps_core(
    df: pd.DataFrame,
    sizes: "pd.Series[float]",
    n: int,
    seed: Optional[int],
    replace: bool,
) -> pd.DataFrame:
    """核心 PPS 抽样，假定 sizes 已校验且全部 > 0."""
    total = sizes.sum()
    rng = random.Random(seed)
    probs = (sizes / total).tolist()

    if replace:
        indices = rng.choices(range(len(df)), weights=probs, k=n)
        result = df.iloc[indices].reset_index(drop=True)
    else:
        remaining = list(range(len(df)))
        remaining_probs = list(probs)
        selected = []
        for _ in range(n):
            total_w = sum(remaining_probs)
            if total_w <= 0:
                break
            normalized = [p / total_w for p in remaining_probs]
            idx_in_remain = rng.choices(range(len(remaining)), weights=normalized, k=1)[0]
            selected.append(remaining[idx_in_remain])
            remaining.pop(idx_in_remain)
            remaining_probs.pop(idx_in_remain)
        result = df.iloc[selected].reset_index(drop=True)

    return result


def _pps_blend_equal_and_weighted(
    df: pd.DataFrame,
    sizes: "pd.Series[float]",
    n: int,
    seed: Optional[int],
    replace: bool,
) -> pd.DataFrame:
    """等概率与 PPS 各 50% 混合回退策略."""
    n_equal = max(1, n // 2)
    n_weighted = n - n_equal

    total = sizes.sum()
    rng = random.Random(seed)

    if replace:
        equal_indices = rng.choices(range(len(df)), k=n_equal)
        weighted_indices = rng.choices(range(len(df)), weights=(sizes / total).tolist(), k=n_weighted)
        all_indices = equal_indices + weighted_indices
        rng.shuffle(all_indices)
        result = df.iloc[all_indices].reset_index(drop=True)
    else:
        selected = set()
        total_w = float(sizes.sum())
        weights = (sizes / total_w).tolist()
        remaining = list(range(len(df)))
        remaining_weights = list(weights)

        equal_pool = list(range(len(df)))
        rng.shuffle(equal_pool)
        for idx in equal_pool:
            if len(selected) >= n_equal:
                break
            selected.add(idx)

        for _ in range(n_weighted):
            if len(selected) >= n:
                break
            if not remaining:
                break
            cur_total = sum(remaining_weights)
            if cur_total <= 0:
                remaining_indices = [i for i in remaining if i not in selected]
                if remaining_indices:
                    pick = rng.choice(remaining_indices)
                    selected.add(pick)
                break
            norm = [w / cur_total for w in remaining_weights]
            pick_local = rng.choices(range(len(remaining)), weights=norm, k=1)[0]
            picked = remaining[pick_local]
            selected.add(picked)
            remaining.pop(pick_local)
            remaining_weights.pop(pick_local)

        if len(selected) < n:
            leftover = [i for i in range(len(df)) if i not in selected]
            rng.shuffle(leftover)
            for idx in leftover:
                if len(selected) >= n:
                    break
                selected.add(idx)

        result = df.iloc[sorted(selected)].reset_index(drop=True)

    return result.head(n)


def _pps_cap_and_redistribute(
    df: pd.DataFrame,
    sizes: "pd.Series[float]",
    n: int,
    seed: Optional[int],
    replace: bool,
) -> pd.DataFrame:
    """截断最大权重并重新分配回退策略.

    将任何超过 0.5 的权重截断到 0.5，溢出部分平均分配给其余项。
    """
    total = float(sizes.sum())
    probs = sizes / total
    cap = 0.5
    capped = probs.copy()

    overflow = 0.0
    dominant_indices = []
    for i, p in enumerate(capped):
        if p > cap:
            overflow += p - cap
            capped.iloc[i] = cap
            dominant_indices.append(i)

    if dominant_indices and overflow > 0:
        non_dominant = [i for i in range(len(capped)) if i not in dominant_indices]
        if non_dominant:
            per_item = overflow / len(non_dominant)
            for i in non_dominant:
                capped.iloc[i] += per_item
        else:
            per_item = overflow / len(dominant_indices)
            for i in dominant_indices:
                capped.iloc[i] += per_item

    rng = random.Random(seed)
    weights = capped.tolist()

    if replace:
        indices = rng.choices(range(len(df)), weights=weights, k=n)
        result = df.iloc[indices].reset_index(drop=True)
    else:
        remaining = list(range(len(df)))
        remaining_weights = list(weights)
        selected = []
        for _ in range(n):
            cur_total = sum(remaining_weights)
            if cur_total <= 0:
                break
            norm = [w / cur_total for w in remaining_weights]
            idx_local = rng.choices(range(len(remaining)), weights=norm, k=1)[0]
            selected.append(remaining[idx_local])
            remaining.pop(idx_local)
            remaining_weights.pop(idx_local)
        result = df.iloc[selected].reset_index(drop=True)

    return result


def apply_sampling(df: pd.DataFrame, rule: SamplingRule) -> pd.DataFrame:
    """根据抽样规则执行抽样."""
    population = len(df)
    if population == 0:
        raise ValueError("输入数据为空")

    n = determine_sample_size(population, rule)

    if rule.method == SamplingMethod.SIMPLE or rule.method == SamplingMethod.RANDOM:
        return simple_random_sample(df, n, rule.random_seed)
    elif rule.method == SamplingMethod.SYSTEMATIC:
        return systematic_sample(df, n, rule.random_seed)
    elif rule.method == SamplingMethod.STRATIFIED:
        if not rule.stratify_by:
            raise ValueError("分层抽样需要指定 stratify_by 参数")
        return stratified_sample(
            df, n, rule.stratify_by, rule.random_seed, rule.min_per_group, rule.weights or None
        )
    elif rule.method == SamplingMethod.CLUSTER:
        if not rule.cluster_by:
            raise ValueError("整群抽样需要指定 cluster_by 参数")
        return cluster_sample(df, n, rule.cluster_by, rule.random_seed, rule.cluster_min_clusters)
    elif rule.method == SamplingMethod.PPS:
        if not rule.pps_size_col:
            raise ValueError("PPS 抽样需要指定 pps_size_col 参数 (规模列)")
        return pps_sample(
            df, n, rule.pps_size_col, rule.random_seed,
            replace=rule.pps_replace,
            imbalance_threshold=rule.pps_imbalance_threshold,
            fallback=rule.pps_fallback,
        )
    else:
        raise ValueError(f"未知的抽样方法: {rule.method}")


def load_rule_from_config(config: Dict[str, Any]) -> SamplingRule:
    """从配置字典加载抽样规则."""
    method = SamplingMethod(config.get("method", "random"))
    mode = SampleSizeMode(config.get("mode", "percentage"))

    return SamplingRule(
        method=method,
        mode=mode,
        sample_size=int(config.get("sample_size", 0)),
        percentage=float(config.get("percentage", 0.10)),
        confidence_level=float(config.get("confidence_level", 0.95)),
        margin_of_error=float(config.get("margin_of_error", 0.05)),
        expected_defect_rate=float(config.get("expected_defect_rate", 0.05)),
        stratify_by=config.get("stratify_by"),
        cluster_by=config.get("cluster_by"),
        cluster_min_clusters=int(config.get("cluster_min_clusters", 2)),
        pps_size_col=config.get("pps_size_col"),
        pps_replace=bool(config.get("pps_replace", True)),
        pps_imbalance_threshold=float(config.get("pps_imbalance_threshold", 0.001)),
        pps_fallback=bool(config.get("pps_fallback", True)),
        random_seed=config.get("random_seed", 42),
        min_per_group=int(config.get("min_per_group", 1)),
        weights=config.get("weights", {}) or {},
    )
