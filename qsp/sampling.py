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


def pps_sample(
    df: pd.DataFrame,
    n: int,
    size_col: str,
    seed: Optional[int] = None,
    replace: bool = True,
) -> pd.DataFrame:
    """PPS 抽样 (Probability Proportional to Size).

    按与规模大小成比例的概率抽样。
    size_col 为规模指标列，值越大被抽中概率越高。
    replace=True 为有放回 PPS（汉森-赫维茨估计），replace=False 为无放回 PPS。
    """
    if size_col not in df.columns:
        raise ValueError(f"规模列 '{size_col}' 不存在于数据中")

    sizes = pd.to_numeric(df[size_col], errors="coerce")
    if sizes.isna().any():
        raise ValueError(f"规模列 '{size_col}' 含有非数值或空值")
    if (sizes < 0).any():
        raise ValueError(f"规模列 '{size_col}' 含有负值，无法作为抽样概率权重")
    total = sizes.sum()
    if total <= 0:
        raise ValueError(f"规模列 '{size_col}' 总和非正，无法计算概率")
    if not replace and (sizes == 0).any() and n > (sizes > 0).sum():
        raise ValueError(
            f"无放回 PPS 抽样中，规模为 0 的样本无法被抽中；"
            f"有效样本数 {(sizes > 0).sum()} 小于请求样本量 {n}"
        )

    if n <= 0:
        raise ValueError("样本量 n 必须为正整数")
    if n > len(df):
        n = len(df)

    rng = random.Random(seed)
    probs = (sizes / total).tolist()

    if replace:
        indices = rng.choices(range(len(df)), weights=probs, k=n)
        result = df.iloc[indices].reset_index(drop=True)
    else:
        if n > len(df):
            raise ValueError("无放回 PPS 抽样样本量不能超过总体数量")
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
        return pps_sample(df, n, rule.pps_size_col, rule.random_seed, rule.pps_replace)
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
        random_seed=config.get("random_seed", 42),
        min_per_group=int(config.get("min_per_group", 1)),
        weights=config.get("weights", {}) or {},
    )
