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
) -> pd.DataFrame:
    """整群抽样."""
    if cluster_by not in df.columns:
        raise ValueError(f"整群列 '{cluster_by}' 不存在于数据中")

    clusters = list(df[cluster_by].unique())
    if not clusters:
        raise ValueError("没有可用的聚类")

    rng = random.Random(seed)
    rng.shuffle(clusters)

    selected = []
    current_count = 0
    for c in clusters:
        cluster_df = df[df[cluster_by] == c]
        if current_count + len(cluster_df) <= n * 1.5 or not selected:
            selected.append(cluster_df)
            current_count += len(cluster_df)
        if current_count >= n:
            break

    result = pd.concat(selected, ignore_index=True)
    if len(result) > n:
        result = result.sample(n=n, random_state=seed)
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
        return cluster_sample(df, n, rule.cluster_by, rule.random_seed)
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
        random_seed=config.get("random_seed", 42),
        min_per_group=int(config.get("min_per_group", 1)),
        weights=config.get("weights", {}) or {},
    )
