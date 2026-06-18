"""覆盖率报告模块 - 生成抽样覆盖率和统计报告."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class CoverageStats:
    """单维度覆盖率统计."""
    dimension: str
    total_categories: int
    covered_categories: int
    coverage_rate: float
    uncovered: List[str]
    details: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class CoverageReport:
    """完整覆盖率报告."""
    population_size: int
    sample_size: int
    overall_rate: float
    batch_stats: Optional[CoverageStats] = None
    dimension_stats: Dict[str, CoverageStats] = field(default_factory=dict)
    numerical_summary: Optional[pd.DataFrame] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def compute_overall_coverage(population: pd.DataFrame, sample: pd.DataFrame) -> float:
    """计算总体样本覆盖率."""
    n_pop = len(population)
    n_sam = len(sample)
    if n_pop == 0:
        return 0.0
    return n_sam / n_pop


def compute_dimension_coverage(
    population: pd.DataFrame,
    sample: pd.DataFrame,
    dimension: str,
) -> CoverageStats:
    """计算某个维度（列）的分类覆盖率."""
    if dimension not in population.columns:
        raise ValueError(f"维度列 '{dimension}' 不存在于总体数据中")

    pop_cats = set(population[dimension].dropna().astype(str).unique())
    if dimension in sample.columns:
        sam_cats = set(sample[dimension].dropna().astype(str).unique())
    else:
        sam_cats = set()

    covered = pop_cats & sam_cats
    uncovered = sorted(pop_cats - sam_cats)

    rate = len(covered) / len(pop_cats) if pop_cats else 0.0

    details: Dict[str, Dict[str, int]] = {}
    for cat in sorted(pop_cats):
        pop_cnt = int((population[dimension].astype(str) == cat).sum())
        sam_cnt = int((sample[dimension].astype(str) == cat).sum()) if dimension in sample.columns else 0
        details[cat] = {"population": pop_cnt, "sample": sam_cnt}

    return CoverageStats(
        dimension=dimension,
        total_categories=len(pop_cats),
        covered_categories=len(covered),
        coverage_rate=rate,
        uncovered=uncovered,
        details=details,
    )


def compute_numerical_summary(
    population: pd.DataFrame,
    sample: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """计算数值列在总体和样本中的统计摘要对比."""
    if columns is None:
        columns = population.select_dtypes(include="number").columns.tolist()

    rows = []
    for col in columns:
        if col not in population.columns:
            continue
        pop_series = population[col].dropna()
        sam_series = sample[col].dropna() if col in sample.columns else pd.Series(dtype="float64")

        for tag, s in (("population", pop_series), ("sample", sam_series)):
            if len(s) == 0:
                rows.append({
                    "column": col,
                    "source": tag,
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "p25": None,
                    "median": None,
                    "p75": None,
                    "max": None,
                })
                continue
            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            rows.append({
                "column": col,
                "source": tag,
                "count": int(desc.get("count", 0)),
                "mean": float(desc.get("mean", 0)),
                "std": float(desc.get("std", 0)),
                "min": float(desc.get("min", 0)),
                "p25": float(desc.get("25%", 0)),
                "median": float(desc.get("50%", 0)),
                "p75": float(desc.get("75%", 0)),
                "max": float(desc.get("max", 0)),
            })

    return pd.DataFrame(rows)


def generate_report(
    population: pd.DataFrame,
    sample: pd.DataFrame,
    dimensions: Optional[List[str]] = None,
    batch_col: Optional[str] = None,
    numerical_cols: Optional[List[str]] = None,
) -> CoverageReport:
    """生成完整覆盖率报告."""
    overall = compute_overall_coverage(population, sample)

    dim_stats: Dict[str, CoverageStats] = {}

    batch_stats = None
    if batch_col and batch_col in population.columns:
        batch_stats = compute_dimension_coverage(population, sample, batch_col)

    if dimensions:
        for dim in dimensions:
            if dim == batch_col:
                continue
            try:
                cs = compute_dimension_coverage(population, sample, dim)
                dim_stats[dim] = cs
            except ValueError:
                continue

    num_summary = None
    try:
        num_summary = compute_numerical_summary(population, sample, numerical_cols)
    except Exception:
        num_summary = None

    return CoverageReport(
        population_size=len(population),
        sample_size=len(sample),
        overall_rate=overall,
        batch_stats=batch_stats,
        dimension_stats=dim_stats,
        numerical_summary=num_summary,
    )


def report_to_text(report: CoverageReport) -> str:
    """将报告渲染为可读文本."""
    lines = []
    lines.append("=" * 60)
    lines.append("质检抽样覆盖率报告")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"总体数量: {report.population_size}")
    lines.append(f"样本数量: {report.sample_size}")
    lines.append(f"总体抽样率: {report.overall_rate * 100:.2f}%")
    lines.append("")

    if report.batch_stats:
        bs = report.batch_stats
        lines.append(f"--- 批次维度: {bs.dimension} ---")
        lines.append(f"总批次数: {bs.total_categories}  已覆盖: {bs.covered_categories}  覆盖率: {bs.coverage_rate * 100:.2f}%")
        if bs.uncovered:
            lines.append(f"未覆盖批次: {', '.join(bs.uncovered)}")
        lines.append("批次详情:")
        for cat, info in bs.details.items():
            p_cnt = info["population"]
            s_cnt = info["sample"]
            r = (s_cnt / p_cnt * 100) if p_cnt > 0 else 0
            lines.append(f"  {cat}: 总体{p_cnt} / 抽样{s_cnt} ({r:.1f}%)")
        lines.append("")

    if report.dimension_stats:
        for dim, ds in report.dimension_stats.items():
            lines.append(f"--- 维度: {dim} ---")
            lines.append(f"类别总数: {ds.total_categories}  已覆盖: {ds.covered_categories}  覆盖率: {ds.coverage_rate * 100:.2f}%")
            if ds.uncovered:
                lines.append(f"未覆盖类别: {', '.join(ds.uncovered)}")
            lines.append("")

    if report.numerical_summary is not None and not report.numerical_summary.empty:
        lines.append("--- 数值列统计对比 ---")
        lines.append(report.numerical_summary.to_string(index=False))
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
