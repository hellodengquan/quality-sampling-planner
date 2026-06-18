"""风险分析模块 - 抽样风险评估与提示."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from .report import CoverageReport
from .sampling import SamplingRule


class RiskLevel(str, Enum):
    """风险级别."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskAlert:
    """风险提示."""
    level: RiskLevel
    category: str
    title: str
    description: str
    suggestion: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAnalysis:
    """风险分析结果."""
    alerts: List[RiskAlert] = field(default_factory=list)

    @property
    def level_counts(self) -> Dict[str, int]:
        counts = {lvl: 0 for lvl in RiskLevel}
        for a in self.alerts:
            counts[a.level] += 1
        return {k.value: v for k, v in counts.items()}

    @property
    def has_critical(self) -> bool:
        return any(a.level == RiskLevel.CRITICAL for a in self.alerts)

    @property
    def has_high(self) -> bool:
        return any(a.level in (RiskLevel.HIGH, RiskLevel.CRITICAL) for a in self.alerts)

    def by_level(self, level: RiskLevel) -> List[RiskAlert]:
        return [a for a in self.alerts if a.level == level]


def _add_alert(alerts: List[RiskAlert], level: RiskLevel, category: str,
               title: str, description: str, suggestion: str, **data):
    alerts.append(RiskAlert(
        level=level,
        category=category,
        title=title,
        description=description,
        suggestion=suggestion,
        data=dict(data),
    ))


def _check_sample_size(population: int, sample: int, rule: SamplingRule, alerts: List[RiskAlert]):
    rate = sample / population if population > 0 else 0

    if sample < 1:
        _add_alert(alerts, RiskLevel.CRITICAL, "样本量", "样本量为零",
                   f"样本大小为 {sample}，无法进行有效质检。",
                   "请检查抽样规则配置或增加样本量。", population=population, sample=sample)
        return

    if rate < 0.01:
        _add_alert(alerts, RiskLevel.HIGH, "样本量", "抽样率过低",
                   f"总体 {population}，抽样 {sample}，抽样率仅 {rate * 100:.2f}%，可能漏检。",
                   "建议提高抽样比例或采用分层抽样保证覆盖。",
                   population=population, sample=sample, rate=rate)
    elif rate < 0.03:
        _add_alert(alerts, RiskLevel.MEDIUM, "样本量", "抽样率偏低",
                   f"抽样率 {rate * 100:.2f}%，对于关键生产批次需谨慎。",
                   "可根据缺陷历史数据考虑提高抽样比例。",
                   rate=rate)

    if population >= 1000 and sample < 30:
        _add_alert(alerts, RiskLevel.HIGH, "样本量", "统计意义样本不足",
                   f"总体 {population} 但样本仅 {sample}，难以支撑统计推断。",
                   "建议使用统计模式 (STATISTICAL) 计算样本量或至少抽样 30 件。",
                   population=population, sample=sample)


def _check_coverage(report: CoverageReport, alerts: List[RiskAlert],
                    min_batch_coverage: float = 0.9,
                    min_dim_coverage: float = 0.85):
    if report.batch_stats:
        br = report.batch_stats.coverage_rate
        if br < min_batch_coverage:
            lvl = RiskLevel.CRITICAL if br < 0.5 else RiskLevel.HIGH
            _add_alert(alerts, lvl, "批次覆盖", f"批次覆盖率不足 ({br * 100:.1f}%)",
                       f"共 {report.batch_stats.total_categories} 个批次，"
                       f"仅覆盖 {report.batch_stats.covered_categories} 个，"
                       f"未覆盖: {report.batch_stats.uncovered}",
                       "请使用分层抽样按批次分配样本，或补充未覆盖批次的样本。",
                       coverage_rate=br, uncovered=report.batch_stats.uncovered)

    for dim, ds in report.dimension_stats.items():
        dr = ds.coverage_rate
        if dr < min_dim_coverage:
            lvl = RiskLevel.HIGH if dr < 0.5 else RiskLevel.MEDIUM
            _add_alert(alerts, lvl, "维度覆盖", f"维度 '{dim}' 覆盖率不足 ({dr * 100:.1f}%)",
                       f"类别 {ds.total_categories}，覆盖 {ds.covered_categories}，"
                       f"未覆盖: {ds.uncovered}",
                       f"建议将 '{dim}' 纳入分层抽样的 stratify_by 参数。",
                       dimension=dim, coverage_rate=dr, uncovered=ds.uncovered)


def _check_small_batches(report: CoverageReport, alerts: List[RiskAlert], min_per_batch: int = 2):
    if not report.batch_stats:
        return
    small = []
    zero = []
    for cat, info in report.batch_stats.details.items():
        if info["sample"] == 0 and info["population"] > 0:
            zero.append(cat)
        elif 0 < info["sample"] < min_per_batch:
            small.append(cat)
    if zero:
        _add_alert(alerts, RiskLevel.CRITICAL, "批次覆盖", "存在未抽样批次",
                   f"以下批次完全未抽中: {zero}",
                   "请强制包含这些批次或调整抽样策略。", batches=zero)
    if small:
        _add_alert(alerts, RiskLevel.MEDIUM, "样本分布", "部分批次抽样数量过少",
                   f"以下批次抽样不足 {min_per_batch}: {small}",
                   f"可提高 min_per_group 参数至至少 {min_per_batch}。",
                   batches=small, threshold=min_per_batch)


def _check_numerical_bias(report: CoverageReport, alerts: List[RiskReport], bias_threshold: float = 0.2):
    if report.numerical_summary is None or report.numerical_summary.empty:
        return
    df = report.numerical_summary
    cols = df["column"].unique()
    for col in cols:
        pop_row = df[(df["column"] == col) & (df["source"] == "population")]
        sam_row = df[(df["column"] == col) & (df["source"] == "sample")]
        if pop_row.empty or sam_row.empty:
            continue
        pop_mean = pop_row["mean"].values[0]
        sam_mean = sam_row["mean"].values[0]
        if pop_mean is None or sam_mean is None:
            continue
        if pop_mean == 0:
            relative_diff = 0.0 if sam_mean == 0 else 1.0
        else:
            relative_diff = abs(sam_mean - pop_mean) / abs(pop_mean)
        if relative_diff > bias_threshold:
            lvl = RiskLevel.HIGH if relative_diff > bias_threshold * 2 else RiskLevel.MEDIUM
            _add_alert(alerts, lvl, "统计偏差", f"数值列 '{col}' 均值偏离较大",
                       f"总体均值 {pop_mean:.4g}，样本均值 {sam_mean:.4g}，相对偏差 {relative_diff * 100:.1f}%",
                       "检查样本是否存在抽样偏差，可考虑重新抽样或分层抽样。",
                       column=col, pop_mean=pop_mean, sam_mean=sam_mean, bias=relative_diff)


def _check_method_vs_data(df: pd.DataFrame, rule: SamplingRule, alerts: List[RiskAlert]):
    from .sampling import SamplingMethod

    if rule.method == SamplingMethod.STRATIFIED:
        if not rule.stratify_by or rule.stratify_by not in df.columns:
            pass
        else:
            nunique = df[rule.stratify_by].nunique(dropna=True)
            if nunique > len(df) * 0.5:
                _add_alert(alerts, RiskLevel.MEDIUM, "抽样策略", "分层维度过稀疏",
                           f"stratify_by='{rule.stratify_by}' 有 {nunique} 个唯一值，基数过高，可能导致分组样本不足。",
                           "考虑合并类别或更换分层维度。",
                           stratify_by=rule.stratify_by, unique_count=nunique)

    if rule.method == SamplingMethod.SYSTEMATIC and len(df) >= 50:
        _add_alert(alerts, RiskLevel.LOW, "抽样策略", "系统抽样周期性风险",
                   "系统抽样假设数据无周期性规律。若数据按某属性排序，可能引入系统性偏差。",
                   "若不确定，请先打乱数据顺序或改用随机抽样。")


def analyze_risks(
    population: pd.DataFrame,
    sample: pd.DataFrame,
    rule: SamplingRule,
    report: Optional[CoverageReport] = None,
    thresholds: Optional[Dict[str, Any]] = None,
) -> RiskAnalysis:
    """执行风险分析."""
    thresholds = thresholds or {}
    alerts: List[RiskAlert] = []

    if report is None:
        from .report import generate_report
        report = generate_report(
            population, sample,
            batch_col=rule.stratify_by if rule.method.value == "stratified" else None,
        )

    _check_sample_size(len(population), len(sample), rule, alerts)
    _check_coverage(report, alerts,
                    min_batch_coverage=thresholds.get("min_batch_coverage", 0.9),
                    min_dim_coverage=thresholds.get("min_dim_coverage", 0.85))
    _check_small_batches(report, alerts,
                         min_per_batch=thresholds.get("min_per_batch", 2))
    _check_numerical_bias(report, alerts,
                          bias_threshold=thresholds.get("bias_threshold", 0.2))
    _check_method_vs_data(population, rule, alerts)

    if not alerts:
        _add_alert(alerts, RiskLevel.LOW, "综合", "未发现显著风险",
                   "当前抽样方案在样本量、覆盖度与统计偏差方面均通过检查。",
                   "可继续执行质检流程，同时保留抽样记录以便追溯。")

    alerts.sort(key=lambda a: (
        {RiskLevel.CRITICAL: 0, RiskLevel.HIGH: 1, RiskLevel.MEDIUM: 2, RiskLevel.LOW: 3}[a.level],
        a.category, a.title,
    ))

    return RiskAnalysis(alerts=alerts)


def risks_to_text(analysis: RiskAnalysis) -> str:
    """将风险分析结果渲染为可读文本."""
    lines = []
    lines.append("=" * 60)
    lines.append("抽样风险提示")
    lines.append("=" * 60)
    lines.append("")

    lc = analysis.level_counts
    lines.append(f"风险统计 - CRITICAL:{lc.get('CRITICAL', 0)}  HIGH:{lc.get('HIGH', 0)}  "
                 f"MEDIUM:{lc.get('MEDIUM', 0)}  LOW:{lc.get('LOW', 0)}")
    lines.append("")

    for idx, alert in enumerate(analysis.alerts, 1):
        tag = f"[{alert.level.value}]"
        lines.append(f"{idx}. {tag} {alert.category} - {alert.title}")
        lines.append(f"   描述: {alert.description}")
        lines.append(f"   建议: {alert.suggestion}")
        if alert.data:
            detail = ", ".join(f"{k}={v}" for k, v in alert.data.items() if not isinstance(v, list))
            if detail:
                lines.append(f"   详情: {detail}")
        lines.append("")

    if analysis.has_critical:
        lines.append(">>> 存在 CRITICAL 级风险，建议先修正抽样方案再继续质检 <<<")
    elif analysis.has_high:
        lines.append(">>> 存在 HIGH 级风险，请评估影响并考虑建议 <<<")

    lines.append("=" * 60)
    return "\n".join(lines)
