"""风险分析模块测试."""

import pandas as pd
import pytest

from qsp.report import generate_report
from qsp.risk import (
    RiskAlert,
    RiskAnalysis,
    RiskLevel,
    analyze_risks,
    risks_to_text,
)
from qsp.sampling import SamplingMethod, SamplingRule


@pytest.fixture
def large_population():
    N = 1200
    return pd.DataFrame(
        {
            "id": list(range(N)),
            "batch": (["A"] * 300 + ["B"] * 300 + ["C"] * 300 + ["D"] * 300),
            "line": (["L1"] * 150 + ["L2"] * 150) * 4,
            "value": [i * 1.1 for i in range(N)],
        }
    )


class TestRiskDataStructures:
    def test_level_counts(self):
        a = RiskAnalysis(
            alerts=[
                RiskAlert(RiskLevel.CRITICAL, "a", "t", "d", "s"),
                RiskAlert(RiskLevel.HIGH, "b", "t", "d", "s"),
                RiskAlert(RiskLevel.HIGH, "c", "t", "d", "s"),
            ]
        )
        assert a.level_counts == {"LOW": 0, "MEDIUM": 0, "HIGH": 2, "CRITICAL": 1}
        assert a.has_critical
        assert a.has_high

    def test_by_level(self):
        a = RiskAnalysis(
            alerts=[
                RiskAlert(RiskLevel.LOW, "x", "t1", "d", "s"),
                RiskAlert(RiskLevel.HIGH, "y", "t2", "d", "s"),
                RiskAlert(RiskLevel.LOW, "z", "t3", "d", "s"),
            ]
        )
        assert len(a.by_level(RiskLevel.LOW)) == 2
        assert len(a.by_level(RiskLevel.MEDIUM)) == 0


class TestSampleSizeChecks:
    def test_extreme_under_sampling(self, large_population):
        sample = large_population.sample(5, random_state=0)
        rule = SamplingRule()
        analysis = analyze_risks(large_population, sample, rule)
        titles = [a.title for a in analysis.alerts]
        assert "抽样率过低" in titles
        assert "统计意义样本不足" in titles

    def test_zero_sample(self, large_population):
        sample = large_population.head(0)
        rule = SamplingRule()
        analysis = analyze_risks(large_population, sample, rule)
        assert any(a.level == RiskLevel.CRITICAL for a in analysis.alerts)

    def test_sample_sizes_pass(self):
        pop = pd.DataFrame({"id": list(range(100)), "v": list(range(100))})
        sample = pop.sample(30, random_state=0)
        rule = SamplingRule()
        analysis = analyze_risks(pop, sample, rule)
        titles = [a.title for a in analysis.alerts]
        assert "抽样率过低" not in titles
        assert "统计意义样本不足" not in titles


class TestCoverageChecks:
    def test_bad_batch_coverage(self, large_population):
        sample = large_population[large_population["batch"] == "A"].sample(
            30, random_state=0
        )
        rule = SamplingRule()
        report = generate_report(large_population, sample, batch_col="batch")
        analysis = analyze_risks(large_population, sample, rule, report=report)
        assert any("批次覆盖率不足" in a.title for a in analysis.alerts)
        assert any("存在未抽样批次" in a.title for a in analysis.alerts)

    def test_small_per_batch(self, large_population):
        batches = []
        for b in ["A", "B", "C", "D"]:
            batches.append(
                large_population[large_population["batch"] == b].sample(
                    1, random_state=0
                )
            )
        sample = pd.concat(batches, ignore_index=True)
        rule = SamplingRule()
        report = generate_report(large_population, sample, batch_col="batch")
        analysis = analyze_risks(large_population, sample, rule, report=report)
        assert any("抽样数量过少" in a.title for a in analysis.alerts)


class TestNumericalBias:
    def test_bias_detected(self):
        pop = pd.DataFrame({"v": list(range(1000))})
        sample = pd.DataFrame({"v": [900, 910, 920, 930, 940, 950, 960, 970, 980, 990]})
        rule = SamplingRule()
        analysis = analyze_risks(pop, sample, rule)
        titles = [a.title for a in analysis.alerts]
        assert any("均值偏离较大" in t for t in titles)


class TestMethodChecks:
    def test_systematic_periodicity_warning(self, large_population):
        sample = large_population.sample(50, random_state=0)
        rule = SamplingRule(method=SamplingMethod.SYSTEMATIC)
        analysis = analyze_risks(large_population, sample, rule)
        assert any("周期性风险" in a.title for a in analysis.alerts)

    def test_stratified_sparse_dimension(self):
        pop = pd.DataFrame(
            {
                "id": list(range(50)),
                "unique_tag": ["tag_" + str(i) for i in range(50)],
            }
        )
        sample = pop.sample(20, random_state=0)
        rule = SamplingRule(method=SamplingMethod.STRATIFIED, stratify_by="unique_tag")
        analysis = analyze_risks(pop, sample, rule)
        assert any("分层维度过稀疏" in a.title for a in analysis.alerts)


class TestRiskTextOutput:
    def test_risks_to_text(self, large_population):
        sample = large_population[large_population["batch"] == "A"].sample(
            5, random_state=0
        )
        rule = SamplingRule()
        report = generate_report(large_population, sample, batch_col="batch")
        analysis = analyze_risks(large_population, sample, rule, report=report)
        text = risks_to_text(analysis)
        assert "抽样风险提示" in text
        assert "风险统计" in text
        if analysis.has_critical:
            assert "CRITICAL 级风险" in text

    def test_no_risks_shows_green(self):
        pop = pd.DataFrame({"id": list(range(80)), "v": list(range(80))})
        sample = pop.sample(20, random_state=0)
        rule = SamplingRule()
        analysis = analyze_risks(pop, sample, rule)
        text = risks_to_text(analysis)
        assert "未发现显著风险" in text


class TestThresholds:
    def test_custom_thresholds(self, large_population):
        sample = large_population[large_population["batch"].isin(["A", "B"])].sample(
            50, random_state=0
        )
        rule = SamplingRule()
        report = generate_report(large_population, sample, batch_col="batch")
        analysis = analyze_risks(
            large_population,
            sample,
            rule,
            report=report,
            thresholds={"min_batch_coverage": 0.3},
        )
        titles = [a.title for a in analysis.alerts]
        assert "批次覆盖率不足" not in titles
