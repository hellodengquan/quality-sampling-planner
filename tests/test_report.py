"""覆盖率报告模块测试."""

import pandas as pd
import pytest

from qsp.report import (
    compute_dimension_coverage,
    compute_numerical_summary,
    compute_overall_coverage,
    generate_report,
    report_to_text,
)


@pytest.fixture
def population():
    cat_pattern = ["X"] * 15 + ["Y"] * 10 + ["Z"] * 5
    cats = (cat_pattern * 4)[:100]
    return pd.DataFrame(
        {
            "id": list(range(100)),
            "batch": ["A"] * 30 + ["B"] * 40 + ["C"] * 30,
            "category": cats,
            "value": [i * 2 for i in range(100)],
        }
    )


@pytest.fixture
def good_sample(population):
    return pd.concat(
        [
            population[population["batch"] == "A"].sample(10, random_state=1),
            population[population["batch"] == "B"].sample(10, random_state=1),
            population[population["batch"] == "C"].sample(10, random_state=1),
        ],
        ignore_index=True,
    )


@pytest.fixture
def biased_sample(population):
    return (
        population[population["batch"] == "A"]
        .sample(20, random_state=2)
        .reset_index(drop=True)
    )


class TestOverallCoverage:
    def test_basic(self, population):
        sample = population.sample(20, random_state=0)
        assert compute_overall_coverage(population, sample) == pytest.approx(0.2)

    def test_empty_population(self):
        assert compute_overall_coverage(pd.DataFrame(), pd.DataFrame({"a": [1]})) == 0.0


class TestDimensionCoverage:
    def test_full_coverage(self, population, good_sample):
        cs = compute_dimension_coverage(population, good_sample, "batch")
        assert cs.coverage_rate == 1.0
        assert cs.uncovered == []

    def test_partial_coverage(self, population, biased_sample):
        cs = compute_dimension_coverage(population, biased_sample, "batch")
        assert cs.covered_categories == 1
        assert "B" in cs.uncovered
        assert "C" in cs.uncovered

    def test_details(self, population, good_sample):
        cs = compute_dimension_coverage(population, good_sample, "batch")
        for b in ["A", "B", "C"]:
            assert b in cs.details
            assert cs.details[b]["population"] > 0
            assert cs.details[b]["sample"] > 0

    def test_missing_column(self, population, good_sample):
        with pytest.raises(ValueError, match="不存在"):
            compute_dimension_coverage(population, good_sample, "no_col")


class TestNumericalSummary:
    def test_contains_both_sources(self, population, good_sample):
        df = compute_numerical_summary(population, good_sample, columns=["value"])
        sources = set(df["source"].tolist())
        assert sources == {"population", "sample"}

    def test_stats_present(self, population, good_sample):
        df = compute_numerical_summary(population, good_sample, columns=["value"])
        pop = df[df["source"] == "population"].iloc[0]
        assert pop["count"] == 100
        assert pop["mean"] == pytest.approx(99.0, rel=0.01)

    def test_default_numerical_columns(self, population, good_sample):
        df = compute_numerical_summary(population, good_sample)
        cols = set(df["column"].unique())
        assert "id" in cols
        assert "value" in cols
        assert "batch" not in cols


class TestGenerateReport:
    def test_report_structure(self, population, good_sample):
        report = generate_report(
            population,
            good_sample,
            dimensions=["category"],
            batch_col="batch",
        )
        assert report.population_size == 100
        assert report.sample_size == 30
        assert report.batch_stats is not None
        assert "category" in report.dimension_stats
        assert report.numerical_summary is not None

    def test_report_to_text(self, population, good_sample):
        report = generate_report(population, good_sample, batch_col="batch")
        text = report_to_text(report)
        assert "质检抽样覆盖率报告" in text
        assert "总体数量: 100" in text
        assert "总体抽样率:" in text
        assert "批次维度: batch" in text

    def test_no_batch_col(self, population, good_sample):
        report = generate_report(population, good_sample)
        assert report.batch_stats is None
        text = report_to_text(report)
        assert "批次维度" not in text
