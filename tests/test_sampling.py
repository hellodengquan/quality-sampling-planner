"""抽样规则引擎测试."""

import math

import pandas as pd
import pytest

from qsp.sampling import (
    SamplingMethod,
    SampleSizeMode,
    SamplingRule,
    apply_sampling,
    calculate_statistical_sample_size,
    cluster_sample,
    determine_sample_size,
    load_rule_from_config,
    pps_sample,
    simple_random_sample,
    stratified_sample,
    systematic_sample,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": list(range(200)),
        "batch_id": ["A"] * 50 + ["B"] * 50 + ["C"] * 50 + ["D"] * 50,
        "category": (["X"] * 25 + ["Y"] * 25) * 4,
        "value": [i * 1.5 for i in range(200)],
    })


@pytest.fixture
def small_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "batch_id": ["A", "A", "B", "B", "C", "C"],
        "value": [10, 20, 30, 40, 50, 60],
    })


class TestSampleSizeCalculation:
    def test_statistical_sample_size_small_pop(self):
        n = calculate_statistical_sample_size(100, 0.95, 0.05, 0.05)
        assert n >= 1
        assert n <= 100

    def test_statistical_sample_size_large_pop(self):
        n = calculate_statistical_sample_size(100000, 0.95, 0.05, 0.05)
        assert n == pytest.approx(73, abs=5)

    def test_determine_size_fixed(self):
        rule = SamplingRule(mode=SampleSizeMode.FIXED, sample_size=20)
        assert determine_sample_size(1000, rule) == 20

    def test_determine_size_percentage(self):
        rule = SamplingRule(mode=SampleSizeMode.PERCENTAGE, percentage=0.15)
        assert determine_sample_size(100, rule) == 15

    def test_determine_size_at_least_one(self):
        rule = SamplingRule(mode=SampleSizeMode.PERCENTAGE, percentage=0.001)
        assert determine_sample_size(10, rule) == 1

    def test_determine_size_cap_at_population(self):
        rule = SamplingRule(mode=SampleSizeMode.FIXED, sample_size=9999)
        assert determine_sample_size(10, rule) == 10

    def test_determine_size_statistical(self):
        rule = SamplingRule(mode=SampleSizeMode.STATISTICAL, confidence_level=0.95)
        size = determine_sample_size(5000, rule)
        assert size > 50
        assert size < 5000


class TestSimpleRandomSample:
    def test_size(self, sample_df):
        result = simple_random_sample(sample_df, 30, seed=42)
        assert len(result) == 30

    def test_seed_reproducible(self, sample_df):
        a = simple_random_sample(sample_df, 20, seed=7)
        b = simple_random_sample(sample_df, 20, seed=7)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    def test_empty_df_raises(self):
        with pytest.raises(Exception):
            simple_random_sample(pd.DataFrame(), 5)


class TestSystematicSample:
    def test_size(self, sample_df):
        result = systematic_sample(sample_df, 20, seed=42)
        assert len(result) == 20

    def test_all_in_range(self, small_df):
        result = systematic_sample(small_df, 3, seed=1)
        assert len(result) == 3
        for idx in result.index:
            assert 0 <= idx < len(small_df)

    def test_n_equals_population(self, small_df):
        result = systematic_sample(small_df, len(small_df), seed=0)
        assert len(result) == len(small_df)


class TestStratifiedSample:
    def test_keeps_all_strata(self, sample_df):
        result = stratified_sample(sample_df, 40, "batch_id", seed=42, min_per_group=2)
        batches = result["batch_id"].unique()
        assert set(batches) == {"A", "B", "C", "D"}

    def test_min_per_group(self, sample_df):
        result = stratified_sample(sample_df, 10, "batch_id", seed=42, min_per_group=2)
        counts = result["batch_id"].value_counts()
        for c in counts:
            assert c >= 2

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="不存在"):
            stratified_sample(sample_df, 10, "no_such_col")

    def test_weights(self, sample_df):
        weights = {"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.1}
        result = stratified_sample(sample_df, 20, "batch_id", seed=42, weights=weights)
        counts = result["batch_id"].value_counts()
        assert counts.get("A", 0) >= counts.get("D", 0)


class TestClusterSample:
    def test_basic_size_and_clusters(self, sample_df):
        result = cluster_sample(sample_df, 60, "batch_id", seed=42, min_clusters=2)
        assert len(result) >= 2
        assert result["batch_id"].nunique() >= 2

    def test_min_clusters_respected(self, sample_df):
        result = cluster_sample(sample_df, 10, "batch_id", seed=42, min_clusters=3)
        assert result["batch_id"].nunique() >= 3

    def test_min_clusters_defaults_to_total(self, sample_df):
        result = cluster_sample(sample_df, 10, "batch_id", seed=42, min_clusters=100)
        assert result["batch_id"].nunique() == sample_df["batch_id"].nunique()

    def test_size_approx_target(self, sample_df):
        result = cluster_sample(sample_df, 40, "batch_id", seed=42, min_clusters=2)
        assert 1 <= len(result) <= 200

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="不存在"):
            cluster_sample(sample_df, 50, "no_such_col")

    def test_empty_clusters_raises(self):
        df = pd.DataFrame({"g": []})
        with pytest.raises(ValueError):
            cluster_sample(df, 5, "g")

    def test_seed_reproducible(self, sample_df):
        a = cluster_sample(sample_df, 50, "batch_id", seed=7, min_clusters=2)
        b = cluster_sample(sample_df, 50, "batch_id", seed=7, min_clusters=2)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


class TestPPSSample:
    def test_with_replacement_size(self, sample_df):
        result = pps_sample(sample_df, 30, "value", seed=42, replace=True)
        assert len(result) == 30

    def test_without_replacement_size_and_unique(self, sample_df):
        result = pps_sample(sample_df, 30, "value", seed=42, replace=False)
        assert len(result) == 30
        assert result["id"].nunique() == 30

    def test_large_size_equals_population_without_replacement(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "size": [10, 20, 30, 40, 50],
        })
        n = len(df)
        result = pps_sample(df, n, "size", seed=42, replace=False)
        assert len(result) == n
        assert result["id"].nunique() == n

    def test_with_replacement_may_have_duplicates(self, sample_df):
        result = pps_sample(sample_df, 200, "value", seed=42, replace=True)
        assert len(result) == 200

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="不存在"):
            pps_sample(sample_df, 10, "no_such_col")

    def test_negative_size_raises(self, sample_df):
        df = sample_df.copy()
        df["neg_val"] = -df["value"]
        with pytest.raises(ValueError, match="负值"):
            pps_sample(df, 10, "neg_val")

    def test_non_numeric_size_raises(self, sample_df):
        with pytest.raises(ValueError, match="非数值"):
            pps_sample(sample_df, 10, "batch_id")

    def test_zero_n_raises(self, sample_df):
        with pytest.raises(ValueError, match="正整数"):
            pps_sample(sample_df, 0, "value")

    def test_all_zero_size_raises(self):
        df = pd.DataFrame({"x": [0, 0, 0]})
        with pytest.raises(ValueError, match="总和非正"):
            pps_sample(df, 2, "x")

    def test_some_zero_without_replace_raises_when_n_too_large(self):
        df = pd.DataFrame({"x": [0, 1, 1, 1]})
        with pytest.raises(ValueError, match="有效样本数"):
            pps_sample(df, 4, "x", replace=False)

    def test_seed_reproducible(self, sample_df):
        a = pps_sample(sample_df, 25, "value", seed=99, replace=True)
        b = pps_sample(sample_df, 25, "value", seed=99, replace=True)
        pd.testing.assert_frame_equal(a, b)


class TestApplySampling:
    def test_random_method(self, sample_df):
        rule = SamplingRule(method=SamplingMethod.RANDOM, mode=SampleSizeMode.PERCENTAGE, percentage=0.1)
        result = apply_sampling(sample_df, rule)
        assert len(result) == math.ceil(len(sample_df) * 0.1)

    def test_stratified_requires_param(self, sample_df):
        rule = SamplingRule(method=SamplingMethod.STRATIFIED)
        with pytest.raises(ValueError, match="stratify_by"):
            apply_sampling(sample_df, rule)

    def test_cluster_requires_param(self, sample_df):
        rule = SamplingRule(method=SamplingMethod.CLUSTER)
        with pytest.raises(ValueError, match="cluster_by"):
            apply_sampling(sample_df, rule)

    def test_cluster_method(self, sample_df):
        rule = SamplingRule(
            method=SamplingMethod.CLUSTER,
            cluster_by="batch_id",
            cluster_min_clusters=2,
            sample_size=50,
            mode=SampleSizeMode.FIXED,
        )
        result = apply_sampling(sample_df, rule)
        assert len(result) > 0
        assert result["batch_id"].nunique() >= 2

    def test_pps_requires_param(self, sample_df):
        rule = SamplingRule(method=SamplingMethod.PPS)
        with pytest.raises(ValueError, match="pps_size_col"):
            apply_sampling(sample_df, rule)

    def test_pps_method(self, sample_df):
        rule = SamplingRule(
            method=SamplingMethod.PPS,
            pps_size_col="value",
            pps_replace=True,
            sample_size=25,
            mode=SampleSizeMode.FIXED,
        )
        result = apply_sampling(sample_df, rule)
        assert len(result) == 25

    def test_pps_method_without_replace(self, sample_df):
        rule = SamplingRule(
            method=SamplingMethod.PPS,
            pps_size_col="value",
            pps_replace=False,
            sample_size=25,
            mode=SampleSizeMode.FIXED,
        )
        result = apply_sampling(sample_df, rule)
        assert len(result) == 25
        assert result["id"].nunique() == 25

    def test_systematic_method(self, sample_df):
        rule = SamplingRule(method=SamplingMethod.SYSTEMATIC, sample_size=30,
                            mode=SampleSizeMode.FIXED, random_seed=0)
        result = apply_sampling(sample_df, rule)
        assert len(result) == 30

    def test_empty_df_raises(self):
        rule = SamplingRule()
        with pytest.raises(ValueError, match="空"):
            apply_sampling(pd.DataFrame(), rule)

    def test_unknown_method_raises(self, sample_df):
        rule = SamplingRule()
        rule.method = "invalid"
        with pytest.raises(ValueError, match="未知"):
            apply_sampling(sample_df, rule)


class TestLoadRuleFromConfig:
    def test_defaults(self):
        rule = load_rule_from_config({})
        assert rule.method == SamplingMethod.RANDOM
        assert rule.mode == SampleSizeMode.PERCENTAGE
        assert rule.percentage == 0.10

    def test_full_config(self):
        cfg = {
            "method": "stratified",
            "mode": "fixed",
            "sample_size": 50,
            "stratify_by": "batch",
            "random_seed": 123,
            "min_per_group": 3,
            "weights": {"A": 0.6, "B": 0.4},
        }
        rule = load_rule_from_config(cfg)
        assert rule.method == SamplingMethod.STRATIFIED
        assert rule.mode == SampleSizeMode.FIXED
        assert rule.sample_size == 50
        assert rule.stratify_by == "batch"
        assert rule.random_seed == 123
        assert rule.min_per_group == 3
        assert rule.weights == {"A": 0.6, "B": 0.4}

    def test_cluster_config(self):
        cfg = {
            "method": "cluster",
            "cluster_by": "group",
            "cluster_min_clusters": 3,
            "sample_size": 100,
            "mode": "fixed",
        }
        rule = load_rule_from_config(cfg)
        assert rule.method == SamplingMethod.CLUSTER
        assert rule.cluster_by == "group"
        assert rule.cluster_min_clusters == 3

    def test_pps_config(self):
        cfg = {
            "method": "pps",
            "pps_size_col": "weight",
            "pps_replace": False,
            "percentage": 0.15,
            "mode": "percentage",
        }
        rule = load_rule_from_config(cfg)
        assert rule.method == SamplingMethod.PPS
        assert rule.pps_size_col == "weight"
        assert rule.pps_replace is False
        assert rule.percentage == 0.15

    def test_pps_replace_default_true(self):
        cfg = {"method": "pps", "pps_size_col": "w"}
        rule = load_rule_from_config(cfg)
        assert rule.pps_replace is True
