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
    _detect_pps_skew,
    PPSSkewWarning,
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
        with pytest.raises(ValueError, match="全部为零"):
            pps_sample(df, 2, "x")

    def test_some_zero_without_replace_raises_when_n_too_large(self):
        df = pd.DataFrame({"x": [0, 1, 1, 1]})
        with pytest.raises(ValueError, match="排除零权重后有效样本"):
            pps_sample(df, 4, "x", replace=False)

    def test_seed_reproducible(self, sample_df):
        a = pps_sample(sample_df, 25, "value", seed=99, replace=True)
        b = pps_sample(sample_df, 25, "value", seed=99, replace=True)
        pd.testing.assert_frame_equal(a, b)


class TestPPSSkewDetection:
    def test_no_skew(self):
        sizes = pd.Series([10, 20, 30, 40])
        assert _detect_pps_skew(sizes, 3) is None

    def test_zero_weights_detected(self):
        sizes = pd.Series([0, 0, 10, 20])
        warn = _detect_pps_skew(sizes, 3)
        assert warn is not None
        assert warn.reason == "has_zero_weights"
        assert warn.n_imbalanced == 2

    def test_all_zero_detected(self):
        sizes = pd.Series([0, 0, 0])
        warn = _detect_pps_skew(sizes, 2)
        assert warn is not None
        assert warn.reason == "all_zero"

    def test_too_many_tiny_weights(self):
        sizes = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100])
        warn = _detect_pps_skew(sizes, 5, threshold=0.01)
        assert warn is not None
        assert warn.reason == "too_many_tiny_weights"

    def test_single_dominant_weight(self):
        sizes = pd.Series([10000, 1, 1, 1])
        warn = _detect_pps_skew(sizes, 4)
        assert warn is not None
        assert warn.reason == "single_dominant_weight"


class TestPPSFallbackBranches:
    def test_has_zero_weights_fallback_excludes_zeros(self):
        df = pd.DataFrame({
            "id": list(range(10)),
            "size": [0, 0, 0, 10, 20, 30, 40, 50, 60, 70],
        })
        result = pps_sample(df, 5, "size", seed=42, replace=False, fallback=True)
        assert len(result) == 5
        for _, row in result.iterrows():
            assert row["size"] > 0

    def test_has_zero_weights_no_fallback_raises(self):
        df = pd.DataFrame({
            "id": list(range(5)),
            "size": [0, 0, 10, 20, 30],
        })
        with pytest.raises(ValueError, match="极不均衡"):
            pps_sample(df, 3, "size", seed=42, fallback=False)

    def test_all_zero_fallback_raises(self):
        df = pd.DataFrame({"id": [1, 2, 3], "size": [0, 0, 0]})
        with pytest.raises(ValueError, match="全部为零"):
            pps_sample(df, 2, "size", seed=42, fallback=True)

    def test_too_many_tiny_weights_blend_fallback(self):
        sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 100000]
        df = pd.DataFrame({"id": list(range(10)), "size": sizes})
        result = pps_sample(df, 6, "size", seed=42, replace=False, fallback=True)
        assert len(result) == 6
        assert result["id"].nunique() == 6

    def test_single_dominant_weight_cap_fallback(self):
        df = pd.DataFrame({
            "id": list(range(6)),
            "size": [10000, 1, 1, 1, 1, 1],
        })
        result = pps_sample(df, 5, "size", seed=42, replace=False, fallback=True)
        assert len(result) == 5
        assert result["id"].nunique() == 5

    def test_single_dominant_cap_with_replace(self):
        df = pd.DataFrame({
            "id": list(range(6)),
            "size": [10000, 1, 1, 1, 1, 1],
        })
        result = pps_sample(df, 5, "size", seed=42, replace=False, fallback=True)
        assert len(result) == 5
        assert result["id"].nunique() == 5

    def test_single_dominant_cap_with_replace_allows_dup(self):
        df = pd.DataFrame({
            "id": list(range(5)),
            "size": [9999, 1, 1, 1, 1],
        })
        result = pps_sample(df, 8, "size", seed=42, replace=True, fallback=True)
        assert len(result) == 8

    def test_zero_weights_replace_fallback(self):
        df = pd.DataFrame({
            "id": list(range(8)),
            "size": [0, 0, 10, 20, 30, 40, 50, 60],
        })
        result = pps_sample(df, 5, "size", seed=42, replace=True, fallback=True)
        assert len(result) == 5
        for _, row in result.iterrows():
            assert row["size"] > 0

    def test_zero_weights_without_replace_n_exceeds_valid(self):
        df = pd.DataFrame({
            "id": list(range(5)),
            "size": [0, 0, 10, 20, 30],
        })
        with pytest.raises(ValueError, match="排除零权重后有效样本"):
            pps_sample(df, 5, "size", seed=42, replace=False, fallback=True)

    def test_blend_fallback_no_replace_small_n(self):
        sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 100000]
        df = pd.DataFrame({"id": list(range(10)), "size": sizes})
        result = pps_sample(df, 3, "size", seed=42, replace=False, fallback=True)
        assert len(result) == 3


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


class TestPPSExtremeRegression:
    def test_all_one_weights_replace(self):
        df = pd.DataFrame({
            "id": list(range(10)),
            "size": [1] * 10,
        })
        result = pps_sample(df, 5, "size", seed=42, replace=True)
        assert len(result) == 5

    def test_all_one_weights_no_replace(self):
        df = pd.DataFrame({
            "id": list(range(10)),
            "size": [1] * 10,
        })
        result = pps_sample(df, 5, "size", seed=42, replace=False)
        assert len(result) == 5
        assert result["id"].nunique() == 5

    def test_all_one_weights_no_replace_n_equals_population(self):
        df = pd.DataFrame({
            "id": list(range(5)),
            "size": [1] * 5,
        })
        result = pps_sample(df, 5, "size", seed=42, replace=False)
        assert len(result) == 5
        assert result["id"].nunique() == 5
        assert set(result["id"].tolist()) == {0, 1, 2, 3, 4}

    def test_all_one_weights_n_greater_than_population_clamped(self):
        df = pd.DataFrame({
            "id": list(range(3)),
            "size": [1] * 3,
        })
        result = pps_sample(df, 10, "size", seed=42, replace=False)
        assert len(result) == 3
        assert result["id"].nunique() == 3

    def test_all_one_weights_n_equals_one(self):
        df = pd.DataFrame({
            "id": list(range(100)),
            "size": [1] * 100,
        })
        result = pps_sample(df, 1, "size", seed=42, replace=False)
        assert len(result) == 1
        assert 0 <= result.iloc[0]["id"] < 100

    def test_all_zero_single_sample_raises(self):
        df = pd.DataFrame({"id": [1], "size": [0]})
        with pytest.raises(ValueError, match="全部为零"):
            pps_sample(df, 1, "size", seed=42)

    def test_single_sample_nonzero(self):
        df = pd.DataFrame({"id": [1], "size": [100]})
        result = pps_sample(df, 1, "size", seed=42, replace=False)
        assert len(result) == 1
        assert result.iloc[0]["id"] == 1

    def test_single_sample_replace_many(self):
        df = pd.DataFrame({"id": [1], "size": [100]})
        result = pps_sample(df, 5, "size", seed=42, replace=True)
        assert len(result) == 5
        assert (result["id"] == 1).all()

    def test_all_zero_fallback_true_raises(self):
        df = pd.DataFrame({"id": [1, 2, 3], "size": [0, 0, 0]})
        with pytest.raises(ValueError, match="全部为零"):
            pps_sample(df, 2, "size", seed=42, fallback=True)

    def test_all_one_weights_seed_reproducible(self):
        df = pd.DataFrame({
            "id": list(range(20)),
            "size": [1] * 20,
        })
        a = pps_sample(df, 8, "size", seed=99, replace=False)
        b = pps_sample(df, 8, "size", seed=99, replace=False)
        pd.testing.assert_frame_equal(a, b)

    def test_all_one_weights_should_not_trigger_skew(self):
        sizes = pd.Series([1] * 5)
        warn = _detect_pps_skew(sizes, 3, threshold=0.001)
        assert warn is None

    def test_all_one_weights_with_threshold_below_one(self):
        sizes = pd.Series([1] * 5)
        warn = _detect_pps_skew(sizes, 3, threshold=0.1)
        assert warn is None

    def test_nearly_equal_weights_no_skew(self):
        sizes = pd.Series([1, 1.1, 1.2, 0.9, 1.0])
        warn = _detect_pps_skew(sizes, 3)
        assert warn is None

    def test_weight_nan_raises(self):
        df = pd.DataFrame({"id": [1, 2, 3], "size": [1, None, 3]})
        with pytest.raises(ValueError, match="非数值或空值"):
            pps_sample(df, 2, "size")

    def test_weight_non_numeric_string_raises(self):
        df = pd.DataFrame({"id": [1, 2, 3], "size": [1, "abc", 3]})
        with pytest.raises(ValueError, match="非数值或空值"):
            pps_sample(df, 2, "size")

    def test_weight_zero_but_only_one_positive(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4], "size": [0, 0, 0, 100]})
        result = pps_sample(df, 1, "size", seed=42, replace=False, fallback=True)
        assert len(result) == 1
        assert result.iloc[0]["id"] == 4

    def test_weight_zero_but_only_one_positive_replace(self):
        df = pd.DataFrame({"id": [1, 2, 3], "size": [0, 0, 100]})
        result = pps_sample(df, 5, "size", seed=42, replace=True, fallback=True)
        assert len(result) == 5
        assert (result["id"] == 3).all()

    def test_weight_zero_no_fallback_raises(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4], "size": [0, 0, 0, 100]})
        with pytest.raises(ValueError, match="极不均衡"):
            pps_sample(df, 1, "size", seed=42, fallback=False)

    def test_all_one_high_imbalance_threshold(self):
        df = pd.DataFrame({
            "id": list(range(10)),
            "size": [1] * 10,
        })
        result = pps_sample(df, 3, "size", seed=42, replace=False, imbalance_threshold=0.1)
        assert len(result) == 3
        assert result["id"].nunique() == 3

    def test_apply_sampling_pps_all_one_weights(self):
        df = pd.DataFrame({
            "id": list(range(50)),
            "w": [1] * 50,
        })
        rule = SamplingRule(
            method=SamplingMethod.PPS,
            pps_size_col="w",
            pps_replace=False,
            sample_size=10,
            mode=SampleSizeMode.FIXED,
        )
        result = apply_sampling(df, rule)
        assert len(result) == 10
        assert result["id"].nunique() == 10
