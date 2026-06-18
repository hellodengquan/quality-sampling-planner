"""CLI 测试 - 退出码与命令功能验证."""

import os

import pandas as pd
import pytest
from click.testing import CliRunner

from qsp.cli import (
    EXIT_DATA_ERROR,
    EXIT_GENERAL_ERROR,
    EXIT_IO_ERROR,
    EXIT_OK,
    EXIT_RISK_CRITICAL,
    EXIT_RISK_HIGH,
    EXIT_SAMPLING_ERROR,
    cli,
)


@pytest.fixture
def runner():
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


@pytest.fixture
def sample_csv(tmp_path):
    p = tmp_path / "batches.csv"
    pd.DataFrame({
        "id": list(range(200)),
        "batch_id": ["A"] * 80 + ["B"] * 60 + ["C"] * 60,
        "weight": [10 + i for i in range(200)],
        "value": [i * 2 for i in range(200)],
    }).to_csv(p, index=False)
    return str(p)


@pytest.fixture
def tiny_csv(tmp_path):
    p = tmp_path / "tiny.csv"
    pd.DataFrame({
        "id": [1, 2, 3],
        "batch_id": ["X", "X", "Y"],
        "size": [100, 200, 300],
    }).to_csv(p, index=False)
    return str(p)


class TestInspectExitCodes:
    def test_inspect_ok(self, runner, sample_csv):
        result = runner.invoke(cli, ["inspect", sample_csv, "-b", "batch_id"])
        assert result.exit_code == EXIT_OK
        assert "数据概览" in result.stdout

    def test_inspect_missing_file_click_handles(self, runner, tmp_path):
        p = str(tmp_path / "no_such.csv")
        result = runner.invoke(cli, ["inspect", p])
        assert result.exit_code != EXIT_OK

    def test_inspect_invalid_batch_col(self, runner, sample_csv):
        result = runner.invoke(cli, ["inspect", sample_csv, "-b", "no_such_col"])
        assert result.exit_code == EXIT_OK
        assert "批次列" in result.stdout
        assert "无有效值" in result.output or "不存在" in result.output


class TestPlanExitCodes:
    def test_plan_ok_default(self, runner, sample_csv):
        result = runner.invoke(cli, [
            "plan", sample_csv,
            "--method", "random",
            "--mode", "percentage",
            "-p", "0.1",
            "-q",
        ])
        assert result.exit_code == EXIT_OK

    def test_plan_stratified_ok(self, runner, sample_csv):
        result = runner.invoke(cli, [
            "plan", sample_csv,
            "--method", "stratified",
            "--stratify-by", "batch_id",
            "--mode", "fixed",
            "-n", "30",
            "-q",
        ])
        assert result.exit_code == EXIT_OK

    def test_plan_missing_stratify_param(self, runner, sample_csv):
        result = runner.invoke(cli, [
            "plan", sample_csv,
            "--method", "stratified",
            "-q",
        ])
        assert result.exit_code == EXIT_SAMPLING_ERROR

    def test_plan_missing_pps_param(self, runner, sample_csv):
        result = runner.invoke(cli, [
            "plan", sample_csv,
            "--method", "pps",
            "-q",
        ])
        assert result.exit_code == EXIT_SAMPLING_ERROR

    def test_plan_pps_ok(self, runner, sample_csv):
        result = runner.invoke(cli, [
            "plan", sample_csv,
            "--method", "pps",
            "--pps-size-col", "weight",
            "--mode", "fixed",
            "-n", "25",
            "-q",
        ])
        assert result.exit_code == EXIT_OK

    def test_plan_cluster_ok(self, runner, sample_csv):
        result = runner.invoke(cli, [
            "plan", sample_csv,
            "--method", "cluster",
            "--cluster-by", "batch_id",
            "--cluster-min-clusters", "2",
            "--mode", "fixed",
            "-n", "50",
            "-q",
        ])
        assert result.exit_code == EXIT_OK

    def test_plan_invalid_batch_col(self, runner, sample_csv):
        result = runner.invoke(cli, [
            "plan", sample_csv,
            "-b", "no_such_col",
            "-q",
        ])
        assert result.exit_code == EXIT_DATA_ERROR

    def test_plan_batch_filter_empty(self, runner, sample_csv):
        result = runner.invoke(cli, [
            "plan", sample_csv,
            "-b", "batch_id",
            "--batches", "NOT_EXIST",
            "-q",
        ])
        assert result.exit_code == EXIT_DATA_ERROR

    def test_plan_output_sample(self, runner, sample_csv, tmp_path):
        out = str(tmp_path / "sample.csv")
        result = runner.invoke(cli, [
            "plan", sample_csv,
            "--method", "random",
            "-n", "20",
            "--mode", "fixed",
            "-os", out,
            "-q",
        ])
        assert result.exit_code == EXIT_OK
        assert os.path.exists(out)
        df = pd.read_csv(out)
        assert len(df) == 20

    def test_plan_output_report(self, runner, sample_csv, tmp_path):
        out = str(tmp_path / "report.json")
        result = runner.invoke(cli, [
            "plan", sample_csv,
            "-b", "batch_id",
            "-d", "batch_id",
            "-n", "50",
            "--mode", "fixed",
            "--output-report", out,
            "-q",
        ])
        assert result.exit_code == EXIT_OK
        assert os.path.exists(out)

    def test_plan_critical_risk_exit_code(self, runner, tiny_csv):
        result = runner.invoke(cli, [
            "plan", tiny_csv,
            "-b", "batch_id",
            "-n", "1",
            "--mode", "fixed",
            "--fail-on-critical",
            "-q",
        ])
        assert result.exit_code == EXIT_RISK_CRITICAL

    def test_plan_no_fail_on_critical(self, runner, tiny_csv):
        result = runner.invoke(cli, [
            "plan", tiny_csv,
            "-b", "batch_id",
            "-n", "1",
            "--mode", "fixed",
            "--no-fail-on-critical",
            "-q",
        ])
        assert result.exit_code == EXIT_OK

    def test_plan_fail_on_high(self, runner, sample_csv):
        result = runner.invoke(cli, [
            "plan", sample_csv,
            "-b", "batch_id",
            "--method", "random",
            "-n", "1",
            "--mode", "fixed",
            "--fail-on-high",
            "--no-fail-on-critical",
            "-q",
        ])
        assert result.exit_code == EXIT_RISK_HIGH


class TestExampleRuleCommand:
    def test_example_rule_yaml(self, runner, tmp_path):
        out = str(tmp_path / "rule.yaml")
        result = runner.invoke(cli, ["example-rule", out])
        assert result.exit_code == EXIT_OK
        assert os.path.exists(out)
        with open(out, "r") as f:
            content = f.read()
        assert "method" in content
        assert "pps_size_col" in content

    def test_example_rule_json(self, runner, tmp_path):
        out = str(tmp_path / "rule.json")
        result = runner.invoke(cli, ["example-rule", out, "--format", "json"])
        assert result.exit_code == EXIT_OK
        assert os.path.exists(out)


class TestExitConstants:
    def test_exit_codes_distinct(self):
        codes = {
            EXIT_OK,
            EXIT_GENERAL_ERROR,
            EXIT_SAMPLING_ERROR,
            EXIT_RISK_CRITICAL,
            EXIT_RISK_HIGH,
            EXIT_DATA_ERROR,
            EXIT_IO_ERROR,
        }
        assert len(codes) == 7

    def test_exit_ok_is_zero(self):
        assert EXIT_OK == 0
