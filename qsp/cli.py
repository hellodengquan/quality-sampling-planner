"""CLI 入口 - 质检抽样规划命令行工具."""

from __future__ import annotations

import json as _json
import os
import sys
from typing import List, Optional

import click
import pandas as pd
import yaml

from .data_loader import BatchDataset, load_batch_data, validate_dataset
from .report import generate_report, report_to_text
from .risk import analyze_risks, risks_to_text
from .sampling import SamplingRule, SamplingMethod, SampleSizeMode, apply_sampling, load_rule_from_config


EXIT_OK = 0
EXIT_GENERAL_ERROR = 1
EXIT_SAMPLING_ERROR = 2
EXIT_RISK_CRITICAL = 3
EXIT_RISK_HIGH = 4
EXIT_DATA_ERROR = 5
EXIT_IO_ERROR = 6
EXIT_CONFIG_MISSING = 7
EXIT_RULE_INVALID = 8
EXIT_DATA_MISSING_COL = 9


@click.group()
@click.version_option(version="0.1.0", prog_name="qsp")
def cli():
    """质检抽样规划 CLI - 批次数据导入、抽样规则配置、覆盖率报告、风险提示."""
    pass


class ConfigMissingError(Exception):
    pass


class RuleInvalidError(Exception):
    pass


class DataMissingColumnError(Exception):
    pass


def _load_rule(ctx, param, value) -> Optional[SamplingRule]:
    if not value:
        return None
    if not os.path.exists(value):
        ctx.meta["rule_load_error"] = ConfigMissingError(f"规则文件不存在: {value}")
        return None
    ext = os.path.splitext(value)[1].lower().lstrip(".")
    try:
        with open(value, "r", encoding="utf-8") as f:
            if ext in ("yaml", "yml"):
                cfg = yaml.safe_load(f) or {}
            elif ext == "json":
                cfg = _json.load(f) or {}
            else:
                ctx.meta["rule_load_error"] = RuleInvalidError(
                    f"规则文件仅支持 .yaml/.yml/.json，收到 .{ext}"
                )
                return None
    except Exception as e:
        ctx.meta["rule_load_error"] = RuleInvalidError(f"规则文件解析失败: {e}")
        return None
    try:
        return load_rule_from_config(cfg)
    except (ValueError, KeyError) as e:
        ctx.meta["rule_load_error"] = RuleInvalidError(f"规则配置非法: {e}")
        return None


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--batch-col", "-b", default=None, help="批次列名")
@click.option("--format", "fmt", default=None, help="强制指定格式 (csv/excel/json/yaml)")
@click.option("--output", "-o", type=click.Path(), default=None, help="导出校验结果到 JSON")
def inspect(input_path: str, batch_col: Optional[str], fmt: Optional[str], output: Optional[str]):
    """检查批次数据并输出基本信息."""
    try:
        ds = load_batch_data(input_path, batch_col=batch_col, fmt=fmt)
    except FileNotFoundError as e:
        click.echo(f"[ERROR] 文件不存在: {e}", err=True)
        sys.exit(EXIT_IO_ERROR)
    except ValueError as e:
        click.echo(f"[ERROR] 数据格式错误: {e}", err=True)
        sys.exit(EXIT_DATA_ERROR)
    except Exception as e:
        click.echo(f"[ERROR] 加载数据失败: {e}", err=True)
        sys.exit(EXIT_GENERAL_ERROR)

    click.echo("=" * 60)
    click.echo("数据概览")
    click.echo("=" * 60)
    click.echo(f"源文件: {ds.source}")
    click.echo(f"行数: {ds.rows}")
    click.echo(f"列数: {len(ds.columns)}")
    click.echo(f"列名: {', '.join(ds.columns)}")

    if ds.batch_col:
        batches = ds.get_batches()
        click.echo(f"批次列: {ds.batch_col}  (共 {len(batches)} 个批次)")
        if batches:
            show = batches[:10]
            click.echo(f"批次示例: {', '.join(show)}{' ...' if len(batches) > 10 else ''}")

    issues = validate_dataset(ds, required_cols=None)
    click.echo("")
    click.echo("数据校验:")
    if not issues:
        click.echo("  ✓ 未发现问题")
    else:
        for i in issues:
            click.echo(f"  ! {i}")

    if output:
        result = {
            "source": ds.source,
            "rows": ds.rows,
            "columns": ds.columns,
            "batch_col": ds.batch_col,
            "batches": ds.get_batches(),
            "issues": issues,
        }
        with open(output, "w", encoding="utf-8") as f:
            _json.dump(result, f, ensure_ascii=False, indent=2)
        click.echo(f"\n已导出检查结果: {output}")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--rule-file", "-r", callback=_load_rule, help="抽样规则 YAML/JSON 文件")
@click.option("--method", type=click.Choice(["random", "simple", "stratified", "systematic", "cluster", "pps"]),
              default=None, help="抽样方法 (覆盖规则文件)")
@click.option("--mode", type=click.Choice(["fixed", "percentage", "statistical"]),
              default=None, help="样本量模式")
@click.option("--sample-size", "-n", type=int, default=0, help="fixed 模式下样本量")
@click.option("--percentage", "-p", type=float, default=0.0, help="percentage 模式下比例 0~1")
@click.option("--stratify-by", default=None, help="分层抽样列名")
@click.option("--cluster-by", default=None, help="整群抽样列名")
@click.option("--cluster-min-clusters", type=int, default=None, help="整群抽样最小群数")
@click.option("--pps-size-col", default=None, help="PPS 抽样规模列名")
@click.option("--pps-replace/--pps-no-replace", default=None, help="PPS 抽样是否有放回")
@click.option("--pps-fallback/--pps-no-fallback", default=None,
              help="PPS 极不均衡时是否自动回退 (默认开启)")
@click.option("--batch-col", "-b", default=None, help="批次列名 (用于报告)")
@click.option("--batches", "batch_filter", default=None, help="只处理指定批次，逗号分隔")
@click.option("--dimensions", "-d", default=None, help="覆盖率报告维度，逗号分隔")
@click.option("--output-sample", "-os", type=click.Path(), default=None, help="导出抽样结果 (CSV)")
@click.option("--output-report", type=click.Path(), default=None, help="导出覆盖率报告 (JSON)")
@click.option("--fail-on-critical/--no-fail-on-critical", default=True,
              help="存在 CRITICAL 风险时返回非零退出码 (默认开启)")
@click.option("--fail-on-high/--no-fail-on-high", default=False,
              help="存在 HIGH 风险时返回非零退出码 (默认关闭)")
@click.option("--seed", type=int, default=42, help="随机种子")
@click.option("--quiet", "-q", is_flag=True, help="仅输出警告和错误")
def plan(
    input_path: str,
    rule_file: Optional[SamplingRule],
    method: Optional[str],
    mode: Optional[str],
    sample_size: int,
    percentage: float,
    stratify_by: Optional[str],
    cluster_by: Optional[str],
    cluster_min_clusters: Optional[int],
    pps_size_col: Optional[str],
    pps_replace: Optional[bool],
    pps_fallback: Optional[bool],
    batch_col: Optional[str],
    batch_filter: Optional[str],
    dimensions: Optional[str],
    output_sample: Optional[str],
    output_report: Optional[str],
    fail_on_critical: bool,
    fail_on_high: bool,
    seed: int,
    quiet: bool,
):
    """执行抽样规划并生成报告与风险提示."""
    exit_code = EXIT_OK

    try:
        ds = load_batch_data(input_path, batch_col=batch_col)
    except FileNotFoundError as e:
        click.echo(f"[ERROR] 文件不存在: {e}", err=True)
        sys.exit(EXIT_IO_ERROR)
    except ValueError as e:
        click.echo(f"[ERROR] 数据格式错误: {e}", err=True)
        sys.exit(EXIT_DATA_ERROR)
    except Exception as e:
        click.echo(f"[ERROR] 加载数据失败: {e}", err=True)
        sys.exit(EXIT_GENERAL_ERROR)

    issues = validate_dataset(ds, required_cols=None)
    if issues and not quiet:
        click.echo("[WARN] 数据校验提示:")
        for i in issues:
            click.echo(f"       {i}")

    rule_load_error = click.get_current_context().meta.get("rule_load_error")
    if rule_load_error is not None:
        if isinstance(rule_load_error, ConfigMissingError):
            click.echo(f"[ERROR] 配置缺失: {rule_load_error}", err=True)
            sys.exit(EXIT_CONFIG_MISSING)
        elif isinstance(rule_load_error, RuleInvalidError):
            click.echo(f"[ERROR] 规则非法: {rule_load_error}", err=True)
            sys.exit(EXIT_RULE_INVALID)

    try:
        rule = rule_file or SamplingRule()
    except ConfigMissingError as e:
        click.echo(f"[ERROR] 配置缺失: {e}", err=True)
        sys.exit(EXIT_CONFIG_MISSING)
    except RuleInvalidError as e:
        click.echo(f"[ERROR] 规则非法: {e}", err=True)
        sys.exit(EXIT_RULE_INVALID)

    rule.random_seed = seed
    if method:
        try:
            rule.method = SamplingMethod(method)
        except ValueError as e:
            click.echo(f"[ERROR] 非法抽样方法 '{method}': {e}", err=True)
            sys.exit(EXIT_RULE_INVALID)
    if mode:
        try:
            rule.mode = SampleSizeMode(mode)
        except ValueError as e:
            click.echo(f"[ERROR] 非法样本量模式 '{mode}': {e}", err=True)
            sys.exit(EXIT_RULE_INVALID)
    if sample_size > 0:
        rule.sample_size = sample_size
    if percentage > 0:
        rule.percentage = percentage
    if stratify_by:
        rule.stratify_by = stratify_by
    if cluster_by:
        rule.cluster_by = cluster_by
    if cluster_min_clusters is not None:
        rule.cluster_min_clusters = cluster_min_clusters
    if pps_size_col:
        rule.pps_size_col = pps_size_col
    if pps_replace is not None:
        rule.pps_replace = pps_replace
    if pps_fallback is not None:
        rule.pps_fallback = pps_fallback

    required_cols = set()
    if rule.stratify_by:
        required_cols.add(rule.stratify_by)
    if rule.cluster_by:
        required_cols.add(rule.cluster_by)
    if rule.pps_size_col:
        required_cols.add(rule.pps_size_col)

    effective_batch_col = batch_col or rule.stratify_by
    if effective_batch_col:
        required_cols.add(effective_batch_col)

    missing_cols = [c for c in required_cols if c not in ds.columns]
    if missing_cols:
        click.echo(f"[ERROR] 数据缺少必需列: {missing_cols}", err=True)
        sys.exit(EXIT_DATA_MISSING_COL)

    if batch_filter:
        selected = [x.strip() for x in batch_filter.split(",") if x.strip()]
        if selected:
            population = ds.filter_by_batch(selected)
            if population.empty:
                click.echo(f"[ERROR] 批次过滤后无数据，检查: {selected}", err=True)
                sys.exit(EXIT_DATA_ERROR)
        else:
            population = ds.df.copy()
    else:
        population = ds.df.copy()

    try:
        sample = apply_sampling(population, rule)
    except ValueError as e:
        err_msg = str(e)
        if "需要指定" in err_msg:
            click.echo(f"[ERROR] 抽样参数缺失: {e}", err=True)
            sys.exit(EXIT_CONFIG_MISSING)
        elif "极不均衡" in err_msg or "无法" in err_msg:
            click.echo(f"[ERROR] 抽样规则无法执行: {e}", err=True)
            sys.exit(EXIT_RULE_INVALID)
        else:
            click.echo(f"[ERROR] 抽样参数错误: {e}", err=True)
            sys.exit(EXIT_SAMPLING_ERROR)
    except Exception as e:
        click.echo(f"[ERROR] 抽样执行失败: {e}", err=True)
        sys.exit(EXIT_SAMPLING_ERROR)

    dim_list = [x.strip() for x in dimensions.split(",")] if dimensions else None
    report = generate_report(population, sample, dimensions=dim_list, batch_col=effective_batch_col)
    risk = analyze_risks(population, sample, rule, report=report)

    if not quiet:
        click.echo("")
        click.echo(report_to_text(report))
        click.echo("")
        click.echo(risks_to_text(risk))

    if output_sample:
        try:
            sample.to_csv(output_sample, index=False, encoding="utf-8-sig")
            if not quiet:
                click.echo(f"\n抽样结果已导出: {output_sample} ({len(sample)} 行)")
        except Exception as e:
            click.echo(f"[ERROR] 写出抽样结果失败: {e}", err=True)
            sys.exit(EXIT_IO_ERROR)

    if output_report:
        data = {
            "population_size": report.population_size,
            "sample_size": report.sample_size,
            "overall_rate": report.overall_rate,
            "batch_stats": (
                {
                    "dimension": report.batch_stats.dimension,
                    "total": report.batch_stats.total_categories,
                    "covered": report.batch_stats.covered_categories,
                    "rate": report.batch_stats.coverage_rate,
                    "uncovered": report.batch_stats.uncovered,
                    "details": report.batch_stats.details,
                }
                if report.batch_stats else None
            ),
            "dimensions": {
                dim: {
                    "total": cs.total_categories,
                    "covered": cs.covered_categories,
                    "rate": cs.coverage_rate,
                    "uncovered": cs.uncovered,
                    "details": cs.details,
                }
                for dim, cs in report.dimension_stats.items()
            },
            "risks": [
                {
                    "level": a.level.value,
                    "category": a.category,
                    "title": a.title,
                    "description": a.description,
                    "suggestion": a.suggestion,
                    "data": a.data,
                }
                for a in risk.alerts
            ],
        }
        try:
            with open(output_report, "w", encoding="utf-8") as f:
                _json.dump(data, f, ensure_ascii=False, indent=2)
            if not quiet:
                click.echo(f"覆盖率与风险报告已导出: {output_report}")
        except Exception as e:
            click.echo(f"[ERROR] 写出报告失败: {e}", err=True)
            sys.exit(EXIT_IO_ERROR)

    if risk.has_critical and fail_on_critical:
        exit_code = EXIT_RISK_CRITICAL
    elif risk.has_high and fail_on_high:
        exit_code = EXIT_RISK_HIGH

    sys.exit(exit_code)


@cli.command()
@click.argument("output_path", type=click.Path(dir_okay=False))
@click.option("--format", "fmt", type=click.Choice(["yaml", "json"]), default="yaml")
def example_rule(output_path: str, fmt: str):
    """生成示例抽样规则配置文件."""
    cfg = {
        "method": "stratified",
        "mode": "statistical",
        "sample_size": 0,
        "percentage": 0.10,
        "confidence_level": 0.95,
        "margin_of_error": 0.05,
        "expected_defect_rate": 0.03,
        "stratify_by": "batch_id",
        "cluster_by": None,
        "cluster_min_clusters": 2,
        "pps_size_col": None,
        "pps_replace": True,
        "random_seed": 42,
        "min_per_group": 2,
        "weights": {},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        if fmt == "yaml":
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        else:
            _json.dump(cfg, f, ensure_ascii=False, indent=2)
    click.echo(f"示例规则已生成: {output_path}")


if __name__ == "__main__":
    cli()
