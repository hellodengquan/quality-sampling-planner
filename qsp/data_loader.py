"""数据导入模块 - 支持多格式批次数据导入."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


@dataclass
class BatchDataset:
    """批次数据集."""

    df: pd.DataFrame
    source: str
    batch_col: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return len(self.df)

    @property
    def columns(self) -> List[str]:
        return list(self.df.columns)

    def get_batches(self) -> List[str]:
        """获取所有批次标识列表."""
        if self.batch_col and self.batch_col in self.df.columns:
            return sorted(
                self.df[self.batch_col].dropna().astype(str).unique().tolist()
            )
        return []

    def filter_by_batch(self, batch_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """按批次过滤数据."""
        if not batch_ids or not self.batch_col:
            return self.df.copy()
        mask = self.df[self.batch_col].astype(str).isin(batch_ids)
        return self.df[mask].copy()


def _detect_format(path: str) -> str:
    """根据文件后缀检测格式."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in ("csv", "txt"):
        return "csv"
    elif ext in ("xlsx", "xls"):
        return "excel"
    elif ext in ("json", "jsonl", "ndjson"):
        return "json"
    elif ext in ("yaml", "yml"):
        return "yaml"
    else:
        raise ValueError(f"不支持的文件格式: .{ext}")


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """加载 CSV 文件."""
    defaults = {"encoding": "utf-8-sig"}
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def load_excel(path: str, **kwargs) -> pd.DataFrame:
    """加载 Excel 文件."""
    return pd.read_excel(path, **kwargs)


def load_json(path: str, **kwargs) -> pd.DataFrame:
    """加载 JSON / JSON Lines 文件."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in ("jsonl", "ndjson"):
        return pd.read_json(path, lines=True, **kwargs)
    try:
        return pd.read_json(path, lines=False, **kwargs)
    except ValueError:
        return pd.read_json(path, lines=True, **kwargs)


def load_yaml(path: str) -> pd.DataFrame:
    """加载 YAML 数据文件."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            return pd.DataFrame(data["records"])
        return pd.DataFrame([data])
    else:
        raise ValueError("YAML 文件格式不支持，需为列表或包含 records 的字典")


def load_dataframe(path: str, fmt: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """通用 DataFrame 加载函数."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"路径不是文件: {path}")

    fmt = (fmt or _detect_format(path)).lower()

    if fmt == "csv":
        return load_csv(path, **kwargs)
    elif fmt == "excel":
        return load_excel(path, **kwargs)
    elif fmt == "json":
        return load_json(path, **kwargs)
    elif fmt == "yaml":
        return load_yaml(path)
    else:
        raise ValueError(f"未知的格式: {fmt}")


def load_batch_data(
    path: str,
    batch_col: Optional[str] = None,
    fmt: Optional[str] = None,
    **kwargs,
) -> BatchDataset:
    """加载批次数据集."""
    df = load_dataframe(path, fmt=fmt, **kwargs)
    meta = {"path": path, "rows": len(df), "columns": list(df.columns)}
    return BatchDataset(df=df, source=path, batch_col=batch_col, meta=meta)


def validate_dataset(
    ds: BatchDataset, required_cols: Optional[List[str]] = None
) -> List[str]:
    """校验数据集，返回问题列表."""
    issues: List[str] = []
    if ds.rows == 0:
        issues.append("数据集为空（0 行）")
    if not ds.columns:
        issues.append("数据集没有列")
    if ds.df.isnull().all().all():
        issues.append("数据集所有值均为空")

    null_cols = ds.df.columns[ds.df.isnull().all()].tolist()
    if null_cols:
        issues.append(f"存在全空列: {null_cols}")

    if required_cols:
        missing = [c for c in required_cols if c not in ds.columns]
        if missing:
            issues.append(f"缺少必需列: {missing}")

    if ds.batch_col:
        if ds.batch_col not in ds.columns:
            issues.append(f"指定的批次列 '{ds.batch_col}' 不存在")
        else:
            batches = ds.get_batches()
            if not batches:
                issues.append(f"批次列 '{ds.batch_col}' 无有效值")

    dup_count = int(ds.df.duplicated().sum())
    if dup_count > 0:
        issues.append(f"存在 {dup_count} 行完全重复的数据")

    return issues
