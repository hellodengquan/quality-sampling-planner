"""数据导入模块测试."""

import os

import pandas as pd
import pytest
import yaml

from qsp.data_loader import (
    BatchDataset,
    load_batch_data,
    load_csv,
    load_dataframe,
    load_yaml,
    validate_dataset,
)


@pytest.fixture
def tmp_csv(tmp_path):
    p = tmp_path / "data.csv"
    pd.DataFrame({
        "id": [1, 2, 3, 4],
        "batch_id": ["A", "A", "B", "B"],
        "value": [10, 20, 30, 40],
    }).to_csv(p, index=False, encoding="utf-8")
    return str(p)


@pytest.fixture
def tmp_json(tmp_path):
    p = tmp_path / "data.json"
    pd.DataFrame([
        {"id": 1, "batch_id": "X", "value": 100},
        {"id": 2, "batch_id": "Y", "value": 200},
    ]).to_json(p, orient="records", force_ascii=False)
    return str(p)


@pytest.fixture
def tmp_yaml(tmp_path):
    p = tmp_path / "data.yaml"
    data = [
        {"id": 1, "batch_id": "P", "value": 5},
        {"id": 2, "batch_id": "Q", "value": 6},
        {"id": 3, "batch_id": "Q", "value": 7},
    ]
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)
    return str(p)


@pytest.fixture
def empty_csv(tmp_path):
    p = tmp_path / "empty.csv"
    pd.DataFrame(columns=["a", "b"]).to_csv(p, index=False)
    return str(p)


class TestLoadFormats:
    def test_load_csv(self, tmp_csv):
        df = load_csv(tmp_csv)
        assert len(df) == 4
        assert list(df.columns) == ["id", "batch_id", "value"]

    def test_load_dataframe_csv(self, tmp_csv):
        df = load_dataframe(tmp_csv)
        assert len(df) == 4

    def test_load_dataframe_json(self, tmp_json):
        df = load_dataframe(tmp_json)
        assert len(df) == 2

    def test_load_dataframe_yaml(self, tmp_yaml):
        df = load_dataframe(tmp_yaml)
        assert len(df) == 3

    def test_force_format(self, tmp_csv):
        df = load_dataframe(tmp_csv, fmt="csv")
        assert len(df) == 4

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataframe(str(tmp_path / "no_such.csv"))

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_text("dummy")
        with pytest.raises(ValueError, match="不支持"):
            load_dataframe(str(p))


class TestBatchDataset:
    def test_properties(self):
        df = pd.DataFrame({"x": [1, 2], "batch": ["A", "B"]})
        ds = BatchDataset(df=df, source="test.csv", batch_col="batch")
        assert ds.rows == 2
        assert ds.columns == ["x", "batch"]
        assert ds.get_batches() == ["A", "B"]

    def test_filter_by_batch(self):
        df = pd.DataFrame({"id": [1, 2, 3], "batch": ["A", "B", "C"]})
        ds = BatchDataset(df=df, source="t", batch_col="batch")
        f = ds.filter_by_batch(["A", "C"])
        assert len(f) == 2
        assert set(f["batch"].unique()) == {"A", "C"}

    def test_filter_by_batch_none_returns_all(self):
        df = pd.DataFrame({"id": [1, 2], "batch": ["A", "B"]})
        ds = BatchDataset(df=df, source="t", batch_col="batch")
        f = ds.filter_by_batch(None)
        assert len(f) == 2

    def test_no_batch_col(self):
        df = pd.DataFrame({"a": [1]})
        ds = BatchDataset(df=df, source="t")
        assert ds.get_batches() == []


class TestValidateDataset:
    def test_clean_data(self, tmp_csv):
        ds = load_batch_data(tmp_csv, batch_col="batch_id")
        issues = validate_dataset(ds)
        assert all("完全重复" not in i and "全空" not in i for i in issues)

    def test_empty_dataset(self):
        ds = BatchDataset(df=pd.DataFrame(), source="e")
        issues = validate_dataset(ds)
        assert any("空" in i for i in issues)

    def test_missing_batch_col(self):
        df = pd.DataFrame({"a": [1, 2]})
        ds = BatchDataset(df=df, source="t", batch_col="no_such")
        issues = validate_dataset(ds)
        assert any("批次列" in i for i in issues)

    def test_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        ds = BatchDataset(df=df, source="t")
        issues = validate_dataset(ds)
        assert any("重复" in i for i in issues)

    def test_required_cols_missing(self):
        df = pd.DataFrame({"a": [1]})
        ds = BatchDataset(df=df, source="t")
        issues = validate_dataset(ds, required_cols=["b", "c"])
        assert any("缺少必需列" in i for i in issues)


class TestLoadBatchData:
    def test_with_batch(self, tmp_csv):
        ds = load_batch_data(tmp_csv, batch_col="batch_id")
        assert ds.batch_col == "batch_id"
        assert ds.get_batches() == ["A", "B"]

    def test_no_batch(self, tmp_json):
        ds = load_batch_data(tmp_json)
        assert ds.batch_col is None
        assert ds.rows == 2

    def test_yaml_records_wrapper(self, tmp_path):
        p = tmp_path / "wrap.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump({"records": [{"k": 1}, {"k": 2}]}, f)
        df = load_yaml(str(p))
        assert len(df) == 2
        assert "k" in df.columns
