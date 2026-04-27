from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, redirect, render_template, request, url_for, flash
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass

APP_TITLE = os.getenv("APP_TITLE", "LDPE 熔体仿真参数智能推荐库")
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-before-public-deploy")
DATA_FILE_NAME = os.getenv("LDPE_DATA_FILE", "")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
ZHIPU_API_BASE = os.getenv("ZHIPU_API_BASE", "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4-flash-250414")
ZHIPU_TIMEOUT = float(os.getenv("ZHIPU_TIMEOUT", "45"))
ENABLE_WEB_FALLBACK = os.getenv("ENABLE_WEB_FALLBACK", "true").lower() in {"1", "true", "yes", "on"}
WEB_SEARCH_COUNT = int(os.getenv("WEB_SEARCH_COUNT", "5"))

LOCAL_MIN_SAMPLES = int(os.getenv("LOCAL_MIN_SAMPLES", "3"))
LOCAL_MATCH_SCORE_THRESHOLD = float(os.getenv("LOCAL_MATCH_SCORE_THRESHOLD", "5.5"))

# 严格工况覆盖判定：只要用户输入的温度/压力/剪切率超出本地库有效范围，
# 即使文本和质量评分导致匹配分较高，也判定为“本地不足”，从而触发联网兜底。
STRICT_RANGE_COVERAGE = os.getenv("STRICT_RANGE_COVERAGE", "true").lower() in {"1", "true", "yes", "on"}
LOCAL_MIN_JOINT_COVERAGE = int(os.getenv("LOCAL_MIN_JOINT_COVERAGE", "1"))
RANGE_T_MARGIN_C = float(os.getenv("RANGE_T_MARGIN_C", "10"))
RANGE_P_MARGIN_MPA = float(os.getenv("RANGE_P_MARGIN_MPA", "2"))
RANGE_GAMMA_MARGIN_RATIO = float(os.getenv("RANGE_GAMMA_MARGIN_RATIO", "0.25"))

MAX_LOCAL_CONTEXT_ROWS = int(os.getenv("MAX_LOCAL_CONTEXT_ROWS", "6"))
ADMIN_UPLOAD_TOKEN = os.getenv("ADMIN_UPLOAD_TOKEN", "").strip()

app = Flask(__name__)
app.secret_key = SECRET_KEY

NUMERIC_COLUMNS = [
    "T_min", "T_max", "P_min", "P_max", "gamma_min", "gamma_max",
    "rho", "cp", "k", "mu_ref", "A", "E_mu", "K", "m", "n", "quality_score",
    "channel_length", "channel_diameter", "melt_index", "MFI", "Mw", "Mn",
]

TARGETS = {
    "mu_ref": {"label": "参考黏度 mu_ref", "unit": "Pa·s", "kind": "rheology"},
    "rho": {"label": "密度 rho", "unit": "kg/m³", "kind": "thermal"},
    "cp": {"label": "比热 cp", "unit": "J/(kg·K)", "kind": "thermal"},
    "k": {"label": "导热系数 k", "unit": "W/(m·K)", "kind": "thermal"},
    "A": {"label": "Arrhenius/指数模型系数 A", "unit": "随模型定义", "kind": "viscosity_model"},
    "E_mu": {"label": "黏流活化能 E_mu", "unit": "J/mol 或 kJ/mol，按来源记录", "kind": "viscosity_model"},
    "K": {"label": "幂律/Carreau 稠度系数 K", "unit": "随模型定义", "kind": "viscosity_model"},
    "m": {"label": "幂律/模型指数 m", "unit": "无量纲或按来源记录", "kind": "viscosity_model"},
}

CONFIDENCE_MAP = {"A": 1.2, "B": 0.9, "C": 0.65}

CONTEXT_FIELDS = [
    "param_id", "source_id", "material_name", "grade_or_mfi", "ldpe_grade_detail",
    "T_min", "T_max", "P_min", "P_max", "gamma_min", "gamma_max",
    "rho", "cp", "k", "mu_ref", "A", "E_mu", "K", "m", "n", "vis_model",
    "application_scene", "experimental_method", "equipment_or_setup", "channel_geometry",
    "channel_length", "channel_diameter", "main_result_summary", "data_extraction_method",
    "limitation_note", "confidence_level", "quality_score", "remark", "title", "year", "doi", "source_type",
]

SEARCHABLE_TEXT_FIELDS = [
    "material_name", "grade_or_mfi", "ldpe_grade_detail", "application_scene",
    "experimental_method", "equipment_or_setup", "channel_geometry",
    "main_result_summary", "data_extraction_method", "limitation_note", "remark",
    "title", "source_type", "vis_model",
]

FIELD_LABELS = {
    "param_id": "参数编号",
    "source_id": "来源编号",
    "material_name": "材料名称",
    "grade_or_mfi": "牌号/MFI",
    "ldpe_grade_detail": "LDPE 牌号或型号",
    "T_min": "温度下限/℃",
    "T_max": "温度上限/℃",
    "P_min": "压力下限/MPa",
    "P_max": "压力上限/MPa",
    "gamma_min": "剪切率下限/s⁻¹",
    "gamma_max": "剪切率上限/s⁻¹",
    "rho": "密度/kg·m⁻³",
    "cp": "比热/J·kg⁻¹·K⁻¹",
    "k": "导热系数/W·m⁻¹·K⁻¹",
    "mu_ref": "参考黏度/Pa·s",
    "A": "黏度模型系数 A",
    "E_mu": "黏流活化能 E_mu",
    "K": "稠度系数 K",
    "m": "模型指数 m",
    "n": "模型指数 n",
    "vis_model": "黏度模型",
    "application_scene": "应用场景",
    "experimental_method": "试验方法",
    "equipment_or_setup": "设备或装置",
    "channel_geometry": "通道形状",
    "channel_length": "通道长度",
    "channel_diameter": "通道直径/间隙",
    "main_result_summary": "文献主要结果简述",
    "data_extraction_method": "参数提取方式",
    "limitation_note": "数据局限性",
    "confidence_level": "置信等级",
    "quality_score": "质量评分",
    "remark": "备注",
    "title": "来源题名",
    "year": "年份",
    "doi": "DOI",
    "source_type": "来源类型",
}


@dataclass
class LocalResult:
    value: Optional[float]
    method: str
    sample_count: int
    max_score: float
    average_score: float
    matched_rows: pd.DataFrame
    note: str
    sufficient: bool
    coverage: Dict[str, Any] = field(default_factory=dict)
    insufficiency_reasons: List[str] = field(default_factory=list)


def find_data_file() -> Path:
    if DATA_FILE_NAME:
        path = BASE_DIR / DATA_FILE_NAME
        if path.exists():
            return path
    candidates = [
        "LDPE_dedup_confidence_review.xlsx",
        "LDPE_参数数据库_填充10条(1).xlsx",
        "LDPE_参数数据库_填充10条（1）.xlsx",
        "LDPE_参数数据库_填充10条.xlsx",
    ]
    for name in candidates:
        path = BASE_DIR / name
        if path.exists():
            return path
    for path in BASE_DIR.glob("*.xlsx"):
        if not path.name.startswith("~$"):
            return path
    raise FileNotFoundError(f"未在 {BASE_DIR} 找到可用 Excel 数据库文件。")


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_sheet_robust(path: Path, sheet_name: str, expected_id: str) -> pd.DataFrame:
    """兼容两类 Excel：第一行说明+第二行字段名，或第一行即字段名。"""
    errors: List[str] = []
    for header in (1, 0):
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, header=header)
            df = _clean_columns(df).dropna(how="all")
            if expected_id in df.columns:
                return df
            errors.append(f"header={header}: 未找到 {expected_id}")
        except Exception as exc:  # pragma: no cover - deployment diagnostics
            errors.append(f"header={header}: {exc}")
    raise ValueError(f"工作表 {sheet_name} 读取失败：" + "; ".join(errors))


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = find_data_file()
    source_df = _read_sheet_robust(path, "source_table", "source_id")
    param_df = _read_sheet_robust(path, "ldpe_param_table", "param_id")

    if "param_id" in param_df.columns:
        param_df = param_df[param_df["param_id"].astype(str).str.startswith("PARAM_", na=False)]
    if "source_id" in source_df.columns:
        source_df = source_df[source_df["source_id"].astype(str).str.startswith("SRC_", na=False)]

    for col in NUMERIC_COLUMNS:
        if col in param_df.columns:
            param_df[col] = pd.to_numeric(param_df[col], errors="coerce")
        if col in source_df.columns:
            source_df[col] = pd.to_numeric(source_df[col], errors="coerce")

    return source_df, param_df


def merged_table() -> pd.DataFrame:
    source_df, param_df = load_data()
    join_cols = [c for c in source_df.columns if c not in param_df.columns or c == "source_id"]
    merged = param_df.merge(source_df[join_cols], on="source_id", how="left") if "source_id" in param_df.columns else param_df
    for col in CONTEXT_FIELDS + list(TARGETS.keys()):
        if col not in merged.columns:
            merged[col] = np.nan
    return merged


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "-"}


def safe_float(value: Any) -> Optional[float]:
    try:
        if is_empty(value):
            return None
        return float(value)
    except Exception:
        return None


def format_number(value: Any, digits: int = 5) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    if abs(number) >= 10000 or (0 < abs(number) < 0.001):
        return f"{number:.3e}"
    return f"{number:.{digits}g}"


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[a-zA-Z0-9_\-\.]+|[\u4e00-\u9fff]{2,}", text)
    stop = {"ldpe", "melt", "polymer", "parameter", "参数", "熔体", "工况", "仿真", "fluent"}
    return [t for t in tokens if t not in stop and len(t) >= 2]


def row_text(row: pd.Series) -> str:
    parts = []
    for col in SEARCHABLE_TEXT_FIELDS:
        if col in row.index and not is_empty(row.get(col)):
            parts.append(str(row.get(col)))
    return " ".join(parts).lower()


def text_match_score(row: pd.Series, query_text: str) -> float:
    tokens = tokenize(query_text)
    if not tokens:
        return 0.0
    text = row_text(row)
    hits = sum(1 for t in tokens if t.lower() in text)
    return min(2.0, hits * 0.35)



def _value_in_row_range(row: pd.Series, value: Optional[float], min_col: str, max_col: str) -> bool:
    """判断单条样本的某一工况范围是否覆盖输入值。"""
    if value is None or pd.isna(row.get(min_col)) or pd.isna(row.get(max_col)):
        return True
    return float(row[min_col]) <= value <= float(row[max_col])


def _single_dimension_coverage(
    work: pd.DataFrame,
    label: str,
    value: Optional[float],
    min_col: str,
    max_col: str,
    unit: str,
    abs_margin: Optional[float] = None,
    ratio_margin: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """给出温度/压力/剪切率的覆盖状态，用于决定是否触发网络兜底。"""
    if value is None or min_col not in work.columns or max_col not in work.columns:
        return None
    ranges = work[[min_col, max_col]].dropna().copy()
    if ranges.empty:
        return {
            "field": label,
            "value": value,
            "unit": unit,
            "status": "no_range_data",
            "covered_rows": 0,
            "range_rows": 0,
            "global_min": None,
            "global_max": None,
            "message": f"{label}缺少本地范围数据，无法确认覆盖。",
        }

    ranges[min_col] = pd.to_numeric(ranges[min_col], errors="coerce")
    ranges[max_col] = pd.to_numeric(ranges[max_col], errors="coerce")
    ranges = ranges.dropna()
    if ranges.empty:
        return {
            "field": label,
            "value": value,
            "unit": unit,
            "status": "no_range_data",
            "covered_rows": 0,
            "range_rows": 0,
            "global_min": None,
            "global_max": None,
            "message": f"{label}缺少可解析的本地范围数据，无法确认覆盖。",
        }

    covered_mask = (ranges[min_col] <= value) & (value <= ranges[max_col])
    covered_rows = int(covered_mask.sum())
    global_min = float(ranges[min_col].min())
    global_max = float(ranges[max_col].max())

    if covered_rows > 0:
        status = "covered"
        message = f"{label}={format_number(value)} {unit} 被 {covered_rows} 条本地样本范围覆盖。"
    elif global_min <= value <= global_max:
        status = "gap_in_domain"
        message = (
            f"{label}={format_number(value)} {unit} 位于数据库全局范围 "
            f"{format_number(global_min)}–{format_number(global_max)} {unit} 内，但没有单条样本范围直接覆盖。"
        )
    else:
        if value < global_min:
            gap = global_min - value
            direction = "低于"
        else:
            gap = value - global_max
            direction = "高于"
        if abs_margin is not None:
            margin = abs_margin
        else:
            reference = max(abs(global_min), abs(global_max), abs(value), 1.0)
            margin = reference * (ratio_margin if ratio_margin is not None else 0.0)
        status = "near_extrapolation" if gap <= margin else "out_of_domain"
        message = (
            f"{label}={format_number(value)} {unit} {direction}数据库全局范围 "
            f"{format_number(global_min)}–{format_number(global_max)} {unit}，超出 {format_number(gap)} {unit}。"
        )

    return {
        "field": label,
        "value": value,
        "unit": unit,
        "status": status,
        "covered_rows": covered_rows,
        "range_rows": int(len(ranges)),
        "global_min": global_min,
        "global_max": global_max,
        "message": message,
    }


def coverage_diagnostics(
    work: pd.DataFrame,
    temperature: Optional[float],
    pressure: Optional[float],
    shear_rate: Optional[float],
) -> Dict[str, Any]:
    """基于本地库有效范围判断输入工况是否属于库内覆盖，而不是仅看相似度得分。"""
    reports = []
    for report in [
        _single_dimension_coverage(work, "温度", temperature, "T_min", "T_max", "℃", abs_margin=RANGE_T_MARGIN_C),
        _single_dimension_coverage(work, "压力", pressure, "P_min", "P_max", "MPa", abs_margin=RANGE_P_MARGIN_MPA),
        _single_dimension_coverage(work, "剪切率", shear_rate, "gamma_min", "gamma_max", "s⁻¹", ratio_margin=RANGE_GAMMA_MARGIN_RATIO),
    ]:
        if report is not None:
            reports.append(report)

    provided = len(reports)
    uncovered = [r for r in reports if r["status"] != "covered"]
    out_of_domain = [r for r in reports if r["status"] in {"near_extrapolation", "out_of_domain"}]

    joint_coverage_count = 0
    if provided and not work.empty:
        joint_mask = pd.Series(True, index=work.index)
        if temperature is not None and {"T_min", "T_max"}.issubset(work.columns):
            joint_mask &= work.apply(lambda r: _value_in_row_range(r, temperature, "T_min", "T_max"), axis=1)
        if pressure is not None and {"P_min", "P_max"}.issubset(work.columns):
            joint_mask &= work.apply(lambda r: _value_in_row_range(r, pressure, "P_min", "P_max"), axis=1)
        if shear_rate is not None and {"gamma_min", "gamma_max"}.issubset(work.columns):
            joint_mask &= work.apply(lambda r: _value_in_row_range(r, shear_rate, "gamma_min", "gamma_max"), axis=1)
        joint_coverage_count = int(joint_mask.sum())

    coverage_ok = True
    if STRICT_RANGE_COVERAGE and provided:
        coverage_ok = (len(uncovered) == 0 and joint_coverage_count >= LOCAL_MIN_JOINT_COVERAGE)

    if not provided:
        overall = "not_checked"
        summary = "未输入温度、压力或剪切率，未执行严格工况覆盖判定。"
    elif coverage_ok:
        overall = "covered"
        summary = f"输入工况被本地库范围覆盖，联合覆盖样本数 {joint_coverage_count}。"
    else:
        overall = "uncovered"
        parts = [r["message"] for r in uncovered]
        if joint_coverage_count < LOCAL_MIN_JOINT_COVERAGE:
            parts.append(f"同时覆盖当前温度、压力、剪切率的联合样本数为 {joint_coverage_count}，低于阈值 {LOCAL_MIN_JOINT_COVERAGE}。")
        summary = "；".join(parts)

    return {
        "strict_enabled": STRICT_RANGE_COVERAGE,
        "coverage_ok": coverage_ok,
        "overall": overall,
        "provided_fields": provided,
        "joint_coverage_count": joint_coverage_count,
        "min_joint_coverage": LOCAL_MIN_JOINT_COVERAGE,
        "uncovered_fields": [r["field"] for r in uncovered],
        "out_of_domain_fields": [r["field"] for r in out_of_domain],
        "dimension_reports": reports,
        "summary": summary,
    }


def range_score(row: pd.Series, temperature: Optional[float], pressure: Optional[float], shear_rate: Optional[float], query_text: str = "") -> float:
    score = 0.0
    if temperature is not None and pd.notna(row.get("T_min")) and pd.notna(row.get("T_max")):
        if row["T_min"] <= temperature <= row["T_max"]:
            score += 3.0
        else:
            distance = min(abs(temperature - row["T_min"]), abs(temperature - row["T_max"]))
            score += max(0.0, 1.2 - distance / 45.0)
    if pressure is not None and pd.notna(row.get("P_min")) and pd.notna(row.get("P_max")):
        if row["P_min"] <= pressure <= row["P_max"]:
            score += 2.0
        else:
            distance = min(abs(pressure - row["P_min"]), abs(pressure - row["P_max"]))
            score += max(0.0, 1.0 - distance / 8.0)
    if shear_rate is not None and pd.notna(row.get("gamma_min")) and pd.notna(row.get("gamma_max")):
        if row["gamma_min"] <= shear_rate <= row["gamma_max"]:
            score += 3.0
        else:
            distance = min(abs(shear_rate - row["gamma_min"]), abs(shear_rate - row["gamma_max"]))
            score += max(0.0, 1.2 - distance / 400.0)

    quality = safe_float(row.get("quality_score")) or 0.0
    score += quality * 0.35
    score += CONFIDENCE_MAP.get(str(row.get("confidence_level", "")).strip(), 0.45)
    score += text_match_score(row, query_text)
    return float(score)


def local_weighted_estimate(df: pd.DataFrame, target: str, query: Dict[str, Any]) -> LocalResult:
    work = df.copy()
    if target not in work.columns:
        return LocalResult(None, "无数据", 0, 0, 0, pd.DataFrame(), "数据库中没有该目标参数字段。", False)
    work = work[work[target].notna()].copy()

    vis_model = str(query.get("vis_model") or "").strip()
    grade = str(query.get("grade") or "").strip()

    if vis_model:
        work = work[work["vis_model"].astype(str).str.contains(re.escape(vis_model), case=False, na=False)]
    if grade:
        grade_mask = (
            work["grade_or_mfi"].astype(str).str.contains(re.escape(grade), case=False, na=False) |
            work["ldpe_grade_detail"].astype(str).str.contains(re.escape(grade), case=False, na=False) |
            work["material_name"].astype(str).str.contains(re.escape(grade), case=False, na=False)
        )
        if grade_mask.any():
            work = work[grade_mask]

    if work.empty:
        return LocalResult(None, "本地库无可用记录", 0, 0, 0, work, "当前限定条件下，本地库没有可用于该目标参数的记录。", False)

    query_text = " ".join(str(query.get(k) or "") for k in [
        "application_scene", "experimental_method", "equipment_or_setup", "channel_geometry", "question", "grade", "vis_model"
    ])
    temperature = safe_float(query.get("temperature"))
    pressure = safe_float(query.get("pressure"))
    shear_rate = safe_float(query.get("shear_rate"))

    coverage = coverage_diagnostics(work, temperature, pressure, shear_rate)

    work["match_score"] = work.apply(lambda r: range_score(r, temperature, pressure, shear_rate, query_text), axis=1)
    work = work.sort_values(["match_score", "quality_score"], ascending=[False, False])
    top = work.head(min(MAX_LOCAL_CONTEXT_ROWS, len(work))).copy()
    sample_count = len(top)
    max_score = float(top["match_score"].max()) if sample_count else 0.0
    average_score = float(top["match_score"].mean()) if sample_count else 0.0

    numeric_values = pd.to_numeric(top[target], errors="coerce").dropna()
    if numeric_values.empty:
        value = None
    else:
        valid_rows = top[pd.to_numeric(top[target], errors="coerce").notna()].copy()
        weights = np.maximum(valid_rows["match_score"].fillna(0).values, 1e-6)
        value = float(np.average(pd.to_numeric(valid_rows[target], errors="coerce").values, weights=weights))

    score_ok = sample_count >= LOCAL_MIN_SAMPLES and max_score >= LOCAL_MATCH_SCORE_THRESHOLD
    coverage_ok = (not STRICT_RANGE_COVERAGE) or coverage.get("coverage_ok", True)
    sufficient = score_ok and coverage_ok

    insufficiency_reasons: List[str] = []
    if sample_count < LOCAL_MIN_SAMPLES:
        insufficiency_reasons.append(f"相似样本数 {sample_count} 低于阈值 {LOCAL_MIN_SAMPLES}。")
    if max_score < LOCAL_MATCH_SCORE_THRESHOLD:
        insufficiency_reasons.append(f"最高匹配分 {max_score:.2f} 低于阈值 {LOCAL_MATCH_SCORE_THRESHOLD:g}。")
    if STRICT_RANGE_COVERAGE and not coverage_ok:
        insufficiency_reasons.append("输入工况未被本地库温度/压力/剪切率范围严格覆盖。")
        if coverage.get("summary"):
            insufficiency_reasons.append(str(coverage["summary"]))

    note = (
        f"按温度/压力/剪切率覆盖、文本场景匹配、质量评分和置信等级加权；"
        f"阈值：样本数≥{LOCAL_MIN_SAMPLES}、最高匹配分≥{LOCAL_MATCH_SCORE_THRESHOLD:g}，"
        f"且严格覆盖开启时需至少 {LOCAL_MIN_JOINT_COVERAGE} 条样本同时覆盖输入温度/压力/剪切率。"
    )
    return LocalResult(value, "规则加权推荐", sample_count, max_score, average_score, top, note, sufficient, coverage, insufficiency_reasons)


def knn_estimate(df: pd.DataFrame, target: str, temperature: Optional[float], pressure: Optional[float], shear_rate: Optional[float]) -> Optional[float]:
    feature_cols = ["T_min", "T_max", "P_min", "P_max", "gamma_min", "gamma_max", "quality_score"]
    if target not in df.columns or any(c not in df.columns for c in feature_cols):
        return None
    work = df[df[target].notna()].copy()
    if len(work) < 5:
        return None
    X = work[feature_cols]
    y = pd.to_numeric(work[target], errors="coerce")
    valid = y.notna()
    if valid.sum() < 5:
        return None
    X = X[valid]
    y = y[valid]
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=min(3, len(X)), weights="distance")),
    ])
    model.fit(X, y)
    temp = temperature if temperature is not None else float(pd.to_numeric(work["T_min"], errors="coerce").median())
    press = pressure if pressure is not None else float(pd.to_numeric(work["P_min"], errors="coerce").median())
    gamma = shear_rate if shear_rate is not None else float(pd.to_numeric(work["gamma_min"], errors="coerce").median())
    query = pd.DataFrame([{
        "T_min": temp, "T_max": temp, "P_min": press, "P_max": press,
        "gamma_min": gamma, "gamma_max": gamma,
        "quality_score": float(pd.to_numeric(work["quality_score"], errors="coerce").median()) if "quality_score" in work else 4.0,
    }])
    return float(model.predict(query)[0])


def compact_record(row: Dict[str, Any], keep_target: Optional[str] = None) -> Dict[str, Any]:
    keys = list(CONTEXT_FIELDS)
    if keep_target and keep_target not in keys:
        keys.append(keep_target)
    clean: Dict[str, Any] = {}
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if is_empty(value):
            continue
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        clean[FIELD_LABELS.get(key, key)] = value
    if "match_score" in row and not is_empty(row.get("match_score")):
        clean["匹配分"] = round(float(row["match_score"]), 3)
    return clean


def local_context_payload(local: LocalResult, target: str) -> List[Dict[str, Any]]:
    if local.matched_rows is None or local.matched_rows.empty:
        return []
    return [compact_record(row, target) for row in local.matched_rows.to_dict(orient="records")]


def call_zhipu(messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, max_tokens: int = 1600) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    if not ZHIPU_API_KEY:
        return "", [], "未配置 ZHIPU_API_KEY，已跳过大模型解释。"
    url = f"{ZHIPU_API_BASE}/chat/completions"
    payload: Dict[str, Any] = {
        "model": ZHIPU_MODEL,
        "messages": messages,
        "temperature": 0.18,
        "top_p": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=ZHIPU_TIMEOUT,
        )
        if resp.status_code >= 400:
            return "", [], f"智谱 API 调用失败：HTTP {resp.status_code}，{resp.text[:500]}"
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        web_results = data.get("web_search") or []
        return str(content or "").strip(), web_results, None
    except Exception as exc:  # pragma: no cover - network dependent
        return "", [], f"智谱 API 调用异常：{exc}"


def build_local_prompt(query: Dict[str, Any], target: str, local: LocalResult, context_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    target_meta = TARGETS[target]
    system = (
        "你是严谨的 LDPE 聚合物熔体 Fluent 仿真参数推荐解释助手。"
        "必须只依据用户给出的本地数据库匹配结果和有效字段进行解释；缺失字段静默不提；"
        "禁止编造参数、来源、模型系数、DOI 或实验条件；禁止把网络资料混入本地库结论。"
        "输出应简明、可复现，适合学术研究。"
    )
    user = {
        "任务": "生成本地数据库推荐解释",
        "目标参数": f"{target_meta['label']} ({target_meta['unit']})",
        "用户工况": query,
        "本地库判定": {
            "sufficient": local.sufficient,
            "sample_count": local.sample_count,
            "max_score": round(local.max_score, 3),
            "average_score": round(local.average_score, 3),
            "threshold": {
                "sample_count_min": LOCAL_MIN_SAMPLES,
                "match_score_min": LOCAL_MATCH_SCORE_THRESHOLD,
                "strict_range_coverage": STRICT_RANGE_COVERAGE,
                "joint_coverage_min": LOCAL_MIN_JOINT_COVERAGE,
            },
            "coverage": local.coverage,
            "insufficiency_reasons": local.insufficiency_reasons,
        },
        "推荐值": None if local.value is None else {"value": local.value, "unit": target_meta["unit"], "method": local.method},
        "相似样本有效字段": context_rows,
        "输出格式要求": [
            "用中文输出，不超过 500 字。",
            "分为：推荐结论、依据、适用边界、Fluent 调用提示。",
            "如果样本不足，要明确说明本地库只能给出候选参考，不得定性为真值。",
            "只提及样本中实际存在的字段；不要出现‘未提供’‘缺失’等冗余描述。",
        ],
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def build_web_prompt(query: Dict[str, Any], target: str, local: LocalResult, context_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    target_meta = TARGETS[target]
    system = (
        "你是 LDPE 熔体 Fluent 仿真参数的网络补充参考助手。"
        "仅当本地库相近样本不足时，基于联网检索结果提供临时参考；"
        "必须区分本地库与网络资料，网络资料不得自动入库，不得宣称为最终真值；"
        "必须标注可追溯来源角标或来源标题；无法核实时要明确保守表述。"
    )
    search_prompt = (
        "你是一名高分子流变与 Fluent 非等温仿真参数审校助手。请从网络搜索结果 {search_result} 中，"
        "只提取与 LDPE 熔体热物性、流变参数、温度依赖黏度模型或管道/挤出流动实验相关的信息；"
        "按来源给出摘要，标注 ref 编号、标题、年份/发布日期；不要把网络结果写成数据库已收录数据。"
    )
    tools = [{
        "type": "web_search",
        "web_search": {
            "enable": True,
            "search_engine": "search_pro",
            "search_result": True,
            "search_prompt": search_prompt,
            "count": WEB_SEARCH_COUNT,
            "search_recency_filter": "noLimit",
            "content_size": "high",
        },
    }]
    search_terms = [
        "LDPE melt viscosity temperature shear rate Fluent",
        "low density polyethylene melt thermal conductivity specific heat density",
    ]
    if query.get("application_scene"):
        search_terms.append(str(query["application_scene"]))
    if query.get("experimental_method"):
        search_terms.append(str(query["experimental_method"]))
    if query.get("vis_model"):
        search_terms.append(str(query["vis_model"]))
    user = {
        "任务": "本地库不足时的联网补充解释",
        "检索关键词": " ; ".join(search_terms),
        "目标参数": f"{target_meta['label']} ({target_meta['unit']})",
        "用户工况": query,
        "本地库不足原因": {
            "sample_count": local.sample_count,
            "max_score": round(local.max_score, 3),
            "threshold": {
                "sample_count_min": LOCAL_MIN_SAMPLES,
                "match_score_min": LOCAL_MATCH_SCORE_THRESHOLD,
                "strict_range_coverage": STRICT_RANGE_COVERAGE,
                "joint_coverage_min": LOCAL_MIN_JOINT_COVERAGE,
            },
            "coverage": local.coverage,
            "insufficiency_reasons": local.insufficiency_reasons,
        },
        "本地相似样本摘要": context_rows,
        "输出格式要求": [
            "用中文输出，分为：本地库判定、网络补充参考、使用限制。",
            "网络补充只作为临时参考，禁止建议自动写入数据库。",
            "不要编造具体数值；只有搜索结果明确给出的数值才可复述，并必须带来源 ref。",
            "不要输出与 LDPE 熔体参数无关的泛泛内容。",
        ],
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}], tools


def build_query_from_request(req: request) -> Dict[str, Any]:
    return {
        "target": req.form.get("target", "mu_ref").strip() or "mu_ref",
        "temperature": req.form.get("temperature", "").strip(),
        "pressure": req.form.get("pressure", "").strip(),
        "shear_rate": req.form.get("shear_rate", "").strip(),
        "grade": req.form.get("grade", "").strip(),
        "vis_model": req.form.get("vis_model", "").strip(),
        "application_scene": req.form.get("application_scene", "").strip(),
        "experimental_method": req.form.get("experimental_method", "").strip(),
        "equipment_or_setup": req.form.get("equipment_or_setup", "").strip(),
        "channel_geometry": req.form.get("channel_geometry", "").strip(),
        "channel_length": req.form.get("channel_length", "").strip(),
        "channel_diameter": req.form.get("channel_diameter", "").strip(),
        "question": req.form.get("question", "").strip(),
    }


def recommend(query: Dict[str, Any]) -> Dict[str, Any]:
    df = merged_table()
    target = str(query.get("target") or "mu_ref")
    if target not in TARGETS:
        target = "mu_ref"
        query["target"] = target
    temperature = safe_float(query.get("temperature"))
    pressure = safe_float(query.get("pressure"))
    shear_rate = safe_float(query.get("shear_rate"))

    local = local_weighted_estimate(df, target, query)
    knn_value = knn_estimate(df, target, temperature, pressure, shear_rate)
    context_rows = local_context_payload(local, target)
    local_messages = build_local_prompt(query, target, local, context_rows)
    local_explanation, _, local_error = call_zhipu(local_messages, tools=None, max_tokens=1200)

    if not local_explanation:
        # 无 API Key 或 API 异常时，仍给出可用的本地规则解释，保证网站不崩溃。
        target_meta = TARGETS[target]
        value_text = "-" if local.value is None else f"{format_number(local.value)} {target_meta['unit']}"
        local_explanation = (
            f"推荐结论：{target_meta['label']} 的本地加权推荐值为 {value_text}。"
            f"依据：选取 {local.sample_count} 条相近样本，最高匹配分 {local.max_score:.2f}，"
            f"采用温度、压力、剪切率、场景文本、质量评分和置信等级综合加权。"
            f"适用边界：该结果仅适用于当前数据库覆盖范围，样本不足、匹配分偏低或工况超出库内范围时只能作为 Fluent 初值或敏感性分析候选值。"
        )
        if local.insufficiency_reasons:
            local_explanation += "不足原因：" + "；".join(local.insufficiency_reasons)

    web_explanation = ""
    web_results: List[Dict[str, Any]] = []
    web_error: Optional[str] = None
    should_web = ENABLE_WEB_FALLBACK and not local.sufficient
    if should_web:
        web_messages, tools = build_web_prompt(query, target, local, context_rows)
        web_explanation, web_results, web_error = call_zhipu(web_messages, tools=tools, max_tokens=1800)
        if not web_explanation and not web_error:
            web_explanation = "联网检索未返回可用补充内容。"

    return {
        "query": query,
        "target": target,
        "target_meta": TARGETS[target],
        "local": {
            "value": local.value,
            "value_text": "-" if local.value is None else format_number(local.value),
            "method": local.method,
            "sample_count": local.sample_count,
            "max_score": local.max_score,
            "average_score": local.average_score,
            "sufficient": local.sufficient,
            "note": local.note,
            "coverage": local.coverage,
            "insufficiency_reasons": local.insufficiency_reasons,
            "explanation": local_explanation,
            "error": local_error,
        },
        "knn": None if knn_value is None else {"value": knn_value, "value_text": format_number(knn_value), "method": "KNN 相似案例推荐", "note": "基于最相似历史工况的轻量 KNN 预测，仅作辅助校核。"},
        "prediction_methods": {
            "rule_weighted": {
                "name": "方法1：规则加权推荐",
                "description": "根据温度、压力、剪切率匹配程度，综合数据质量评分和置信等级计算加权平均值。",
                "value": local.value,
                "value_text": "-" if local.value is None else format_number(local.value),
                "sample_count": local.sample_count,
                "max_score": local.max_score,
                "note": local.note,
            },
            "knn": None if knn_value is None else {
                "name": "方法2：KNN 相似案例推荐",
                "description": "找到最相似的历史工况，基于邻近数据进行预测。",
                "value": knn_value,
                "value_text": format_number(knn_value),
                "note": "基于当前数据库的轻量 KNN 相似案例预测；数据库较小时，只作辅助参考。",
            },
        },
        "matched_rows": context_rows,
        "web": {
            "triggered": should_web,
            "explanation": web_explanation,
            "results": web_results,
            "error": web_error,
            "policy": "网络结果仅为临时补充参考，禁止自动入库；入库前需要人工核验来源、单位、工况范围和版权合规性。",
        },
        "config": {
            "model": ZHIPU_MODEL,
            "local_min_samples": LOCAL_MIN_SAMPLES,
            "local_match_score_threshold": LOCAL_MATCH_SCORE_THRESHOLD,
            "web_fallback_enabled": ENABLE_WEB_FALLBACK,
            "strict_range_coverage": STRICT_RANGE_COVERAGE,
            "local_min_joint_coverage": LOCAL_MIN_JOINT_COVERAGE,
        },
    }


@app.context_processor
def inject_globals() -> Dict[str, Any]:
    return {
        "app_title": APP_TITLE,
        "zhipu_model": ZHIPU_MODEL,
        "api_key_ready": bool(ZHIPU_API_KEY),
        "format_number": format_number,
        "field_labels": FIELD_LABELS,
    }


@app.route("/")
def index():
    df = merged_table()
    vis_models = sorted([str(x) for x in df["vis_model"].dropna().unique().tolist()])
    summary = {
        "total_records": int(len(df)),
        "sources": int(df["source_id"].nunique()) if "source_id" in df.columns else 0,
        "vis_models": vis_models,
        "temp_range": f"{format_number(df['T_min'].min(), 4)} ~ {format_number(df['T_max'].max(), 4)} ℃" if not df.empty else "-",
        "api_status": "已配置" if ZHIPU_API_KEY else "未配置",
    }
    cols = ["param_id", "material_name", "grade_or_mfi", "ldpe_grade_detail", "T_min", "T_max", "mu_ref", "vis_model", "quality_score"]
    latest = df[[c for c in cols if c in df.columns]].head(10).to_dict(orient="records")
    return render_template("index.html", summary=summary, latest=latest)


@app.route("/database")
def database():
    df = merged_table()
    keyword = request.args.get("keyword", "").strip()
    vis_model = request.args.get("vis_model", "").strip()
    confidence = request.args.get("confidence", "").strip()
    scene = request.args.get("scene", "").strip()

    if keyword:
        mask = pd.Series(False, index=df.index)
        for col in SEARCHABLE_TEXT_FIELDS:
            if col in df.columns:
                mask |= df[col].astype(str).str.contains(keyword, case=False, na=False)
        df = df[mask]
    if scene and "application_scene" in df.columns:
        df = df[df["application_scene"].astype(str).str.contains(scene, case=False, na=False)]
    if vis_model:
        df = df[df["vis_model"].astype(str) == vis_model]
    if confidence:
        df = df[df["confidence_level"].astype(str) == confidence]

    vis_models = sorted([str(x) for x in merged_table()["vis_model"].dropna().unique().tolist()])
    records = df.head(300).to_dict(orient="records")
    return render_template("database.html", records=records, vis_models=vis_models, keyword=keyword, scene=scene,
                           vis_model=vis_model, confidence=confidence, total=len(df))


@app.route("/recommend", methods=["GET", "POST"])
def recommend_page():
    df = merged_table()
    vis_models = sorted([str(x) for x in df["vis_model"].dropna().unique().tolist()])
    result = None
    form_data = {
        "target": "mu_ref",
        "temperature": "190",
        "pressure": "1",
        "shear_rate": "100",
        "grade": "",
        "vis_model": "",
        "application_scene": "LDPE 熔体管道输送 Fluent 非等温流动",
        "experimental_method": "",
        "equipment_or_setup": "",
        "channel_geometry": "圆管",
        "channel_length": "",
        "channel_diameter": "",
        "question": "请推荐该工况下可用于 Fluent 的 LDPE 熔体参数，并说明是否需要联网补充。",
    }
    if request.method == "POST":
        form_data = build_query_from_request(request)
        try:
            result = recommend(form_data.copy())
        except Exception as exc:
            flash(f"推荐失败：{exc}", "danger")
    return render_template("recommend.html", targets=TARGETS, vis_models=vis_models, result=result, form_data=form_data)


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    df = merged_table()
    vis_models = sorted([str(x) for x in df["vis_model"].dropna().unique().tolist()])
    result_rule = None
    result_knn = None
    form_data: Dict[str, Any] = {
        "target": "mu_ref",
        "temperature": "190",
        "pressure": "1",
        "shear_rate": "100",
        "grade": "",
        "vis_model": "",
        "application_scene": "LDPE 熔体管道输送 Fluent 非等温流动",
        "experimental_method": "",
        "equipment_or_setup": "",
        "channel_geometry": "圆管",
        "channel_length": "",
        "channel_diameter": "",
        "question": "",
    }

    if request.method == "POST":
        try:
            form_data = build_query_from_request(request)
            target = str(form_data.get("target") or "mu_ref")
            temperature = safe_float(form_data.get("temperature"))
            pressure = safe_float(form_data.get("pressure"))
            shear_rate = safe_float(form_data.get("shear_rate"))
            if target not in TARGETS:
                target = "mu_ref"
                form_data["target"] = target

            result_rule = local_weighted_estimate(df, target, form_data)
            knn_value = knn_estimate(df, target, temperature, pressure, shear_rate)
            valid_count = int(df[target].notna().sum()) if target in df.columns else 0
            result_knn = {
                "method": "KNN 相似案例推荐",
                "value": knn_value,
                "value_text": "-" if knn_value is None else format_number(knn_value),
                "sample_count": valid_count,
                "note": "找到最相似的历史工况，基于邻近数据进行预测；数据库较小时只作辅助校核。" if knn_value is not None else "样本量不足或目标字段缺少有效数值，本次未启用 KNN。",
            }
        except Exception as exc:
            flash(f"预测失败：{exc}", "danger")

    return render_template(
        "predict.html",
        targets=TARGETS,
        vis_models=vis_models,
        result_rule=result_rule,
        result_knn=result_knn,
        form_data=form_data,
    )


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    payload = request.get_json(silent=True) or {}
    query = {
        "target": str(payload.get("target", "mu_ref")),
        "temperature": payload.get("temperature", ""),
        "pressure": payload.get("pressure", ""),
        "shear_rate": payload.get("shear_rate", ""),
        "grade": payload.get("grade", ""),
        "vis_model": payload.get("vis_model", ""),
        "application_scene": payload.get("application_scene", ""),
        "experimental_method": payload.get("experimental_method", ""),
        "equipment_or_setup": payload.get("equipment_or_setup", ""),
        "channel_geometry": payload.get("channel_geometry", ""),
        "channel_length": payload.get("channel_length", ""),
        "channel_diameter": payload.get("channel_diameter", ""),
        "question": payload.get("question", ""),
    }
    try:
        return jsonify(recommend(query))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health")
def health():
    try:
        df = merged_table()
        return jsonify({
            "ok": True,
            "records": len(df),
            "data_file": find_data_file().name,
            "zhipu_model": ZHIPU_MODEL,
            "api_key_ready": bool(ZHIPU_API_KEY),
            "web_fallback_enabled": ENABLE_WEB_FALLBACK,
            "strict_range_coverage": STRICT_RANGE_COVERAGE,
            "local_min_joint_coverage": LOCAL_MIN_JOINT_COVERAGE,
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        token = request.form.get("token", "").strip()
        if not ADMIN_UPLOAD_TOKEN or token != ADMIN_UPLOAD_TOKEN:
            flash("公共部署环境已启用上传保护：请配置并输入 ADMIN_UPLOAD_TOKEN。", "danger")
            return redirect(url_for("upload"))
        file = request.files.get("file")
        if not file or not file.filename:
            flash("请先选择 Excel 文件。", "danger")
            return redirect(url_for("upload"))
        if not file.filename.lower().endswith((".xlsx", ".xls")):
            flash("仅支持上传 Excel 文件。", "danger")
            return redirect(url_for("upload"))
        target = find_data_file()
        file.save(target)
        flash("数据库文件已更新。", "success")
        return redirect(url_for("database"))
    return render_template("upload.html", upload_enabled=bool(ADMIN_UPLOAD_TOKEN))


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
