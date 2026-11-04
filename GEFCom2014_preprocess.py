#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEFCom2014 Load 数据预处理脚本

读取 data/GEFCom2014/Load/Task 1..15 下的 Lx-train.csv，
清洗后按时间顺序拼接为一条完整的小时级负荷序列。

清洗规则：
1. 只读取 Lx-train.csv，忽略 benchmark / solution / submission 文件。
2. 解析 GEFCom 紧凑时间戳（MDDYYYY / MMDYYYY 等变长格式），
   利用小时连续性消解 MDD / MMD 歧义。
3. 时间戳无法解析的行剔除并计数；LOAD 与 w1..w25 转为数值。
4. 单文件内同一时间戳若 LOAD 冲突，剔除该时间戳并记录；否则去重。
5. 跨 Task 合并：时间戳取并集；同一时间戳优先保留 LOAD 非缺失的行，
   仍有多个取值时以 Task 编号更大（更新）的为准，并记录冲突数。
6. 不插补 LOAD，不删除负荷极值。

输出（--output 目录下）：
- gefcom2014_load_merged.csv  按时间升序的合并序列
- preprocess_report.json      清洗与合并报告

运行：
    python GEFCom2014_preprocess.py
    python GEFCom2014_preprocess.py --input "D:/data/GEFCom2014/Load" --output "D:/out"
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path

import pandas as pd

TIME_CANDIDATES = ["TIMESTAMP", "timestamp", "Timestamp", "DATE", "date", "datetime", "DateTime"]
LOAD_CANDIDATES = ["LOAD", "load", "Load", "demand", "Demand", "y"]
ZONE_CANDIDATES = ["ZONEID", "zoneid", "ZoneID", "zone_id", "ZONE"]
WEATHER_PATTERN = re.compile(r"^w\d+$", re.IGNORECASE)
EXCLUDE_KEYWORDS = ("benchmark", "solution", "submission")


def find_column(columns, candidates):
    lookup = {str(c).strip().lstrip("﻿").lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def timestamp_candidates(value):
    """返回紧凑日期字符串的所有合法解释，暂不擅自消解 MDD / MMD 歧义。"""
    if pd.isna(value):
        return []
    s = str(value).strip()
    if re.search(r"[-/]", s):
        parsed = pd.to_datetime(s, errors="coerce")
        return [] if pd.isna(parsed) else [pd.Timestamp(parsed)]
    m = re.fullmatch(r"(\d+)\s+(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if not m:
        parsed = pd.to_datetime(s, errors="coerce")
        return [] if pd.isna(parsed) else [pd.Timestamp(parsed)]
    date_digits, hour, minute, second = m.groups()
    if len(date_digits) < 6:
        return []
    year = int(date_digits[-4:])
    prefix = date_digits[:-4]
    if len(prefix) == 2:
        splits = [(int(prefix[0]), int(prefix[1]))]
    elif len(prefix) == 3:
        # 可能是 M DD，也可能是 MM D；需要结合文件中的时间顺序消歧。
        splits = [(int(prefix[0]), int(prefix[1:])),
                  (int(prefix[:2]), int(prefix[2:]))]
    elif len(prefix) == 4:
        splits = [(int(prefix[:2]), int(prefix[2:]))]
    else:
        return []
    out = []
    for month, day in splits:
        try:
            ts = pd.Timestamp(datetime.datetime(
                year, month, day, int(hour), int(minute), int(second or 0)
            ))
            if ts not in out:
                out.append(ts)
        except ValueError:
            pass
    return out


def parse_gefcom_timestamps(values):
    """
    先尝试首条记录的各个合法日期解释，再沿时间顺序选择最接近上一小时的候选。
    以连续小时数最多、非小时间隔最少为优先，解决首条记录本身存在歧义的情况。
    """
    vals = list(values)
    all_candidates = [timestamp_candidates(v) for v in vals]
    possible_starts = all_candidates[0] if all_candidates else []
    if not possible_starts:
        # 首条无效时，使用逐行贪心解析，保留无效时间为 NaT。
        result, previous = [], None
        for candidates in all_candidates:
            if not candidates:
                result.append(pd.NaT)
                continue
            chosen = candidates[0] if previous is None else min(
                candidates, key=lambda x: abs((x - (previous + pd.Timedelta(hours=1))).total_seconds())
            )
            result.append(chosen)
            previous = chosen
        return pd.Series(result, index=values.index, dtype="datetime64[ns]")

    best_result, best_score = None, None
    for start in possible_starts:
        result, previous = [], None
        for candidates in all_candidates:
            if not candidates:
                result.append(pd.NaT)
                continue
            if previous is None:
                chosen = start if start in candidates else candidates[0]
            else:
                expected = previous + pd.Timedelta(hours=1)
                chosen = min(candidates, key=lambda x: abs((x - expected).total_seconds()))
            result.append(chosen)
            previous = chosen
        valid = pd.Series(result).dropna()
        if len(valid) > 1:
            delta = valid.diff().dropna().dt.total_seconds() / 3600
            non_hour = int((delta != 1).sum())
            abs_deviation = float((delta - 1).abs().sum())
        else:
            non_hour, abs_deviation = 0, 0.0
        score = (non_hour, abs_deviation)
        if best_score is None or score < best_score:
            best_score, best_result = score, result
    return pd.Series(best_result, index=values.index, dtype="datetime64[ns]")


def normalise_columns(df):
    df = df.copy()
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    time_col = find_column(df.columns, TIME_CANDIDATES)
    load_col = find_column(df.columns, LOAD_CANDIDATES)
    zone_col = find_column(df.columns, ZONE_CANDIDATES)
    if time_col is None:
        raise ValueError(f"找不到时间字段；当前字段：{list(df.columns)}")
    if load_col is None:
        raise ValueError(f"找不到负荷字段；当前字段：{list(df.columns)}")
    rename = {time_col: "timestamp", load_col: "load"}
    if zone_col is not None:
        rename[zone_col] = "zoneid"
    for c in df.columns:
        if WEATHER_PATTERN.fullmatch(str(c).strip()):
            rename[c] = str(c).strip().lower()
    return df.rename(columns=rename)


def task_number(path):
    m = re.search(r"L(\d+)-train\.csv$", path.name, re.I)
    return int(m.group(1)) if m else 0


def inspect_gaps(ts):
    vals = pd.DatetimeIndex(ts.dropna().sort_values().unique())
    if len(vals) < 2:
        return {"unique_timestamps": int(len(vals)), "hourly_interval_count": 0,
                "non_hourly_interval_count": 0, "largest_gap_hours": None}
    delta = pd.Series(vals[1:] - vals[:-1])
    hours = delta.dt.total_seconds() / 3600
    return {
        "unique_timestamps": int(len(vals)),
        "hourly_interval_count": int((hours == 1).sum()),
        "non_hourly_interval_count": int((hours != 1).sum()),
        "largest_gap_hours": float(hours.max()),
        "gap_examples_hours": [float(x) for x in hours[hours != 1].value_counts().head(10).index.tolist()],
    }


def clean_file(path):
    """清洗单个 Lx-train.csv，返回 (DataFrame, 统计信息)。"""
    raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df = normalise_columns(raw)
    df["timestamp"] = parse_gefcom_timestamps(df["timestamp"])
    df["load"] = pd.to_numeric(df["load"], errors="coerce")
    weather_cols = sorted(
        [c for c in df.columns if WEATHER_PATTERN.fullmatch(str(c))],
        key=lambda x: int(re.search(r"\d+", x).group()),
    )
    for c in weather_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    invalid_ts_rows = int(df["timestamp"].isna().sum())
    df = df[df["timestamp"].notna()].copy()

    duplicate_rows = int(df["timestamp"].duplicated(keep=False).sum())
    conflict_times_excluded = 0
    if duplicate_rows:
        load_counts = df.groupby("timestamp")["load"].nunique(dropna=True)
        conflict_times = load_counts[load_counts > 1].index
        conflict_times_excluded = int(len(conflict_times))
        if conflict_times_excluded:
            df = df[~df["timestamp"].isin(conflict_times)].copy()
        df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="first")

    keep_cols = ["timestamp", "load"] + weather_cols
    if "zoneid" in df.columns:
        keep_cols.append("zoneid")
    df = df[keep_cols].sort_values("timestamp").reset_index(drop=True)

    stats = {
        "file": path.name,
        "task": f"L{task_number(path)}",
        "rows_in_raw": int(len(raw)),
        "rows_after_clean": int(len(df)),
        "invalid_timestamp_rows_dropped": invalid_ts_rows,
        "duplicate_timestamp_rows_before_dedup": duplicate_rows,
        "conflict_timestamps_excluded": conflict_times_excluded,
        "start_timestamp": str(df["timestamp"].min()) if len(df) else None,
        "end_timestamp": str(df["timestamp"].max()) if len(df) else None,
        "load_valid_rows": int(df["load"].notna().sum()),
        "load_missing_rows": int(df["load"].isna().sum()),
    }
    return df, stats


def merge_tasks(frames):
    """
    跨 Task 合并：时间戳取并集；同一时间戳优先保留 LOAD 非缺失的行，
    其余情况以 Task 编号更大（更新）的为准。返回 (合并结果, LOAD 冲突时间戳数)。
    """
    merged = pd.concat(frames, ignore_index=True, sort=False)
    non_null = merged[merged["load"].notna()]
    load_conflicts = int((non_null.groupby("timestamp")["load"].nunique() > 1).sum())

    merged["_has_load"] = merged["load"].notna().astype(int)
    merged = merged.sort_values(["timestamp", "_has_load", "task_num"])
    merged = merged.drop_duplicates("timestamp", keep="last")
    merged = merged.drop(columns="_has_load").sort_values("timestamp").reset_index(drop=True)
    return merged, load_conflicts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default=r"C:\Users\Administrator\Desktop\EPSR\data\GEFCom2014\Load",
                        help="GEFCom2014 Load 根目录（包含 Task 1..15 文件夹）")
    parser.add_argument("--output", default=r"C:\Users\Administrator\Desktop\EPSR\data\GEFCom2014\Load\clean",
                        help="输出目录（dataset.py 的 load_GEFCom2014 从这里读取）")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    outdir = Path(args.output).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    paths = [p for p in input_dir.rglob("*-train.csv")
             if not any(k in p.name.lower() for k in EXCLUDE_KEYWORDS)]
    paths = sorted(paths, key=task_number)
    if not paths:
        raise FileNotFoundError(f"没有找到匹配的 *-train.csv：{input_dir}")
    if len(paths) != 15:
        print(f"警告：预期 15 个 Task 文件，实际找到 {len(paths)} 个")

    frames, file_reports = [], []
    for path in paths:
        try:
            df, stats = clean_file(path)
        except Exception as exc:
            file_reports.append({"file": path.name, "error": str(exc)})
            continue
        df["task_num"] = task_number(path)
        df["source_task"] = f"L{task_number(path)}"
        frames.append(df)
        file_reports.append(stats)
    if not frames:
        raise RuntimeError("所有文件均处理失败，无法合并")

    merged, load_conflicts = merge_tasks(frames)
    weather_cols = [c for c in merged.columns if WEATHER_PATTERN.fullmatch(str(c))]
    out_cols = (["timestamp"] + (["zoneid"] if "zoneid" in merged.columns else [])
                + ["load"] + weather_cols + ["source_task"])
    outpath = outdir / "gefcom2014_load_merged.csv"
    merged[out_cols].to_csv(outpath, index=False, encoding="utf-8-sig")

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(outdir),
        "task_files_found": len(paths),
        "files": file_reports,
        "merge": {
            "total_rows": int(len(merged)),
            "unique_timestamps": int(merged["timestamp"].nunique()),
            "start_timestamp": str(merged["timestamp"].min()),
            "end_timestamp": str(merged["timestamp"].max()),
            "load_valid_rows": int(merged["load"].notna().sum()),
            "load_missing_rows": int(merged["load"].isna().sum()),
            "load_conflict_timestamps_resolved_by_latest_task": load_conflicts,
            "load_min": float(merged["load"].min()) if merged["load"].notna().any() else None,
            "load_max": float(merged["load"].max()) if merged["load"].notna().any() else None,
            "weather_missing_counts": {c: int(merged[c].isna().sum()) for c in weather_cols},
            "time_intervals": inspect_gaps(merged["timestamp"]),
        },
        "notes": [
            "合并序列取 15 个 Task 时间戳的并集；重复时间戳优先取 LOAD 非缺失的行，"
            "仍有冲突时以 Task 编号更大的为准。",
            "不插补 LOAD；构造监督学习样本时仅使用有效标签和完整历史窗口。",
            "不删除极端负荷；异常值阈值应基于训练集统计确定。",
        ],
        "output_file": outpath.name,
    }
    with open(outdir / "preprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"完成：{len(frames)}/{len(paths)} 个文件清洗成功并合并")
    print(f"合并序列：{len(merged)} 行，{merged['timestamp'].min()} ~ {merged['timestamp'].max()}")
    print(f"输出：{outpath}")
    print(f"报告：{outdir / 'preprocess_report.json'}")


if __name__ == "__main__":
    main()
