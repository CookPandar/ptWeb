from __future__ import annotations

import os
import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(
    "/home/zhangshuwen/Collab-Overcooked/runs/rl/train/initial_rollout_cache"
)
DEFAULT_FILE = DEFAULT_DATA_DIR / "rollout_rank0_u00001.pt"
DEFAULT_TOKENIZER = Path("/home/zhangshuwen/Collab-Overcooked/runs/Chef")
DEFAULT_TRAIN_DIR = Path("/home/zhangshuwen/Collab-Overcooked/runs/rl/train")
DEFAULT_REWARD_CSV = DEFAULT_TRAIN_DIR / "reward_curve.csv"
DEFAULT_TRAIN_CSV = DEFAULT_TRAIN_DIR / "train_curve.csv"
MAX_LIST_FILES = 200
MAX_ITEMS = 5000
MAX_TENSOR_PREVIEW = 64
TEXT_FIELD_NAMES = {"prompt_ids", "response_ids", "critic_input_ids"}


app = FastAPI(title="PT Rollout Viewer")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


def _normalize_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    return path


def _resolve_existing_path(path_str: str | None, default: Path | None = None) -> Path | None:
    if path_str:
        return _normalize_path(path_str)
    if default and default.exists():
        return default.resolve()
    return None


def _tensor_preview(tensor: torch.Tensor, limit: int = MAX_TENSOR_PREVIEW) -> dict[str, Any]:
    cpu_tensor = tensor.detach().cpu()
    flat = cpu_tensor.reshape(-1)
    preview = flat[:limit].tolist()
    result: dict[str, Any] = {
        "type": "tensor",
        "dtype": str(cpu_tensor.dtype),
        "shape": list(cpu_tensor.shape),
        "numel": int(cpu_tensor.numel()),
        "preview": preview,
        "truncated": int(cpu_tensor.numel()) > limit,
    }
    if cpu_tensor.numel() and cpu_tensor.is_floating_point():
        result["stats"] = {
            "min": float(cpu_tensor.min().item()),
            "max": float(cpu_tensor.max().item()),
            "mean": float(cpu_tensor.float().mean().item()),
        }
    return result


def _primitive_summary(value: Any) -> dict[str, Any]:
    return {"type": type(value).__name__, "value": value}


def _summarize_value(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        summary = _tensor_preview(value)
        summary.pop("preview", None)
        summary["preview_hidden"] = True
        return summary
    if isinstance(value, dict):
        return {
            "type": "dict",
            "size": len(value),
            "keys": list(value.keys())[:50],
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "size": len(value),
            "preview_types": [type(v).__name__ for v in value[:10]],
        }
    if isinstance(value, (int, float, str, bool)) or value is None:
        return _primitive_summary(value)
    return {"type": type(value).__name__, "repr": repr(value)[:240]}


@lru_cache(maxsize=8)
def _load_tokenizer_cached(path_str: str):
    return AutoTokenizer.from_pretrained(path_str, trust_remote_code=True)


def _load_tokenizer(path: Path | None):
    if path is None:
        return None
    return _load_tokenizer_cached(str(path))


def _decode_tokens(tokenizer: Any, value: Any) -> dict[str, Any] | None:
    if tokenizer is None or not isinstance(value, torch.Tensor):
        return None
    if value.dim() != 1:
        return None
    try:
        ids = value.detach().cpu().tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False)
        return {
            "text": text,
            "char_length": len(text),
            "line_count": text.count("\n") + 1 if text else 0,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _item_summary(item: Any, index: int, tokenizer: Any | None = None) -> dict[str, Any]:
    if isinstance(item, dict):
        summary: dict[str, Any] = {"index": index, "type": "dict", "fields": {}}
        for key, value in item.items():
            field_summary = _summarize_value(value)
            if key in TEXT_FIELD_NAMES:
                decoded = _decode_tokens(tokenizer, value)
                if decoded is not None:
                    field_summary["decoded"] = decoded
            summary["fields"][key] = field_summary
        return summary
    if isinstance(item, (list, tuple)):
        return {
            "index": index,
            "type": type(item).__name__,
            "size": len(item),
            "items": [_summarize_value(v) for v in item[:20]],
        }
    return {"index": index, "type": type(item).__name__, "value": _summarize_value(item)}


def _dataset_overview(data: Any) -> dict[str, Any]:
    if isinstance(data, list):
        sample_types: dict[str, int] = {}
        for item in data[:100]:
            key = type(item).__name__
            sample_types[key] = sample_types.get(key, 0) + 1
        return {"root_type": "list", "length": len(data), "sample_types": sample_types}
    if isinstance(data, dict):
        return {"root_type": "dict", "keys": list(data.keys())[:100], "length": len(data)}
    return {"root_type": type(data).__name__}


@lru_cache(maxsize=16)
def _load_file_cached(path_str: str, mtime_ns: int) -> Any:
    del mtime_ns
    return torch.load(path_str, map_location="cpu")


def _load_file(path: Path) -> Any:
    stat = path.stat()
    return _load_file_cached(str(path), stat.st_mtime_ns)


def _build_rows(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        return []

    rows = []
    for idx, item in enumerate(data[:MAX_ITEMS]):
        if isinstance(item, dict):
            row = {
                "index": idx,
                "agent_index": item.get("agent_index"),
                "timestep": item.get("timestep"),
                "reward": item.get("reward"),
                "value": item.get("value"),
                "log_prob": item.get("log_prob"),
                "done": item.get("done"),
                "prompt_len": int(item["prompt_ids"].numel()) if isinstance(item.get("prompt_ids"), torch.Tensor) else None,
                "response_len": int(item["response_ids"].numel()) if isinstance(item.get("response_ids"), torch.Tensor) else None,
                "critic_len": int(item["critic_input_ids"].numel()) if isinstance(item.get("critic_input_ids"), torch.Tensor) else None,
            }
        else:
            row = {"index": idx, "type": type(item).__name__}
        rows.append(row)
    return rows


def _coerce_csv_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for idx, row in enumerate(reader, start=1):
            parsed = {k: _coerce_csv_value(v) for k, v in row.items()}
            parsed["_round"] = idx
            rows.append(parsed)
        return rows


def _csv_numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    names: list[str] = []
    for key in rows[0].keys():
        values = [row.get(key) for row in rows if row.get(key) is not None]
        if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            names.append(key)
    return names


def _series_payload(
    rows: list[dict[str, Any]],
    x_key: str,
    y_keys: list[str],
) -> dict[str, Any]:
    series = []
    for y_key in y_keys:
        points = []
        for row in rows:
            x_val = row.get(x_key)
            y_val = row.get(y_key)
            if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
                points.append({"x": x_val, "y": y_val})
        series.append({"name": y_key, "points": points})
    return {"x_key": x_key, "series": series}


def _csv_summary(path: Path, preferred_metrics: list[str]) -> dict[str, Any]:
    rows = _load_csv_rows(path)
    numeric_columns = _csv_numeric_columns(rows)
    x_key = "_round" if rows else ""
    selected = [name for name in preferred_metrics if name in numeric_columns]
    if not selected:
        selected = [name for name in numeric_columns if name != x_key][:4]
    stat = path.stat()
    return {
        "path": str(path),
        "mtime": stat.st_mtime,
        "size_bytes": stat.st_size,
        "row_count": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "numeric_columns": numeric_columns,
        "default_x_key": x_key,
        "default_metrics": selected,
        "rows": rows,
        "chart": _series_payload(rows, x_key, selected) if x_key and selected else {"x_key": x_key, "series": []},
    }


def _build_chart_groups(rows: list[dict[str, Any]], groups: list[dict[str, Any]], default_x_key: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    numeric_columns = _csv_numeric_columns(rows)
    for group in groups:
        metrics = [name for name in group["metrics"] if name in numeric_columns]
        payloads.append(
            {
                "title": group["title"],
                "metrics": metrics,
                "chart": _series_payload(rows, default_x_key, metrics) if default_x_key and metrics else {"x_key": default_x_key, "series": []},
            }
        )
    return payloads


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_file": str(DEFAULT_FILE),
            "default_dir": str(DEFAULT_DATA_DIR),
            "default_tokenizer": str(DEFAULT_TOKENIZER),
            "default_reward_csv": str(DEFAULT_REWARD_CSV),
            "default_train_csv": str(DEFAULT_TRAIN_CSV),
        },
    )


@app.get("/api/list")
async def list_pt_files(
    dir_path: str = Query(default=str(DEFAULT_DATA_DIR)),
) -> dict[str, Any]:
    directory = _normalize_path(dir_path)
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {directory}")

    files = []
    for path in sorted(directory.glob("*.pt"))[:MAX_LIST_FILES]:
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return {"directory": str(directory), "files": files}


@app.get("/api/file")
async def get_file_summary(path: str) -> dict[str, Any]:
    file_path = _normalize_path(path)
    if file_path.suffix != ".pt":
        raise HTTPException(status_code=400, detail="Only .pt files are supported.")
    data = _load_file(file_path)
    return {
        "path": str(file_path),
        "overview": _dataset_overview(data),
        "rows": _build_rows(data),
    }


@app.get("/api/item")
async def get_item_detail(
    path: str,
    index: int = Query(ge=0),
    tokenizer_path: str | None = None,
) -> dict[str, Any]:
    file_path = _normalize_path(path)
    data = _load_file(file_path)
    tokenizer_root = _resolve_existing_path(tokenizer_path, DEFAULT_TOKENIZER)
    tokenizer = _load_tokenizer(tokenizer_root)
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="Root object is not a list.")
    if index >= len(data):
        raise HTTPException(status_code=404, detail=f"Index out of range: {index}")
    return {
        "path": str(file_path),
        "tokenizer_path": str(tokenizer_root) if tokenizer_root else None,
        "detail": _item_summary(data[index], index, tokenizer=tokenizer),
    }


@app.get("/api/csv")
async def get_csv_summary(
    path: str,
    metrics: str | None = None,
    x_key: str | None = None,
) -> dict[str, Any]:
    csv_path = _normalize_path(path)
    if csv_path.suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")
    rows = _load_csv_rows(csv_path)
    numeric_columns = _csv_numeric_columns(rows)
    actual_x_key = x_key or ("_round" if rows else "")
    requested_metrics = [part.strip() for part in (metrics or "").split(",") if part.strip()]
    selected = [name for name in requested_metrics if name in numeric_columns and name != actual_x_key]
    if not selected:
        selected = [name for name in numeric_columns if name != actual_x_key][:4]
    stat = csv_path.stat()
    return {
        "path": str(csv_path),
        "mtime": stat.st_mtime,
        "size_bytes": stat.st_size,
        "row_count": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "numeric_columns": numeric_columns,
        "default_x_key": actual_x_key,
        "selected_metrics": selected,
        "chart": _series_payload(rows, actual_x_key, selected) if actual_x_key and selected else {"x_key": actual_x_key, "series": []},
        "tail_rows": rows[-10:],
    }


@app.get("/api/monitor")
async def get_monitor_summary(
    reward_path: str = Query(default=str(DEFAULT_REWARD_CSV)),
    train_path: str = Query(default=str(DEFAULT_TRAIN_CSV)),
) -> dict[str, Any]:
    reward_csv = _normalize_path(reward_path)
    train_csv = _normalize_path(train_path)
    reward_summary = _csv_summary(
        reward_csv,
        preferred_metrics=[
            "agent0_rl_sum",
            "agent1_rl_sum",
        ],
    )
    train_summary = _csv_summary(
        train_csv,
        preferred_metrics=[
            "value_mean",
            "return_mean",
            "explained_var",
        ],
    )
    reward_groups = _build_chart_groups(
        reward_summary["rows"],
        [
            {"title": "Agent Cumulative Reward", "metrics": ["agent0_rl_sum", "agent1_rl_sum"]},
            {"title": "Agent Reward Mean", "metrics": ["agent0_rl_mean", "agent1_rl_mean"]},
            {"title": "Agent Total Reward", "metrics": ["agent0_breakdown_total_sum", "agent1_breakdown_total_sum"]},
        ],
        reward_summary["default_x_key"],
    )
    train_groups = _build_chart_groups(
        train_summary["rows"],
        [
            {"title": "Critic Value", "metrics": ["value_mean", "return_mean"]},
            {"title": "Losses", "metrics": ["loss", "policy_loss", "value_loss"]},
            {"title": "Policy Signals", "metrics": ["reward_mean", "adv_mean", "entropy"]},
            {"title": "Stability", "metrics": ["explained_var"]},
        ],
        train_summary["default_x_key"],
    )
    return {
        "reward": {**reward_summary, "groups": reward_groups},
        "train": {**train_summary, "groups": train_groups},
    }


def main() -> None:
    host = os.environ.get("PT_VIEWER_HOST", "127.0.0.1")
    port = int(os.environ.get("PT_VIEWER_PORT", "8765"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
