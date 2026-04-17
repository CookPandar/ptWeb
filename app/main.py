from __future__ import annotations

import csv
import json
import os
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
HOME_DIR = BASE_DIR.parent


def _resolve_collab_root() -> Path:
    env_root = os.getenv("PTWEB_COLLAB_ROOT", "").strip()
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            HOME_DIR / "Collab-Overcooked-1",
            HOME_DIR / "Collab-Overcooked",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else (HOME_DIR / "Collab-Overcooked-1").resolve()


COLLAB_ROOT = _resolve_collab_root()
DEFAULT_EXPERIMENT_ROOT = Path(
    os.getenv(
        "PTWEB_EXPERIMENT_ROOT",
        "/mnt/volumes/ss-sai-bd-ga/zhangshuwen/Collab-Overcooked-exp",
    )
).expanduser()
DEFAULT_BASELINE_ROOT = DEFAULT_EXPERIMENT_ROOT / "no_paired_comm"
DEFAULT_PAIRED_ROOT = DEFAULT_EXPERIMENT_ROOT / "paired_comm"
DEFAULT_DATA_DIR = COLLAB_ROOT / "rollouts_kl"
DEFAULT_FILE = DEFAULT_DATA_DIR / "rollout_rank0_u00001.pt"
DEFAULT_TOKENIZER = COLLAB_ROOT / "runs" / "Chef"
DEFAULT_TRAIN_DIR = COLLAB_ROOT / "runs" / "rl" / "train"
DEFAULT_REWARD_CSV = DEFAULT_TRAIN_DIR / "reward_curve.csv"
DEFAULT_TRAIN_CSV = DEFAULT_TRAIN_DIR / "train_curve.csv"
DEFAULT_TRAIN_KL_DIR = COLLAB_ROOT / "runs" / "rl" / "train_kl"
DEFAULT_REWARD_KL_CSV = DEFAULT_TRAIN_KL_DIR / "reward_curve.csv"
DEFAULT_TRAIN_KL_CSV = DEFAULT_TRAIN_KL_DIR / "train_curve.csv"
DEFAULT_TRAIN_KL_LRLOW_DIR = COLLAB_ROOT / "runs" / "rl" / "train_kl_actorlr100x"
DEFAULT_REWARD_KL_LRLOW_CSV = DEFAULT_TRAIN_KL_LRLOW_DIR / "reward_curve.csv"
DEFAULT_TRAIN_KL_LRLOW_CSV = DEFAULT_TRAIN_KL_LRLOW_DIR / "train_curve.csv"
DEFAULT_EVAL_DIR = COLLAB_ROOT / "results" / "rl_eval"
DEFAULT_EVAL_KL_DIR = COLLAB_ROOT / "results" / "rl_eval_kl"
DEFAULT_COLLECT_KL_LRLOW_DIR = COLLAB_ROOT / "results" / "rl_collect_kl_actorlr100x"
DEFAULT_EVAL_KL_LRLOW_DIR = COLLAB_ROOT / "results" / "rl_eval_kl_actorlr100x"
DEFAULT_POLICY_RECORDS_DIR = COLLAB_ROOT / "runs" / "rl_policy_records"
DEFAULT_COMPARE_A_LABEL = "No Paired Comm"
DEFAULT_COMPARE_B_LABEL = "Paired Comm"
DEFAULT_COMPARE_A_REWARD_CSV = DEFAULT_BASELINE_ROOT / "runs" / "rl" / "train_kl" / "reward_curve.csv"
DEFAULT_COMPARE_A_TRAIN_CSV = DEFAULT_BASELINE_ROOT / "runs" / "rl" / "train_kl" / "train_curve.csv"
DEFAULT_COMPARE_A_EVAL_DIR = DEFAULT_BASELINE_ROOT / "results" / "rl_eval_kl"
DEFAULT_COMPARE_B_REWARD_CSV = DEFAULT_PAIRED_ROOT / "runs" / "rl" / "train_kl" / "reward_curve.csv"
DEFAULT_COMPARE_B_TRAIN_CSV = DEFAULT_PAIRED_ROOT / "runs" / "rl" / "train_kl" / "train_curve.csv"
DEFAULT_COMPARE_B_EVAL_DIR = DEFAULT_PAIRED_ROOT / "results" / "rl_eval_kl"
MAX_LIST_FILES = 200
MAX_POLICY_RECORD_FILES = 5000
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


def _preview_text(value: Any, limit: int = 160) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _extract_observation_text(prompt: Any) -> str | None:
    if not isinstance(prompt, str):
        return None
    markers = [
        "Current Observation:",
        "<input>\nCurrent Observation:",
    ]
    start = -1
    marker_len = 0
    for marker in markers:
        start = prompt.find(marker)
        if start >= 0:
            marker_len = len(marker)
            break
    if start < 0:
        return None
    observation = prompt[start + marker_len :].strip()
    return observation or None


def _extract_observation_timestep(text: Any) -> int | None:
    if not isinstance(text, str):
        return None
    first_lines = text.strip().splitlines()[:8]
    for line in first_lines:
        stripped = line.strip()
        if not stripped.lower().startswith("timestep"):
            continue
        prefix, _, _ = stripped.partition(":")
        parts = prefix.split()
        if len(parts) < 2:
            continue
        try:
            return int(parts[1])
        except ValueError:
            continue
    return None


def _normalized_policy_call_type(row: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    for key in ("semantic_call_type", "call_type"):
        value = metadata.get(key) if key in metadata else row.get(key)
        if isinstance(value, str) and value:
            normalized = value.strip().lower()
            if normalized == "format_correction":
                return "format_correct"
            return normalized
    return None


def _policy_record_overview(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    agents = sorted(
        {
            row["agent_index"]
            for row in rows
            if isinstance(row.get("agent_index"), int)
        }
    )
    call_types: dict[str, int] = {}
    timesteps = [
        row["timestep"]
        for row in rows
        if isinstance(row.get("timestep"), int)
    ]
    total_reward = 0.0
    reward_count = 0
    for row in rows:
        call_type = row.get("call_type")
        if isinstance(call_type, str) and call_type:
            call_types[call_type] = call_types.get(call_type, 0) + 1
        reward = row.get("reward")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            total_reward += float(reward)
            reward_count += 1
    stat = path.stat()
    return {
        "path": str(path),
        "file_name": path.name,
        "session_name": path.parent.name,
        "mtime": stat.st_mtime,
        "size_bytes": stat.st_size,
        "row_count": len(rows),
        "agents": agents,
        "timestep_min": min(timesteps) if timesteps else None,
        "timestep_max": max(timesteps) if timesteps else None,
        "call_types": call_types,
        "reward_mean": (total_reward / reward_count) if reward_count else None,
    }


def _policy_record_file_entry(root: Path, path: Path) -> dict[str, Any]:
    stat = path.stat()
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        relative_path = path.name
    return {
        "name": path.name,
        "path": str(path),
        "relative_path": str(relative_path),
        "session_name": path.parent.name,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
    }


@lru_cache(maxsize=16)
def _load_jsonl_cached(path_str: str, mtime_ns: int) -> list[dict[str, Any]]:
    del mtime_ns
    rows: list[dict[str, Any]] = []
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx} in {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object on line {idx} in {path}")
            rows.append(value)
    return rows


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    stat = path.stat()
    return _load_jsonl_cached(str(path), stat.st_mtime_ns)


def _build_policy_record_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[:MAX_ITEMS]):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        reward_breakdown = (
            metadata.get("reward_breakdown")
            if isinstance(metadata.get("reward_breakdown"), dict)
            else {}
        )
        prompt = row.get("prompt")
        response = row.get("response")
        observation = _extract_observation_text(prompt) or prompt
        table_rows.append(
            {
                "index": idx,
                "timestep": row.get("timestep"),
                "agent_index": row.get("agent_index"),
                "call_type": row.get("call_type"),
                "semantic_call_type": metadata.get("semantic_call_type"),
                "reward": row.get("reward"),
                "done": row.get("done"),
                "value": metadata.get("value"),
                "log_prob": metadata.get("log_prob"),
                "entropy": metadata.get("entropy"),
                "token_count": metadata.get("token_count"),
                "observation_preview": _preview_text(observation),
                "response_preview": _preview_text(response),
                "sequence_reward": reward_breakdown.get("sequence_reward"),
                "format_reward": reward_breakdown.get("format_reward"),
                "validator_reward": reward_breakdown.get("validator_reward"),
                "communication_reward": reward_breakdown.get("communication_reward"),
            }
        )
    return table_rows


def _policy_record_detail(row: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    reward_breakdown = (
        metadata.get("reward_breakdown")
        if isinstance(metadata.get("reward_breakdown"), dict)
        else {}
    )
    prompt = row.get("prompt")
    observation = _extract_observation_text(prompt) or prompt
    observation_timestep = _extract_observation_timestep(observation) or _extract_observation_timestep(prompt)
    normalized_call_type = _normalized_policy_call_type(row, metadata)
    return {
        "index": index,
        "timestep": row.get("timestep"),
        "observation_timestep": observation_timestep,
        "agent_index": row.get("agent_index"),
        "call_type": row.get("call_type"),
        "normalized_call_type": normalized_call_type,
        "reward": row.get("reward"),
        "done": row.get("done"),
        "observation": observation,
        "prompt": prompt,
        "response": row.get("response"),
        "messages": row.get("messages"),
        "metadata": metadata,
        "reward_breakdown": reward_breakdown,
    }


def _policy_agent_paths(session_dir: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in sorted(session_dir.glob("agent_*.jsonl")):
        stem = path.stem
        try:
            agent_idx = int(stem.split("_")[-1])
        except ValueError:
            continue
        paths[agent_idx] = path
    return paths


def _build_paired_policy_rows(session_dir: Path) -> dict[str, Any]:
    agent_paths = _policy_agent_paths(session_dir)
    agent0_path = agent_paths.get(0)
    agent1_path = agent_paths.get(1)
    agent0_rows = _load_jsonl_rows(agent0_path) if agent0_path else []
    agent1_rows = _load_jsonl_rows(agent1_path) if agent1_path else []
    details_by_agent: dict[int, list[dict[str, Any]]] = {
        0: [_policy_record_detail(row, index) for index, row in enumerate(agent0_rows)],
        1: [_policy_record_detail(row, index) for index, row in enumerate(agent1_rows)],
    }

    communication_by_agent: dict[int, dict[int | None, list[dict[str, Any]]]] = {0: {}, 1: {}}
    communication_indexes: dict[int, set[int]] = {0: set(), 1: set()}

    for agent_idx, details in details_by_agent.items():
        for detail in details:
            logical_timestep = detail.get("observation_timestep")
            if logical_timestep is None:
                logical_timestep = detail.get("timestep")
            detail["logical_timestep"] = logical_timestep
            if detail.get("normalized_call_type") == "communication":
                communication_by_agent[agent_idx].setdefault(logical_timestep, []).append(detail)
                communication_indexes[agent_idx].add(int(detail["index"]))

    event_rows: list[dict[str, Any]] = []
    event_details: list[dict[str, Any]] = []

    comm_timesteps = sorted(
        set(communication_by_agent[0].keys()) | set(communication_by_agent[1].keys()),
        key=lambda value: (-1 if value is None else value),
    )
    for logical_timestep in comm_timesteps:
        agent0_comm = communication_by_agent[0].get(logical_timestep, [])
        agent1_comm = communication_by_agent[1].get(logical_timestep, [])
        for slot_index in range(max(len(agent0_comm), len(agent1_comm))):
            agent0_detail = agent0_comm[slot_index] if slot_index < len(agent0_comm) else None
            agent1_detail = agent1_comm[slot_index] if slot_index < len(agent1_comm) else None
            event_kind = (
                "communication_pair"
                if agent0_detail is not None and agent1_detail is not None
                else "communication_single"
            )
            pair_status = "paired" if event_kind == "communication_pair" else "unmatched"
            event_rows.append(
                {
                    "index": len(event_rows),
                    "event_kind": event_kind,
                    "pair_status": pair_status,
                    "timestep": logical_timestep,
                    "slot_index": slot_index,
                    "agent0_row_index": agent0_detail.get("index") if agent0_detail else None,
                    "agent0_call_type": agent0_detail.get("normalized_call_type") if agent0_detail else None,
                    "agent0_reward": agent0_detail.get("reward") if agent0_detail else None,
                    "agent0_observation_preview": _preview_text(agent0_detail.get("observation")) if agent0_detail else None,
                    "agent1_row_index": agent1_detail.get("index") if agent1_detail else None,
                    "agent1_call_type": agent1_detail.get("normalized_call_type") if agent1_detail else None,
                    "agent1_reward": agent1_detail.get("reward") if agent1_detail else None,
                    "agent1_observation_preview": _preview_text(agent1_detail.get("observation")) if agent1_detail else None,
                }
            )
            event_details.append(
                {
                    "index": len(event_details),
                    "event_kind": event_kind,
                    "pair_status": pair_status,
                    "timestep": logical_timestep,
                    "slot_index": slot_index,
                    "agent0": agent0_detail,
                    "agent1": agent1_detail,
                }
            )

    single_events: list[tuple[int | None, int, dict[str, Any]]] = []
    for agent_idx, details in details_by_agent.items():
        for detail in details:
            detail_index = int(detail["index"])
            if detail_index in communication_indexes[agent_idx]:
                continue
            logical_timestep = detail.get("logical_timestep")
            single_events.append((logical_timestep, len(single_events), {"agent_idx": agent_idx, "detail": detail}))

    single_events.sort(key=lambda item: (-1 if item[0] is None else item[0], item[2]["agent_idx"], int(item[2]["detail"]["index"])))
    for logical_timestep, _, payload in single_events:
        agent_idx = payload["agent_idx"]
        detail = payload["detail"]
        event_rows.append(
            {
                "index": len(event_rows),
                "event_kind": "single",
                "pair_status": "single",
                "timestep": logical_timestep,
                "slot_index": None,
                "agent0_row_index": detail.get("index") if agent_idx == 0 else None,
                "agent0_call_type": detail.get("normalized_call_type") if agent_idx == 0 else None,
                "agent0_reward": detail.get("reward") if agent_idx == 0 else None,
                "agent0_observation_preview": _preview_text(detail.get("observation")) if agent_idx == 0 else None,
                "agent1_row_index": detail.get("index") if agent_idx == 1 else None,
                "agent1_call_type": detail.get("normalized_call_type") if agent_idx == 1 else None,
                "agent1_reward": detail.get("reward") if agent_idx == 1 else None,
                "agent1_observation_preview": _preview_text(detail.get("observation")) if agent_idx == 1 else None,
            }
        )
        event_details.append(
            {
                "index": len(event_details),
                "event_kind": "single",
                "pair_status": "single",
                "timestep": logical_timestep,
                "slot_index": None,
                "agent0": detail if agent_idx == 0 else None,
                "agent1": detail if agent_idx == 1 else None,
            }
        )

    event_rows = sorted(
        event_rows,
        key=lambda row: (
            -1 if row["timestep"] is None else row["timestep"],
            {"communication_pair": 0, "communication_single": 1, "single": 2}.get(row["event_kind"], 3),
            -1 if row["slot_index"] is None else row["slot_index"],
            -1 if row["agent0_row_index"] is None else row["agent0_row_index"],
            -1 if row["agent1_row_index"] is None else row["agent1_row_index"],
        ),
    )
    detail_map = {detail["index"]: detail for detail in event_details}
    reordered_details: list[dict[str, Any]] = []
    reordered_rows: list[dict[str, Any]] = []
    for new_index, row in enumerate(event_rows):
        original_index = row["index"]
        detail = detail_map[original_index]
        row["index"] = new_index
        detail["index"] = new_index
        reordered_rows.append(row)
        reordered_details.append(detail)

    overview = {
        "session_dir": str(session_dir),
        "agent_files": {
            "agent_0": str(agent0_path) if agent0_path else None,
            "agent_1": str(agent1_path) if agent1_path else None,
        },
        "agent_row_counts": {
            "agent_0": len(agent0_rows),
            "agent_1": len(agent1_rows),
        },
        "paired_row_count": len(reordered_rows),
        "pairing_rule": (
            "Only communication rows are paired. Pairing uses observation timestep extracted "
            "from prompt text, then aligns communication rows by order within that timestep. "
            "planner_main / format_correct rows stay as single-agent events."
        ),
    }
    return {
        "overview": overview,
        "rows": reordered_rows[:MAX_ITEMS],
        "details": reordered_details[:MAX_ITEMS],
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


def _build_split_chart_groups(
    rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    default_x_key: str,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    numeric_columns = _csv_numeric_columns(rows)
    for group in groups:
        charts: list[dict[str, Any]] = []
        for subchart in group["subcharts"]:
            metrics = [name for name in subchart["metrics"] if name in numeric_columns]
            charts.append(
                {
                    "title": subchart["title"],
                    "metrics": metrics,
                    "chart": _series_payload(rows, default_x_key, metrics)
                    if default_x_key and metrics
                    else {"x_key": default_x_key, "series": []},
                }
            )
        payloads.append(
            {
                "title": group["title"],
                "charts": charts,
            }
        )
    return payloads


def _pick_metrics(numeric_columns: list[str], candidates: list[str]) -> list[str]:
    return [name for name in candidates if name in numeric_columns]


def _has_any_metric(numeric_columns: list[str], candidates: list[str]) -> bool:
    return any(name in numeric_columns for name in candidates)


def _existing_csv_in_dir(directory: Path, filename: str) -> Path | None:
    path = directory / filename
    if path.exists() and path.is_file():
        return path
    return None


def _static_version() -> int:
    candidates = [
        BASE_DIR / "app" / "static" / "app.js",
        BASE_DIR / "app" / "static" / "style.css",
        BASE_DIR / "app" / "templates" / "index.html",
    ]
    mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
    return int(max(mtimes)) if mtimes else 0


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
            "default_reward_kl_csv": str(DEFAULT_REWARD_KL_CSV),
            "default_train_kl_csv": str(DEFAULT_TRAIN_KL_CSV),
            "default_reward_kl_lrlow_csv": str(DEFAULT_REWARD_KL_LRLOW_CSV),
            "default_train_kl_lrlow_csv": str(DEFAULT_TRAIN_KL_LRLOW_CSV),
            "default_eval_dir": str(DEFAULT_EVAL_DIR),
            "default_eval_kl_dir": str(DEFAULT_EVAL_KL_DIR),
            "default_collect_kl_lrlow_dir": str(DEFAULT_COLLECT_KL_LRLOW_DIR),
            "default_eval_kl_lrlow_dir": str(DEFAULT_EVAL_KL_LRLOW_DIR),
            "default_policy_records_dir": str(DEFAULT_POLICY_RECORDS_DIR),
            "default_compare_a_label": DEFAULT_COMPARE_A_LABEL,
            "default_compare_b_label": DEFAULT_COMPARE_B_LABEL,
            "default_compare_a_reward_csv": str(DEFAULT_COMPARE_A_REWARD_CSV),
            "default_compare_a_train_csv": str(DEFAULT_COMPARE_A_TRAIN_CSV),
            "default_compare_a_eval_dir": str(DEFAULT_COMPARE_A_EVAL_DIR),
            "default_compare_b_reward_csv": str(DEFAULT_COMPARE_B_REWARD_CSV),
            "default_compare_b_train_csv": str(DEFAULT_COMPARE_B_TRAIN_CSV),
            "default_compare_b_eval_dir": str(DEFAULT_COMPARE_B_EVAL_DIR),
            "default_compare_root": str(DEFAULT_EXPERIMENT_ROOT),
            "static_version": _static_version(),
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
    train_numeric = train_summary["numeric_columns"]
    reward_groups = _build_split_chart_groups(
        reward_summary["rows"],
        [
            {
                "title": "Cumulative Reward",
                "subcharts": [
                    {"title": "Agent 0", "metrics": ["agent0_rl_sum"]},
                    {"title": "Agent 1", "metrics": ["agent1_rl_sum"]},
                ],
            },
            {
                "title": "Process Reward",
                "subcharts": [
                    {"title": "Agent 0", "metrics": ["agent0_legacy_process_sum"]},
                    {"title": "Agent 1", "metrics": ["agent1_legacy_process_sum"]},
                ],
            },
            {
                "title": "Communication Reward",
                "subcharts": [
                    {"title": "Agent 0", "metrics": ["agent0_comm_sum"]},
                    {"title": "Agent 1", "metrics": ["agent1_comm_sum"]},
                ],
            },
            {
                "title": "Format Penalty",
                "subcharts": [
                    {"title": "Agent 0", "metrics": ["agent0_format_sum"]},
                    {"title": "Agent 1", "metrics": ["agent1_format_sum"]},
                ],
            },
            {
                "title": "Validator Penalty",
                "subcharts": [
                    {"title": "Agent 0", "metrics": ["agent0_validator_sum"]},
                    {"title": "Agent 1", "metrics": ["agent1_validator_sum"]},
                ],
            },
            {
                "title": "Reward Breakdown Total",
                "subcharts": [
                    {"title": "Agent 0", "metrics": ["agent0_breakdown_total_sum"]},
                    {"title": "Agent 1", "metrics": ["agent1_breakdown_total_sum"]},
                ],
            },
        ],
        reward_summary["default_x_key"],
    )
    if _pick_metrics(
        train_numeric,
        ["agent0_adv_mean", "agent0_return_mean", "agent0_value_mean"],
    ) or _pick_metrics(
        train_numeric,
        ["agent1_adv_mean", "agent1_return_mean", "agent1_value_mean"],
    ):
        train_group_defs = [
            {
                "title": "Agent 0 Critic",
                "metrics": [
                    "agent0_adv_mean",
                    "agent0_return_mean",
                    "agent0_value_mean",
                ],
            },
            {
                "title": "Agent 1 Critic",
                "metrics": [
                    "agent1_adv_mean",
                    "agent1_return_mean",
                    "agent1_value_mean",
                ],
            },
            {
                "title": "Critic Stability",
                "metrics": [
                    "agent0_explained_var",
                    "agent1_explained_var",
                ],
            },
            {
                "title": "Optimization",
                "metrics": [
                    "loss",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                ],
            },
        ]
        if _has_any_metric(
            train_numeric,
            ["approx_kl", "kl_penalty", "kl_penalty_coef"],
        ):
            train_group_defs.append(
                {
                    "title": "KL Divergence",
                    "metrics": [
                        "approx_kl",
                        "kl_penalty",
                        "kl_penalty_coef",
                    ],
                }
            )
        if _has_any_metric(
            train_numeric,
            [
                "clipfrac",
                "value_clipfrac",
            ],
        ):
            train_group_defs.append(
                {
                    "title": "PPO Clip Fraction",
                    "metrics": [
                        "clipfrac",
                        "value_clipfrac",
                    ],
                }
            )
        if _has_any_metric(
            train_numeric,
            [
                "optimizer_steps",
                "stopped_early",
            ],
        ):
            train_group_defs.append(
                {
                    "title": "Optimizer Steps",
                    "metrics": [
                        "optimizer_steps",
                        "stopped_early",
                    ],
                }
            )
        if _has_any_metric(
            train_numeric,
            [
                "actor0_grad_norm",
                "actor1_grad_norm",
                "critic_adapter_grad_norm",
                "value_head_grad_norm",
            ],
        ):
            train_group_defs.append(
                {
                    "title": "Grad Norms",
                    "metrics": [
                        "actor0_grad_norm",
                        "actor1_grad_norm",
                        "critic_adapter_grad_norm",
                        "value_head_grad_norm",
                    ],
                }
            )
        if _has_any_metric(
            train_numeric,
            [
                "actor0_param_delta",
                "actor1_param_delta",
                "critic_adapter_param_delta",
                "value_head_param_delta",
            ],
        ):
            train_group_defs.append(
                {
                    "title": "Param Delta",
                    "metrics": [
                        "actor0_param_delta",
                        "actor1_param_delta",
                        "critic_adapter_param_delta",
                        "value_head_param_delta",
                    ],
                }
            )
    else:
        train_group_defs = [
            {"title": "Critic Value", "metrics": ["value_mean", "return_mean"]},
            {"title": "Losses", "metrics": ["loss", "policy_loss", "value_loss"]},
            {"title": "Policy Signals", "metrics": ["reward_mean", "adv_mean", "entropy"]},
            {"title": "Stability", "metrics": ["explained_var"]},
        ]
        if _has_any_metric(
            train_numeric,
            ["approx_kl", "kl_penalty", "kl_penalty_coef"],
        ):
            train_group_defs.append(
                {
                    "title": "KL Divergence",
                    "metrics": [
                        "approx_kl",
                        "kl_penalty",
                        "kl_penalty_coef",
                    ],
                }
            )
        if _has_any_metric(
            train_numeric,
            ["clipfrac", "value_clipfrac"],
        ):
            train_group_defs.append(
                {
                    "title": "PPO Clip Fraction",
                    "metrics": [
                        "clipfrac",
                        "value_clipfrac",
                    ],
                }
            )
        if _has_any_metric(
            train_numeric,
            ["optimizer_steps", "stopped_early"],
        ):
            train_group_defs.append(
                {
                    "title": "Optimizer Steps",
                    "metrics": [
                        "optimizer_steps",
                        "stopped_early",
                    ],
                }
            )
    train_groups = _build_chart_groups(
        train_summary["rows"],
        train_group_defs,
        train_summary["default_x_key"],
    )
    return {
        "reward": {**reward_summary, "groups": reward_groups},
        "train": {**train_summary, "groups": train_groups},
    }


@app.get("/api/eval_monitor")
async def get_eval_monitor_summary(
    eval_dir: str = Query(default=str(DEFAULT_EVAL_DIR)),
) -> dict[str, Any]:
    directory = _normalize_path(eval_dir)
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {directory}")

    performance_path = _existing_csv_in_dir(directory, "performance_curve.csv")
    episode_path = _existing_csv_in_dir(directory, "episode_return_curve_rank0.csv")
    reward_path = _existing_csv_in_dir(directory, "reward_curve.csv")

    if not any([performance_path, episode_path, reward_path]):
        raise HTTPException(
            status_code=404,
            detail=(
                f"No supported eval CSV files found in {directory}. "
                "Expected performance_curve.csv, episode_return_curve_rank0.csv, or reward_curve.csv."
            ),
        )

    payload: dict[str, Any] = {
        "eval_dir": str(directory),
        "mtime": 0.0,
    }

    if performance_path is not None:
        performance_summary = _csv_summary(
            performance_path,
            preferred_metrics=["success_rate", "avg_episode_return", "avg_team_custom_return"],
        )
        performance_numeric = performance_summary["numeric_columns"]
        performance_groups = _build_chart_groups(
            performance_summary["rows"],
            [
                {
                    "title": "Success And Return",
                    "metrics": [
                        "success_rate",
                        "avg_episode_return",
                        "avg_team_custom_return",
                    ],
                },
                {
                    "title": "Environment Throughput",
                    "metrics": [
                        "env_steps",
                        "episodes_completed",
                        "policy_calls",
                        "avg_calls_per_step",
                    ],
                },
                {
                    "title": "Per-Agent Return",
                    "metrics": [
                        "avg_agent0_custom_return",
                        "avg_agent1_custom_return",
                        "avg_team_custom_return",
                    ],
                },
            ],
            performance_summary["default_x_key"],
        )
        if _has_any_metric(
            performance_numeric,
            ["env_reward_sum", "avg_step_reward", "positive_reward_steps"],
        ):
            performance_groups.append(
                {
                    "title": "Reward Density",
                    "metrics": _pick_metrics(
                        performance_numeric,
                        ["env_reward_sum", "avg_step_reward", "positive_reward_steps"],
                    ),
                    "chart": _series_payload(
                        performance_summary["rows"],
                        performance_summary["default_x_key"],
                        _pick_metrics(
                            performance_numeric,
                            ["env_reward_sum", "avg_step_reward", "positive_reward_steps"],
                        ),
                    ),
                }
            )
        payload["performance"] = {**performance_summary, "groups": performance_groups}
        payload["mtime"] = max(payload["mtime"], performance_summary["mtime"])

    if episode_path is not None:
        episode_summary = _csv_summary(
            episode_path,
            preferred_metrics=["team_custom_return", "agent0_custom_return", "agent1_custom_return"],
        )
        episode_numeric = episode_summary["numeric_columns"]
        episode_groups = _build_chart_groups(
            episode_summary["rows"],
            [
                {
                    "title": "Episode Returns",
                    "metrics": [
                        "episode_return",
                        "team_custom_return",
                        "agent0_custom_return",
                        "agent1_custom_return",
                    ],
                },
                {
                    "title": "Agent 0 Breakdown",
                    "metrics": [
                        "agent0_sequence_sum",
                        "agent0_format_sum",
                        "agent0_validator_sum",
                        "agent0_comm_sum",
                    ],
                },
                {
                    "title": "Agent 1 Breakdown",
                    "metrics": [
                        "agent1_sequence_sum",
                        "agent1_format_sum",
                        "agent1_validator_sum",
                        "agent1_comm_sum",
                    ],
                },
            ],
            episode_summary["default_x_key"],
        )
        if _has_any_metric(episode_numeric, ["episode_len", "had_positive"]):
            episode_groups.append(
                {
                    "title": "Episode Flags",
                    "metrics": _pick_metrics(episode_numeric, ["episode_len", "had_positive"]),
                    "chart": _series_payload(
                        episode_summary["rows"],
                        episode_summary["default_x_key"],
                        _pick_metrics(episode_numeric, ["episode_len", "had_positive"]),
                    ),
                }
            )
        payload["episode"] = {**episode_summary, "groups": episode_groups}
        payload["mtime"] = max(payload["mtime"], episode_summary["mtime"])

    if reward_path is not None:
        reward_summary = _csv_summary(
            reward_path,
            preferred_metrics=["agent0_rl_sum", "agent1_rl_sum"],
        )
        reward_groups = _build_split_chart_groups(
            reward_summary["rows"],
            [
                {
                    "title": "Cumulative Reward",
                    "subcharts": [
                        {"title": "Agent 0", "metrics": ["agent0_rl_sum"]},
                        {"title": "Agent 1", "metrics": ["agent1_rl_sum"]},
                    ],
                },
                {
                    "title": "Process Reward",
                    "subcharts": [
                        {"title": "Agent 0", "metrics": ["agent0_legacy_process_sum"]},
                        {"title": "Agent 1", "metrics": ["agent1_legacy_process_sum"]},
                    ],
                },
                {
                    "title": "Communication Reward",
                    "subcharts": [
                        {"title": "Agent 0", "metrics": ["agent0_comm_sum"]},
                        {"title": "Agent 1", "metrics": ["agent1_comm_sum"]},
                    ],
                },
                {
                    "title": "Format Penalty",
                    "subcharts": [
                        {"title": "Agent 0", "metrics": ["agent0_format_sum"]},
                        {"title": "Agent 1", "metrics": ["agent1_format_sum"]},
                    ],
                },
                {
                    "title": "Validator Penalty",
                    "subcharts": [
                        {"title": "Agent 0", "metrics": ["agent0_validator_sum"]},
                        {"title": "Agent 1", "metrics": ["agent1_validator_sum"]},
                    ],
                },
                {
                    "title": "Reward Breakdown Total",
                    "subcharts": [
                        {"title": "Agent 0", "metrics": ["agent0_breakdown_total_sum"]},
                        {"title": "Agent 1", "metrics": ["agent1_breakdown_total_sum"]},
                    ],
                },
            ],
            reward_summary["default_x_key"],
        )
        payload["reward"] = {**reward_summary, "groups": reward_groups}
        payload["mtime"] = max(payload["mtime"], reward_summary["mtime"])

    return payload


@app.get("/api/policy_records/sessions")
async def list_policy_record_sessions(
    dir_path: str = Query(default=str(DEFAULT_POLICY_RECORDS_DIR)),
) -> dict[str, Any]:
    directory = _normalize_path(dir_path)
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {directory}")

    sessions = []
    session_dirs = [path for path in directory.iterdir() if path.is_dir()]
    for path in sorted(session_dirs, key=lambda item: item.stat().st_mtime, reverse=True):
        jsonl_files = sorted(path.glob("*.jsonl"))
        if not jsonl_files:
            continue
        stat = path.stat()
        sessions.append(
            {
                "name": path.name,
                "path": str(path),
                "mtime": stat.st_mtime,
                "file_count": len(jsonl_files),
                "files": [file.name for file in jsonl_files[:8]],
            }
        )
        if len(sessions) >= MAX_LIST_FILES:
            break
    return {"directory": str(directory), "sessions": sessions}


@app.get("/api/policy_records/all_files")
async def list_all_policy_record_files(
    dir_path: str = Query(default=str(DEFAULT_POLICY_RECORDS_DIR)),
) -> dict[str, Any]:
    directory = _normalize_path(dir_path)
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {directory}")

    files = []
    for path in sorted(directory.rglob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_file():
            continue
        files.append(_policy_record_file_entry(directory, path))
        if len(files) >= MAX_POLICY_RECORD_FILES:
            break
    return {"directory": str(directory), "files": files}


@app.get("/api/policy_records/files")
async def list_policy_record_files(session_path: str) -> dict[str, Any]:
    directory = _normalize_path(session_path)
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {directory}")

    files = []
    for path in sorted(directory.glob("*.jsonl"))[:MAX_LIST_FILES]:
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return {"session_path": str(directory), "files": files}


@app.get("/api/policy_records/file")
async def get_policy_record_file(path: str) -> dict[str, Any]:
    file_path = _normalize_path(path)
    if file_path.suffix.lower() != ".jsonl":
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported.")
    rows = _load_jsonl_rows(file_path)
    table_rows = _build_policy_record_rows(rows)
    return {
        "path": str(file_path),
        "overview": _policy_record_overview(file_path, table_rows),
        "rows": table_rows,
    }


@app.get("/api/policy_records/paired_file")
async def get_paired_policy_record_file(path: str) -> dict[str, Any]:
    file_path = _normalize_path(path)
    if file_path.suffix.lower() != ".jsonl":
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported.")
    session_dir = file_path.parent
    paired = _build_paired_policy_rows(session_dir)
    return {
        "path": str(file_path),
        "session_dir": str(session_dir),
        "overview": paired["overview"],
        "rows": paired["rows"],
    }


@app.get("/api/policy_records/item")
async def get_policy_record_item(
    path: str,
    index: int = Query(ge=0),
) -> dict[str, Any]:
    file_path = _normalize_path(path)
    if file_path.suffix.lower() != ".jsonl":
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported.")
    rows = _load_jsonl_rows(file_path)
    if index >= len(rows):
        raise HTTPException(status_code=404, detail=f"Index out of range: {index}")
    return {
        "path": str(file_path),
        "detail": _policy_record_detail(rows[index], index),
    }


@app.get("/api/policy_records/paired_item")
async def get_paired_policy_record_item(
    path: str,
    index: int = Query(ge=0),
) -> dict[str, Any]:
    file_path = _normalize_path(path)
    if file_path.suffix.lower() != ".jsonl":
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported.")
    paired = _build_paired_policy_rows(file_path.parent)
    details = paired["details"]
    if index >= len(details):
        raise HTTPException(status_code=404, detail=f"Index out of range: {index}")
    return {
        "path": str(file_path),
        "session_dir": str(file_path.parent),
        "detail": details[index],
    }


def main() -> None:
    host = os.environ.get("PT_VIEWER_HOST", "127.0.0.1")
    port = int(os.environ.get("PT_VIEWER_PORT", "8765"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
