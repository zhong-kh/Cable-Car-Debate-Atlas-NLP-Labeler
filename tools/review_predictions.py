"""Read-only prediction review dashboard with a separate human-audit store.

The selected predictions SQLite file is always opened in read-only mode. Human
reviews are written to data/audit_reviews.sqlite3, never into that file or the
annotation workbench database.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sqlite3
import statistics
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr


APP_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = APP_DIR.parent
DATA_DIR = APP_DIR / "data"
AUDIT_DB = DATA_DIR / "audit_reviews.sqlite3"
EXPORT_DIR = DATA_DIR / "exports"

TASK_NAMES = {
    "project_index": "Project Index",
    "support_score": "Support Score",
    "argument_types": "Argument Type(s)",
}
STANCE_OPTIONS = [
    ("Explicit position · assign 0–5", "scored"),
    ("Neutral / no clear preference", "neutral"),
    ("Mixed or conditional position", "mixed_or_conditional"),
    ("Mentions the project, no personal position", "no_stance"),
    ("Cannot determine / needs review", "unclear"),
]
PROJECT_STATUS_OPTIONS = [
    ("One or more named projects", "project_specific"),
    ("General cable-car discussion", "general"),
    ("Unrelated to a cable-car proposal", "unrelated"),
    ("Cannot determine", "unclear"),
]
ARGUMENTS = [
    ("Cost / funding / opportunity cost", "cost_finance"),
    ("Travel time / convenience", "time_convenience"),
    ("Accessibility / network connections", "accessibility_connectivity"),
    ("Capacity / reliability", "capacity_reliability"),
    ("Climate / emissions", "climate_emissions"),
    ("Nature / landscape / trees", "nature_landscape"),
    ("Noise / vibration", "noise_vibration"),
    ("Privacy / visual impact", "privacy_visual"),
    ("Economic growth / jobs / tourism", "economic_development"),
    ("Equity / who benefits or bears impact", "equity_distribution"),
    ("Safety", "safety"),
    ("Construction disruption", "construction_disruption"),
    ("Governance / transparency / participation", "governance_participation"),
    ("Technical / operational feasibility", "technical_feasibility"),
    ("Other argument", "other"),
]
DIRECTIONS = [
    ("Supports", "pro"),
    ("Opposes", "con"),
    ("Mixed / conditional", "mixed"),
    ("Neutral framing", "neutral"),
    ("Unclear", "unclear"),
]

DEFAULT_SOURCE = ROOT_DIR / "Participants - Future City Hackathon" / "Stadt Heilbronn" / "FSI_Heilbronn_Medienbeiträge (1).json"
PREDICTION_FILES = sorted(
    (path for path in (APP_DIR / "data" / "predictions").iterdir() if path.is_file() and path.suffix.casefold() in {".sqlite", ".sqlite3", ".db"}),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
) if (APP_DIR / "data" / "predictions").is_dir() else []
DEFAULT_PREDICTIONS = PREDICTION_FILES[0] if PREDICTION_FILES else APP_DIR / "data" / "predictions" / "predictions_20261114T143627Z_tfidf-baseline-20261114T142555Z.sqlite3"
DATA_CACHE: dict[str, dict[str, Any]] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_audit_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(AUDIT_DB) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS audit_batches (
                batch_id TEXT PRIMARY KEY,
                prediction_db_name TEXT NOT NULL,
                prediction_db_sha256 TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                model_version TEXT NOT NULL,
                task TEXT NOT NULL,
                seed INTEGER NOT NULL,
                population_n INTEGER NOT NULL,
                sampled_n INTEGER NOT NULL,
                include_training INTEGER NOT NULL,
                blind_mode INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_items (
                batch_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                mention_id TEXT NOT NULL,
                task TEXT NOT NULL,
                project_id TEXT NOT NULL DEFAULT '',
                prediction_json TEXT NOT NULL,
                confidence REAL,
                review_required INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                record_json TEXT NOT NULL,
                existing_labels_json TEXT NOT NULL DEFAULT '[]',
                existing_arguments_json TEXT NOT NULL DEFAULT '[]',
                human_label_json TEXT,
                reviewer TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                reviewed_at TEXT,
                PRIMARY KEY (batch_id, ordinal)
            );
            CREATE INDEX IF NOT EXISTS idx_audit_items_batch_reviewed
                ON audit_items(batch_id, reviewed_at, ordinal);
            CREATE TABLE IF NOT EXISTS quick_reviews (
                prediction_db_sha256 TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                mention_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                record_json TEXT NOT NULL,
                predictions_json TEXT NOT NULL,
                scoreable TEXT NOT NULL,
                human_score INTEGER,
                project_verdict TEXT NOT NULL,
                argument_verdict TEXT NOT NULL,
                reviewer TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                reviewed_at TEXT NOT NULL,
                PRIMARY KEY (prediction_db_sha256, mention_id, project_id)
            );
            """
        )


def file_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    try:
        path = Path(str(value)).expanduser().resolve()
    except (TypeError, ValueError, OSError):
        return None
    return path if path.is_file() else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_source_items(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        for key in ("data", "records", "mentions", "items"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError("Source dataset must be a JSON array or an object containing data/records/items.")
    seen: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        base_id = str(item.get("mention_id") or item.get("id") or f"row_{i + 1}")
        seen[base_id] += 1
        mention_id = base_id if seen[base_id] == 1 else f"{base_id}#{seen[base_id]}"
        out.append({
            "mention_id": mention_id,
            "title": str(item.get("title") or ""),
            "content": str(item.get("content") or item.get("text") or item.get("message") or ""),
            "published_at": str(item.get("published_at") or item.get("date") or ""),
            "language_code": str(item.get("language_code") or item.get("language") or "unknown"),
            "media_type_name": str(item.get("media_type_name") or item.get("media_type") or "unknown"),
            "source_name": str(item.get("source_name") or ""),
            "url": str(item.get("url") or ""),
            "country_code": str(item.get("country_code") or ""),
            "sentiment": item.get("sentiment"),
        })
    if not out:
        raise ValueError("No message records found in source dataset.")
    return out


def load_source(path: Path) -> dict[str, dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            raw = json.load(handle)
        rows = normalize_source_items(raw)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = normalize_source_items(list(csv.DictReader(handle, delimiter=delimiter)))
    else:
        raise ValueError("Choose a JSON, CSV or TSV source dataset.")
    return {row["mention_id"]: row for row in rows}


def read_predictions_db(path: Path) -> dict[str, Any]:
    uri = path.as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "predictions" not in tables:
            raise ValueError("Selected SQLite file has no `predictions` table.")
        pred_rows = [dict(r) for r in con.execute("SELECT * FROM predictions ORDER BY task, mention_id, project_id")]
        if not pred_rows:
            raise ValueError("The `predictions` table is empty. Choose the separate Predictions SQLite, not the annotation-only database.")
        annotations = [dict(r) for r in con.execute("SELECT * FROM annotations ORDER BY mention_id, project_id")] if "annotations" in tables else []
        arguments = [dict(r) for r in con.execute("SELECT * FROM argument_labels ORDER BY mention_id, project_id, class_code")] if "argument_labels" in tables else []
    finally:
        con.close()
    return {"tables": tables, "predictions": pred_rows, "annotations": annotations, "arguments": arguments}


def load_data(prediction_file: Any, source_file: Any, source_path_text: str = ""):
    pred_path = file_path(prediction_file)
    source_path = file_path(source_file)
    if source_path is None and (source_path_text or "").strip():
        candidate_path = Path(source_path_text.strip()).expanduser()
        source_path = candidate_path.resolve() if candidate_path.is_file() else None
    if source_path is None and not (source_path_text or "").strip() and DEFAULT_SOURCE.is_file():
        source_path = DEFAULT_SOURCE
    if pred_path is None:
        return "", "Select a Predictions SQLite file first.", gr.update(choices=[], value=None), gr.update(choices=[], value=[]), gr.update(choices=[], value=[])
    if source_path is None:
        return "", "Select the original source JSON/CSV as well. SQLite stores predictions and labels, not the message text.", gr.update(choices=[], value=None), gr.update(choices=[], value=[]), gr.update(choices=[], value=[])
    try:
        db = read_predictions_db(pred_path)
        messages = load_source(source_path)
        pred_sha = sha256_file(pred_path)
        source_sha = sha256_file(source_path)
    except Exception as exc:
        return "", f"Could not load files: {exc}", gr.update(choices=[], value=None), gr.update(choices=[], value=[]), gr.update(choices=[], value=[])

    annotation_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in db["annotations"]:
        annotation_by_id[str(row["mention_id"])].append(row)

    project_names: dict[str, str] = {}
    for row in db["predictions"]:
        try:
            result = json.loads(row["result_json"])
        except (json.JSONDecodeError, TypeError):
            result = {}
        for candidate in result.get("project_candidates", []):
            pid = str(candidate.get("project_id") or "")
            if pid:
                project_names[pid] = str(candidate.get("project_name") or pid)
    for row in db["annotations"]:
        pid = str(row.get("project_id") or "")
        if pid and not pid.startswith("__"):
            project_names.setdefault(pid, str(row.get("other_project_name") or pid))
    project_choices = [(name, pid) for pid, name in sorted(project_names.items(), key=lambda kv: kv[1].casefold()) if not pid.startswith("__")]
    argument_choices = [
        (f"{name} · {direction_name}", f"{code}::{direction}")
        for name, code in ARGUMENTS for direction_name, direction in DIRECTIONS
    ]
    tasks = sorted({str(row["task"]) for row in db["predictions"] if row.get("task") in TASK_NAMES})
    model_versions = sorted({str(row.get("model_version") or "") for row in db["predictions"] if row.get("model_version")})
    handle = hashlib.sha256((pred_sha + source_sha).encode()).hexdigest()[:20]
    DATA_CACHE[handle] = {
        **db, "messages": messages, "annotation_by_id": annotation_by_id,
        "project_choices": project_choices,
        "argument_choices": argument_choices, "project_names": project_names, "prediction_path": pred_path,
        "source_path": source_path, "prediction_sha": pred_sha,
        "source_sha": source_sha, "model_version": ", ".join(model_versions) or "unknown",
    }
    counts = Counter(str(row["task"]) for row in db["predictions"])
    matched_predictions = sum(str(row["mention_id"]) in messages for row in db["predictions"])
    summary = (
        f"Loaded **{len(db['predictions']):,}** predictions across **{len(messages):,}** source messages; **{matched_predictions:,}** prediction rows matched by `mention_id`.  \n"
        f"Prediction file: `{pred_path.name}` · model: `{', '.join(model_versions) or 'unknown'}`  \n"
        f"Source file: `{source_path.name}` · existing human-annotated messages: **{len(annotation_by_id):,}**  \n"
        + " · ".join(f"{TASK_NAMES.get(task, task)}: {n:,}" for task, n in sorted(counts.items()))
    )
    task_choices = [(TASK_NAMES[task], task) for task in tasks]
    return handle, summary, gr.update(choices=task_choices, value=tasks[0] if tasks else None), gr.update(choices=project_choices, value=[]), gr.update(choices=argument_choices, value=[])


def get_audit_rows(batch_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(AUDIT_DB) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute("SELECT * FROM audit_items WHERE batch_id=? ORDER BY ordinal", (batch_id,))]


def make_sample(handle: str, task: str, size: float, seed: float, include_training: bool, blind_mode: bool):
    ctx = DATA_CACHE.get(handle)
    if not ctx:
        return empty_view({"batch_id": "", "index": 0}, "Load the SQLite and source files first.")
    task = str(task or "")
    if task not in TASK_NAMES:
        return empty_view({"batch_id": "", "index": 0}, "Choose a prediction task first.")
    candidates = []
    for row in ctx["predictions"]:
        if row.get("task") != task:
            continue
        try:
            result = json.loads(row["result_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        mention_id = str(row["mention_id"])
        if mention_id not in ctx["messages"]:
            continue
        trained = bool(result.get("is_training_example")) or mention_id in ctx["annotation_by_id"]
        if trained and not include_training:
            continue
        candidates.append((row, result, trained))
    if not candidates:
        return empty_view({"batch_id": "", "index": 0}, "No eligible prediction rows. Enable the diagnostic option to include human-labeled/training examples.")
    n = max(1, min(int(size or 1), len(candidates)))
    seed_int = int(seed or 0)
    picked = random.Random(seed_int).sample(candidates, n)
    batch_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    created = utc_now()
    with sqlite3.connect(AUDIT_DB) as con:
        con.execute(
            "INSERT INTO audit_batches VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (batch_id, ctx["prediction_path"].name, ctx["prediction_sha"], ctx["source_path"].name,
             ctx["source_sha"], ctx["model_version"], task, seed_int, len(candidates), n,
             int(bool(include_training)), int(bool(blind_mode)), created),
        )
        for ordinal, (row, result, trained) in enumerate(picked):
            mention_id = str(row["mention_id"])
            project_id = str(row.get("project_id") or "")
            record = ctx["messages"].get(mention_id, {"mention_id": mention_id})
            existing = ctx["annotation_by_id"].get(mention_id, [])
            args = [a for a in ctx["arguments"] if str(a.get("mention_id")) == mention_id and (not project_id or str(a.get("project_id")) == project_id)]
            snapshot = dict(record)
            snapshot["_in_sample"] = trained
            snapshot["_source_sentiment_note"] = "Source sentiment is not a project-stance gold label."
            con.execute(
                """INSERT INTO audit_items
                (batch_id,ordinal,mention_id,task,project_id,prediction_json,confidence,review_required,model_version,
                 record_json,existing_labels_json,existing_arguments_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (batch_id, ordinal, mention_id, task, project_id,
                 json.dumps(result, ensure_ascii=False), row.get("confidence"), int(bool(row.get("review_required"))),
                 str(row.get("model_version") or ctx["model_version"]), json.dumps(snapshot, ensure_ascii=False),
                 json.dumps(existing, ensure_ascii=False), json.dumps(args, ensure_ascii=False)),
            )
    return render_item(handle, {"batch_id": batch_id, "index": 0, "blind_mode": bool(blind_mode)}, f"Sample created: {n:,} records from a population of {len(candidates):,} (seed {seed_int}).")


def empty_view(state: dict[str, Any], message: str):
    return (
        state, "", "", "", "", "", gr.update(visible=False), {}, [], message,
        gr.update(visible=False), "unclear", gr.update(choices=[], value=[]),
        gr.update(visible=False), "unclear", 3,
        gr.update(visible=False), gr.update(choices=[], value=[]),
        "", "", message, "reviewable", False,
    )


def render_item(handle: str, state: dict[str, Any] | None, notice: str = ""):
    state = state or {"batch_id": "", "index": 0}
    blind_mode = bool(state.get("blind_mode", True))
    batch_id = str(state.get("batch_id") or "")
    if not batch_id:
        return empty_view(state, notice or "Load a database and draw a review sample.")
    rows = get_audit_rows(batch_id)
    if not rows:
        return empty_view(state, "Batch not found. Load the matching database and resume a valid batch ID.")
    idx = int(state.get("index", 0))
    if idx >= len(rows):
        reviewed = sum(bool(r["reviewed_at"]) for r in rows)
        done = list(empty_view(state, f"Batch complete: **{reviewed}/{len(rows)}** reviewed. Export the report below."))
        batch = get_batch(batch_id)
        done[1] = f"Batch `{batch_id}` · {TASK_NAMES.get(batch['task'], batch['task'])} · {reviewed:,}/{len(rows):,} reviewed"
        done[2] = "Sample complete"
        return tuple(done)
    idx = max(0, idx)
    state["index"] = idx
    row = rows[idx]
    try:
        result = json.loads(row["prediction_json"])
        record = json.loads(row["record_json"])
        existing_labels = json.loads(row["existing_labels_json"] or "[]")
        existing_args = json.loads(row["existing_arguments_json"] or "[]")
        human = json.loads(row["human_label_json"]) if row["human_label_json"] else None
    except (json.JSONDecodeError, TypeError):
        result, record, existing_labels, existing_args, human = {}, {}, [], [], None
    batch = get_batch(batch_id)
    project_choices = DATA_CACHE.get(handle, {}).get("project_choices", [])
    argument_choices = DATA_CACHE.get(handle, {}).get("argument_choices", [])
    if not project_choices:
        project_ids = {str(c.get("project_id")) for c in result.get("project_candidates", []) if c.get("project_id")}
        project_choices = [(pid, pid) for pid in sorted(project_ids)]
    predicted_projects = [str(c.get("project_id")) for c in result.get("project_candidates", []) if c.get("project_id")]
    if not argument_choices:
        argument_choices = [(f"{a.get('class_name') or a.get('class_code')} · {a.get('direction')}", f"{a.get('class_code')}::{a.get('direction')}") for a in result.get("arguments", [])]
    predicted_arguments = [f"{a.get('class_code')}::{a.get('direction')}" for a in result.get("arguments", []) if a.get("class_code")]

    task = row["task"]
    saved_project_labels = [a for a in existing_labels if a.get("relevance") == "project_specific"]
    saved_pair_label = next((a for a in existing_labels if str(a.get("project_id") or "") == str(row["project_id"])), None)
    if task == "project_index":
        if human:
            human_status = human.get("status", "project_specific")
            human_projects = human.get("project_ids", predicted_projects)
        elif saved_project_labels:
            human_status = "project_specific"
            human_projects = sorted({str(a.get("project_id")) for a in saved_project_labels if a.get("project_id")})
        elif existing_labels:
            saved_status = next((str(a.get("relevance")) for a in existing_labels if a.get("relevance") in {"general", "unrelated", "unclear"}), "unclear")
            human_status, human_projects = saved_status, []
        elif blind_mode:
            human_status, human_projects = None, []
        else:
            human_status, human_projects = ("project_specific" if predicted_projects else "unclear"), predicted_projects
    else:
        human_status, human_projects = "unclear", []
    if task == "support_score":
        if human:
            human_stance = human.get("stance_status", "scored")
            human_score = human.get("support_score", result.get("support_score_0_5", 3))
        elif saved_pair_label and saved_pair_label.get("relevance") == "project_specific":
            human_stance = saved_pair_label.get("stance_status") or "unclear"
            human_score = saved_pair_label.get("support_score") if saved_pair_label.get("support_score") is not None else result.get("support_score_0_5", 3)
        elif blind_mode:
            human_stance, human_score = None, None
        else:
            human_stance, human_score = "scored", result.get("support_score_0_5", 3)
    else:
        human_stance, human_score = "unclear", 3
    if human:
        human_arguments = human.get("arguments", predicted_arguments)
    elif saved_pair_label:
        human_arguments = sorted({f"{a.get('class_code')}::{a.get('direction')}" for a in existing_args if a.get("class_code")})
    elif blind_mode:
        human_arguments = []
    else:
        human_arguments = predicted_arguments
    show_prediction = not (blind_mode and not row.get("reviewed_at"))
    metadata = [
        f"ID: {row['mention_id']}",
        f"Task: {TASK_NAMES.get(task, task)}",
        f"Project target: {row['project_id'] or 'all candidates'}",
        f"Date: {record.get('published_at') or 'unknown'}",
        f"Language: {record.get('language_code') or 'unknown'}",
        f"Media: {record.get('media_type_name') or 'unknown'}",
        f"Source: {record.get('source_name') or 'unknown'}",
        f"Source sentiment: {record.get('sentiment', 'unknown')} (context only; not a stance label)",
    ]
    if show_prediction:
        metadata.extend([f"Model confidence: {row.get('confidence')}", f"Review required: {'yes' if row.get('review_required') else 'no'}"])
    url = record.get("url")
    if url:
        metadata.append(f"URL: {url}")
    current_notice = notice or ("Previously saved human review." if row.get("reviewed_at") else "Not reviewed yet.")
    review_state_value = "cannot_determine" if human and human.get("cannot_determine") else "reviewable"
    warning = "This row was used for training / already has a human annotation. Diagnostic only; do not count it as independent validation." if record.get("_in_sample") else "Not marked as a training example and no existing annotation for this message."
    unit = "message" if batch["task"] == "project_index" else "message × project"
    batch_info = (
        f"Batch `{batch_id}` · {TASK_NAMES.get(batch['task'], batch['task'])} · "
        f"unit: {unit} · random seed {batch['seed']} · population {batch['population_n']:,} · sample {batch['sampled_n']:,} · "
        f"{sum(bool(r['reviewed_at']) for r in rows):,} reviewed"
    )
    return (
        state, batch_info, f"Record {idx + 1:,} / {len(rows):,}",
        record.get("title") or "(No title)", "\n".join(metadata), record.get("content") or "[No content text in source file]",
        gr.update(visible=show_prediction), result if show_prediction else {}, {"annotations": existing_labels, "arguments": existing_args}, warning,
        gr.update(visible=task == "project_index"), human_status,
        gr.update(choices=project_choices, value=human_projects),
        gr.update(visible=task == "support_score"), human_stance, human_score,
        gr.update(visible=task == "argument_types"),
        gr.update(choices=argument_choices, value=human_arguments),
        row.get("reviewer") or "", row.get("notes") or "", current_notice,
        review_state_value, False,
    )


def get_batch(batch_id: str) -> dict[str, Any]:
    with sqlite3.connect(AUDIT_DB) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM audit_batches WHERE batch_id=?", (batch_id,)).fetchone()
        return dict(row) if row else {}


def resume_batch(handle: str, batch_id: str):
    batch_id = (batch_id or "").strip()
    batch = get_batch(batch_id)
    if not batch:
        return empty_view({"batch_id": "", "index": 0}, "Batch ID not found in the local audit review database.")
    ctx = DATA_CACHE.get(handle)
    if not ctx:
        return empty_view({"batch_id": "", "index": 0}, "Load the matching Predictions SQLite and source dataset before resuming this batch.")
    if ctx["prediction_sha"] != batch["prediction_db_sha256"]:
        return empty_view({"batch_id": "", "index": 0}, "The loaded Predictions SQLite does not match this batch.")
    if ctx["source_sha"] != batch["source_sha256"]:
        return empty_view({"batch_id": "", "index": 0}, "The loaded source dataset does not match this batch.")
    return render_item(handle, {"batch_id": batch_id, "index": 0, "blind_mode": bool(batch["blind_mode"])}, "Resumed saved batch.")


def compare_prediction(task: str, prediction: dict[str, Any], human: dict[str, Any]) -> str:
    if human.get("cannot_determine"):
        return "not_evaluable"
    if task == "project_index":
        status = human.get("status")
        if status == "unclear":
            return "not_evaluable"
        predicted = {str(c.get("project_id")) for c in prediction.get("project_candidates", []) if c.get("project_id")}
        gold = set(human.get("project_ids", [])) if status == "project_specific" else set()
    elif task == "support_score":
        status = human.get("stance_status")
        if status == "unclear":
            return "not_evaluable"
        if status != "scored":
            return "wrong"
        predicted_score = prediction.get("support_score_0_5")
        return "correct" if predicted_score is not None and int(predicted_score) == int(human["support_score"]) else "wrong"
    else:
        predicted = {f"{a.get('class_code')}::{a.get('direction')}" for a in prediction.get("arguments", []) if a.get("class_code")}
        gold = set(human.get("arguments", []))
    if predicted == gold:
        return "correct"
    if predicted & gold:
        return "partial"
    return "wrong"


def preserve_form(handle: str, state: dict[str, Any], message: str, reviewable: str,
                  project_status: str, human_projects: list[str] | None,
                  stance_status: str, human_score: int | None, human_arguments: list[str] | None,
                  reviewer: str, notes: str, confirmed: bool):
    values = list(render_item(handle, state, message))
    ctx = DATA_CACHE.get(handle, {})
    values[11] = project_status
    values[12] = gr.update(choices=ctx.get("project_choices", []), value=human_projects or [])
    values[14] = stance_status
    values[15] = human_score
    values[17] = gr.update(choices=ctx.get("argument_choices", []), value=human_arguments or [])
    values[18] = reviewer or ""
    values[19] = notes or ""
    values[21] = reviewable or "reviewable"
    values[22] = bool(confirmed)
    return tuple(values)


def save_current(handle: str, state: dict[str, Any], confirmed: bool, reviewable: str,
                 project_status: str, human_projects: list[str] | None,
                 stance_status: str, human_score: int | None, human_arguments: list[str] | None,
                 reviewer: str, notes: str, step: int = 0):
    batch_id = str((state or {}).get("batch_id") or "")
    rows = get_audit_rows(batch_id) if batch_id else []
    idx = int((state or {}).get("index", 0))
    if not rows or idx >= len(rows):
        return render_item(handle, state, "No current record to save.")
    if not confirmed:
        return preserve_form(handle, state, "Please confirm that you reviewed the source text before saving.", reviewable, project_status, human_projects, stance_status, human_score, human_arguments, reviewer, notes, confirmed)
    row = rows[idx]
    task = row["task"]
    try:
        prediction = json.loads(row["prediction_json"])
    except json.JSONDecodeError:
        prediction = {}
    if reviewable == "cannot_determine":
        human = {"cannot_determine": True}
    elif task == "project_index":
        if not project_status:
            return preserve_form(handle, state, "Choose a human project status.", reviewable, project_status, human_projects, stance_status, human_score, human_arguments, reviewer, notes, confirmed)
        human_projects = list(human_projects or [])
        if project_status == "project_specific" and not human_projects:
            return preserve_form(handle, state, "Select at least one named project, or change project status.", reviewable, project_status, human_projects, stance_status, human_score, human_arguments, reviewer, notes, confirmed)
        if project_status != "project_specific":
            human_projects = []
        human = {"status": project_status, "project_ids": sorted(set(human_projects))}
    elif task == "support_score":
        if not stance_status:
            return preserve_form(handle, state, "Choose a human stance status.", reviewable, project_status, human_projects, stance_status, human_score, human_arguments, reviewer, notes, confirmed)
        if stance_status == "scored" and human_score is None:
            return preserve_form(handle, state, "Choose a human support score from 0 to 5.", reviewable, project_status, human_projects, stance_status, human_score, human_arguments, reviewer, notes, confirmed)
        human = {"stance_status": stance_status, "support_score": int(human_score) if stance_status == "scored" else None}
    else:
        human = {"arguments": sorted(set(human_arguments or []))}
    verdict = compare_prediction(task, prediction, human)
    with sqlite3.connect(AUDIT_DB) as con:
        con.execute(
            """UPDATE audit_items SET human_label_json=?,reviewer=?,notes=?,reviewed_at=?
               WHERE batch_id=? AND ordinal=?""",
            (json.dumps(human, ensure_ascii=False), (reviewer or "").strip(), (notes or "").strip(), utc_now(), batch_id, row["ordinal"]),
        )
    state = dict(state or {})
    state["index"] = max(0, min(idx + int(step), len(rows)))
    verdict_name = {"correct": "完全一致", "partial": "部分重合", "wrong": "不一致", "not_evaluable": "不可评估"}.get(verdict, verdict)
    return render_item(handle, state, f"Saved. Model vs human label: **{verdict_name}**.")


def navigate(handle: str, state: dict[str, Any], step: int):
    state = dict(state or {})
    batch_id = str(state.get("batch_id") or "")
    rows = get_audit_rows(batch_id) if batch_id else []
    if not rows:
        return empty_view(state, "No active review batch.")
    state["index"] = max(0, min(int(state.get("index", 0)) + step, len(rows)))
    return render_item(handle, state)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def metric_summary(batch: dict[str, Any], reviewed: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    task = batch["task"]
    comparable: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for row in reviewed:
        prediction = json.loads(row["prediction_json"])
        human = json.loads(row["human_label_json"] or "{}")
        verdict = compare_prediction(task, prediction, human)
        if verdict != "not_evaluable":
            comparable.append((row, human, verdict))
    exact = sum(verdict == "correct" for _, _, verdict in comparable)
    metrics: dict[str, Any] = {
        "task": task,
        "sampled_n": int(batch["sampled_n"]),
        "population_n": int(batch["population_n"]),
        "reviewed_n": len(reviewed),
        "evaluable_n": len(comparable),
        "not_evaluable_n": len(reviewed) - len(comparable),
        "exact_matches": exact,
        "exact_match_rate": exact / len(comparable) if comparable else None,
        "wilson_95": wilson_interval(exact, len(comparable)),
        "verdict_counts": dict(Counter(verdict for _, _, verdict in comparable)),
        "include_training": bool(batch["include_training"]),
    }
    details = []
    if task == "support_score":
        scored = []
        status_correct = 0
        for row in reviewed:
            human = json.loads(row["human_label_json"] or "{}")
            if human.get("cannot_determine") or human.get("stance_status") == "unclear":
                continue
            status_correct += human.get("stance_status") == "scored"
            if human.get("stance_status") == "scored" and human.get("support_score") is not None:
                pred_score = json.loads(row["prediction_json"]).get("support_score_0_5")
                if pred_score is not None:
                    scored.append((int(human["support_score"]), int(pred_score)))
        metrics["status_accuracy"] = status_correct / len(comparable) if comparable else None
        if scored:
            abs_errors = [abs(gold - pred) for gold, pred in scored]
            metrics["scored_n"] = len(scored)
            metrics["score_exact_accuracy"] = sum(error == 0 for error in abs_errors) / len(scored)
            metrics["mae"] = statistics.mean(abs_errors)
            metrics["within_one"] = sum(error <= 1 for error in abs_errors) / len(scored)
            if len({g for g, _ in scored}) > 1 and len({p for _, p in scored}) > 1:
                try:
                    from sklearn.metrics import cohen_kappa_score
                    metrics["quadratic_weighted_kappa"] = float(cohen_kappa_score([g for g, _ in scored], [p for _, p in scored], weights="quadratic"))
                except Exception:
                    metrics["quadratic_weighted_kappa"] = None
            matrix = [[sum(g == i and p == j for g, p in scored) for j in range(6)] for i in range(6)]
            metrics["confusion_matrix_human_rows_model_columns"] = matrix
            details.extend([
                f"- Human-scored rows: **{len(scored)}**",
                f"- Exact score accuracy: **{metrics['score_exact_accuracy']:.1%}**",
                f"- Mean absolute error: **{metrics['mae']:.3f}**",
                f"- Within one score: **{metrics['within_one']:.1%}**",
                f"- Quadratic weighted kappa: **{metrics.get('quadratic_weighted_kappa') if metrics.get('quadratic_weighted_kappa') is not None else 'n/a'}**",
                "",
                "Confusion matrix (human score rows; model score columns):",
                "```text", "      " + " ".join(str(i) for i in range(6)),
                *[f"  {i}:  " + " ".join(str(x) for x in matrix[i]) for i in range(6)], "```",
            ])
        details.append(f"- Human stance-status accuracy (model always emits a numeric score): **{metrics['status_accuracy']:.1%}**" if metrics["status_accuracy"] is not None else "- Human stance-status accuracy: n/a")
    elif task == "project_index":
        tp = fp = fn = 0
        for row, human, _ in comparable:
            prediction = json.loads(row["prediction_json"])
            predicted = {str(c.get("project_id")) for c in prediction.get("project_candidates", []) if c.get("project_id")}
            gold = set(human.get("project_ids", [])) if human.get("status") == "project_specific" else set()
            tp += len(predicted & gold); fp += len(predicted - gold); fn += len(gold - predicted)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        metrics.update({"micro_tp": tp, "micro_fp": fp, "micro_fn": fn, "micro_precision": precision, "micro_recall": recall, "micro_f1": f1})
        details.extend([
            f"- Named-project micro precision: **{precision:.1%}**" if precision is not None else "- Named-project micro precision: n/a",
            f"- Named-project micro recall: **{recall:.1%}**" if recall is not None else "- Named-project micro recall: n/a",
            f"- Named-project micro F1: **{f1:.1%}**" if f1 is not None else "- Named-project micro F1: n/a",
        ])
    else:
        tp = fp = fn = 0
        for row, human, _ in comparable:
            prediction = json.loads(row["prediction_json"])
            predicted = {f"{a.get('class_code')}::{a.get('direction')}" for a in prediction.get("arguments", []) if a.get("class_code")}
            gold = set(human.get("arguments", []))
            tp += len(predicted & gold); fp += len(predicted - gold); fn += len(gold - predicted)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        metrics.update({"micro_tp": tp, "micro_fp": fp, "micro_fn": fn, "micro_precision": precision, "micro_recall": recall, "micro_f1": f1})
        details.extend([
            f"- Argument-direction micro precision: **{precision:.1%}**" if precision is not None else "- Argument-direction micro precision: n/a",
            f"- Argument-direction micro recall: **{recall:.1%}**" if recall is not None else "- Argument-direction micro recall: n/a",
            f"- Argument-direction micro F1: **{f1:.1%}**" if f1 is not None else "- Argument-direction micro F1: n/a",
        ])
    lohi = metrics["wilson_95"]
    ci_text = f"95% Wilson interval: {lohi[0]:.1%}–{lohi[1]:.1%}" if lohi else "95% Wilson interval: n/a"
    headline = (
        f"### {TASK_NAMES.get(task, task)} validation sample\n\n"
        f"- Random sample: **{batch['sampled_n']:,}** from population **{batch['population_n']:,}**; seed `{batch['seed']}`.\n"
        f"- Reviewed: **{len(reviewed):,}**; evaluable: **{len(comparable):,}**; cannot determine: **{len(reviewed)-len(comparable):,}**.\n"
        f"- Exact match: **{exact}/{len(comparable)}** ({exact/len(comparable):.1%})" if comparable else
        f"### {TASK_NAMES.get(task, task)} validation sample\n\n- No evaluable reviews yet."
    )
    if comparable:
        headline += f" · {ci_text}.\n"
    headline += "\n" + "\n".join(details)
    if batch["include_training"]:
        headline += "\n\n**Diagnostic only:** this batch allowed training or previously annotated examples; do not report it as independent validation."
    else:
        headline += "\n\nThis batch excluded predictions marked as training examples and messages already present in the selected SQLite annotations."
    if len(reviewed) < int(batch["sampled_n"]):
        headline += "\n\nOnly completed reviews are included; skipped or unfinished records can bias the sample if review completion is selective."
    metrics["summary_markdown"] = headline
    return metrics, headline


def export_report(state: dict[str, Any]):
    batch_id = str((state or {}).get("batch_id") or "")
    batch = get_batch(batch_id) if batch_id else {}
    rows = get_audit_rows(batch_id) if batch_id else []
    if not batch or not rows:
        return "Create or resume a sample batch first.", None, None
    reviewed = [r for r in rows if r.get("reviewed_at")]
    metrics, summary = metric_summary(batch, reviewed)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"validation_{batch_id}"
    md_path = EXPORT_DIR / f"{stem}.md"
    csv_path = EXPORT_DIR / f"{stem}_rows.csv"
    task_name = TASK_NAMES.get(batch["task"], batch["task"])
    warning = "TRAINING EXAMPLES INCLUDED — diagnostic only, not independent validation." if batch["include_training"] else "Training examples and previously annotated message IDs were excluded from the sampling pool."
    report = (
        f"# Human validation report: {task_name}\n\n"
        f"Generated: {utc_now()}\n\n"
        f"- Batch ID: `{batch_id}`\n- Model version: `{batch['model_version']}`\n"
        f"- Predictions SQLite: `{batch['prediction_db_name']}`\n- Prediction SQLite SHA-256: `{batch['prediction_db_sha256']}`\n"
        f"- Source data: `{batch['source_name']}`\n- Source SHA-256: `{batch['source_sha256']}`\n"
        f"- Sampling: simple random sample without replacement; unit `{('message' if batch['task'] == 'project_index' else 'message × project')}`; seed `{batch['seed']}`\n"
        f"- Blind review: {'on' if batch['blind_mode'] else 'off'}\n"
        f"- Population size: {batch['population_n']}\n- Sample size: {batch['sampled_n']}\n"
        f"- Completed reviews: {len(reviewed)}\n\n**Scope note:** {warning}\n\n"
        f"{summary}\n\n"
        "## Interpretation limits\n\n"
        "This report measures agreement with the human labels entered in this batch. It estimates performance only for this prediction pool and sampling frame. If only some sampled rows were completed, review non-response may bias the estimate. A confidence interval summarizes sampling uncertainty; it does not correct systematic label or sampling bias. Source sentiment is context only and is not used as a gold label.\n"
    )
    md_path.write_text(report, encoding="utf-8")
    fields = ["batch_id", "ordinal", "mention_id", "task", "project_id", "title", "content", "published_at", "language_code", "media_type_name", "source_name", "url", "source_sentiment", "prediction_json", "confidence", "review_required", "human_label_json", "verdict", "reviewer", "notes", "reviewed_at"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = json.loads(row["record_json"] or "{}")
            prediction = json.loads(row["prediction_json"] or "{}")
            human = json.loads(row["human_label_json"]) if row.get("human_label_json") else {}
            writer.writerow({
                "batch_id": batch_id, "ordinal": row["ordinal"], "mention_id": row["mention_id"],
                "task": row["task"], "project_id": row["project_id"], "title": record.get("title", ""),
                "content": record.get("content", ""), "published_at": record.get("published_at", ""),
                "language_code": record.get("language_code", ""), "media_type_name": record.get("media_type_name", ""),
                "source_name": record.get("source_name", ""), "url": record.get("url", ""),
                "source_sentiment": record.get("sentiment", ""),
                "prediction_json": json.dumps(prediction, ensure_ascii=False), "confidence": row.get("confidence"),
                "review_required": row.get("review_required"), "human_label_json": json.dumps(human, ensure_ascii=False) if human else "",
                "verdict": compare_prediction(row["task"], prediction, human) if human else "not_reviewed",
                "reviewer": row.get("reviewer", ""), "notes": row.get("notes", ""), "reviewed_at": row.get("reviewed_at", ""),
            })
    return f"Exported {len(reviewed)}/{len(rows)} reviewed records.", str(md_path), str(csv_path)


SCORE_LABELS = [
    "Strongly oppose", "Oppose", "Lean against", "Lean in favor", "Support", "Strongly support",
]
QUICK_SCOREABLE = [
    ("Yes — there is a clear position to score", "yes"),
    ("No — neutral, mixed, no personal position, or wrong project", "no"),
    ("I cannot tell", "unclear"),
]
QUICK_VERDICTS = [
    ("Correct", "correct"), ("Partly correct", "partial"),
    ("Wrong", "wrong"), ("Cannot judge", "unclear"),
]


def prediction_json(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    try:
        return json.loads(row.get("result_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def render_prediction_summary(ctx: dict[str, Any], grouped: dict[str, list[dict[str, Any]]], mention_id: str, target_project: str) -> tuple[str, dict[str, Any]]:
    rows = grouped.get(mention_id, [])
    by_task = {row["task"]: row for row in rows if row["task"] == "project_index"}
    project_row = by_task.get("project_index")
    project_result = prediction_json(project_row)
    candidates = project_result.get("project_candidates", [])
    project_lines = []
    for candidate in candidates:
        name = candidate.get("project_name") or ctx.get("project_names", {}).get(candidate.get("project_id"), candidate.get("project_id", "Unknown project"))
        confidence = candidate.get("confidence")
        confidence_text = f"{float(confidence):.1%}" if confidence is not None else "not provided"
        suffix = " **← selected project for the score below**" if candidate.get("project_id") == target_project else ""
        project_lines.append(f"- **{name}** — model probability {confidence_text}{suffix}")
    if not project_lines:
        project_lines = ["- No project label was produced."]

    task_row = lambda task: next((r for r in rows if r["task"] == task and str(r.get("project_id") or "") == target_project), None)
    support_row = task_row("support_score")
    support_result = prediction_json(support_row)
    predicted_score = support_result.get("support_score_0_5")
    score_conf = (support_row or {}).get("confidence")
    score_lines = [f"- Predicted support score: **{predicted_score}/5 · {SCORE_LABELS[int(predicted_score)]}**" if predicted_score is not None else "- No support score prediction for the selected project."]
    if predicted_score is not None and score_conf is not None:
        score_lines.append(f"- Confidence for this predicted score: **{float(score_conf):.1%}**")
    dist = support_result.get("score_distribution") or []
    if len(dist) == 6:
        score_lines.append("- Score probabilities (0 → 5): " + " · ".join(f"{i}: {float(p):.1%}" for i, p in enumerate(dist)))

    arg_row = task_row("argument_types")
    arg_result = prediction_json(arg_row)
    argument_lines = []
    names = {code: name for name, code in ARGUMENTS}
    direction_names = {code: name for name, code in DIRECTIONS}
    for argument in arg_result.get("arguments", []):
        label = names.get(argument.get("class_code"), argument.get("class_name") or argument.get("class_code") or "Other")
        direction = direction_names.get(argument.get("direction"), argument.get("direction") or "unclear")
        confidence = argument.get("confidence")
        confidence_text = f"{float(confidence):.1%}" if confidence is not None else "not provided"
        argument_lines.append(f"- **{label}** · {direction} — confidence {confidence_text}")
    if not argument_lines:
        argument_lines = ["- No argument labels passed the model's threshold."]
    if arg_row and arg_row.get("confidence") is not None:
        argument_lines.append(f"- Highest argument probability (diagnostic): {float(arg_row['confidence']):.1%}; this is not confidence that the empty label set is correct.")

    model_version = sorted({str(r.get("model_version") or "") for r in rows if r.get("model_version")})
    lines = [
        "### Model predictions",
        f"**Model:** {', '.join(model_version) or 'unknown'} · reported probabilities are raw and not calibrated.",
        "",
        "**Project classification**", *project_lines,
        "",
        f"**Support score for {ctx.get('project_names', {}).get(target_project, target_project)}**", *score_lines,
        "",
        "**Argument labels for the same project**", *argument_lines,
    ]
    snapshot = {
        "project_index": {"prediction": project_result, "confidence": (project_row or {}).get("confidence")},
        "support_score": {"prediction": support_result, "confidence": score_conf},
        "argument_types": {"prediction": arg_result, "confidence": (arg_row or {}).get("confidence")},
        "target_project_id": target_project,
        "model_version": ", ".join(model_version) or "unknown",
    }
    return "\n".join(lines), snapshot


def quick_context(handle: str) -> tuple[dict[str, Any] | None, dict[str, list[dict[str, Any]]]]:
    ctx = DATA_CACHE.get(handle)
    if not ctx:
        return None, {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ctx["predictions"]:
        grouped[str(row["mention_id"])].append(row)
    return ctx, grouped


def quick_candidates(ctx: dict[str, Any], grouped: dict[str, list[dict[str, Any]]]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for mention_id, rows in grouped.items():
        record = ctx["messages"].get(mention_id)
        if not record or mention_id in ctx["annotation_by_id"]:
            continue
        project_row = next((r for r in rows if r["task"] == "project_index"), None)
        project_result = prediction_json(project_row)
        if any(bool(prediction_json(r).get("is_training_example")) for r in rows):
            continue
        project_candidates = sorted(project_result.get("project_candidates", []), key=lambda c: float(c.get("confidence") or 0), reverse=True)
        target_project = str(project_candidates[0].get("project_id") or "") if project_candidates else ""
        support_row = next((r for r in rows if r["task"] == "support_score" and str(r.get("project_id") or "") == target_project), None)
        if support_row:
            candidates.append((mention_id, target_project))
    return candidates


def quick_reviewed_keys(prediction_sha: str) -> set[tuple[str, str]]:
    with sqlite3.connect(AUDIT_DB) as con:
        rows = con.execute("SELECT mention_id, project_id FROM quick_reviews WHERE prediction_db_sha256=?", (prediction_sha,)).fetchall()
    return {(str(row[0]), str(row[1])) for row in rows}


def quick_render(handle: str, state: dict[str, Any] | None = None, message: str = ""):
    state = dict(state or {})
    state["handle"] = handle or state.get("handle", "")
    ctx, grouped = quick_context(state["handle"])
    if not ctx:
        return (state, state.get("summary", ""), "", "", "", "", None, None, "unclear", "unclear", "", "", message or "Load a prediction database first.")
    mention_id = str(state.get("mention_id") or "")
    target_project = str(state.get("project_id") or "")
    if not mention_id or mention_id not in ctx["messages"]:
        return (state, state.get("summary", ""), "", "", "", "", None, None, "unclear", "unclear", "", "", message or "Click the button below to draw a random prediction.")
    record = ctx["messages"][mention_id]
    prediction_markdown, snapshot = render_prediction_summary(ctx, grouped, mention_id, target_project)
    project_name = ctx.get("project_names", {}).get(target_project, target_project or "Unknown project")
    rows_for_id = grouped.get(mention_id, [])
    project_row = next((r for r in rows_for_id if r["task"] == "project_index"), None)
    score_row = next((r for r in rows_for_id if r["task"] == "support_score" and str(r.get("project_id") or "") == target_project), None)
    args_row = next((r for r in rows_for_id if r["task"] == "argument_types" and str(r.get("project_id") or "") == target_project), None)
    info = [
        f"**Message ID:** `{mention_id}`",
        f"**Project being scored:** {project_name}",
        f"**Date:** {record.get('published_at') or 'unknown'} · **Language:** {record.get('language_code') or 'unknown'} · **Source:** {record.get('source_name') or 'unknown'}",
        f"**Prediction review flags:** project={bool((project_row or {}).get('review_required'))}, score={bool((score_row or {}).get('review_required'))}, arguments={bool((args_row or {}).get('review_required'))}",
    ]
    saved = None
    with sqlite3.connect(AUDIT_DB) as con:
        con.row_factory = sqlite3.Row
        saved_row = con.execute("SELECT * FROM quick_reviews WHERE prediction_db_sha256=? AND mention_id=? AND project_id=?", (ctx["prediction_sha"], mention_id, target_project)).fetchone()
        saved = dict(saved_row) if saved_row else None
    human_score = saved.get("human_score") if saved else None
    return (
        state, state.get("summary", ""), "\n\n".join(info), record.get("title") or "(No title)",
        record.get("content") or "[No content text in source file]", prediction_markdown,
        gr.update(value=saved.get("scoreable") if saved else None),
        gr.update(value=human_score), gr.update(value=saved.get("project_verdict", "unclear") if saved else "unclear"),
        gr.update(value=saved.get("argument_verdict", "unclear") if saved else "unclear"),
        saved.get("reviewer", "") if saved else "", saved.get("notes", "") if saved else "",
        message or ("This message has a saved review. You can edit it and save again." if saved else "Read the text, enter your judgment, then save or draw the next random item."),
    )


def quick_load_and_pick(prediction_file: Any, source_file: Any, source_path_text: str):
    handle, summary, *_ = load_data(prediction_file, source_file, source_path_text)
    state = {"handle": handle, "summary": summary, "seen": []}
    if not handle:
        return quick_render("", state, summary)
    return quick_pick_next(handle, state, allow_reviewed=False, notice="Loaded files. Drawing one random message…")


def quick_pick_next(handle: str, state: dict[str, Any] | None, allow_reviewed: bool = False, notice: str = ""):
    state = dict(state or {})
    state["handle"] = handle or state.get("handle", "")
    ctx, grouped = quick_context(state["handle"])
    if not ctx:
        return quick_render(state["handle"], state, "Could not find the loaded database. Load the files first.")
    seen = {tuple(item) for item in state.get("seen", [])}
    current_key = (str(state.get("mention_id") or ""), str(state.get("project_id") or ""))
    if current_key[0]:
        seen.add(current_key)
    excluded = set() if allow_reviewed else quick_reviewed_keys(ctx["prediction_sha"])
    excluded |= seen
    eligible = [key for key in quick_candidates(ctx, grouped) if key not in excluded]
    if not eligible and allow_reviewed:
        eligible = quick_candidates(ctx, grouped)
    if not eligible:
        state.update({"mention_id": "", "project_id": "", "seen": [list(item) for item in seen]})
        return quick_render(state["handle"], state, "No unreviewed predictions remain for this database. Export the reviewed sample or allow repeats.")
    mention_id, project_id = random.SystemRandom().choice(eligible)
    state.update({"mention_id": mention_id, "project_id": project_id, "seen": [list(item) for item in seen]})
    return quick_render(state["handle"], state, notice)


def quick_save(state: dict[str, Any], scoreable: str, human_score: int | None,
               project_verdict: str, argument_verdict: str, reviewer: str, notes: str, advance: bool = False):
    state = dict(state or {})
    handle = str(state.get("handle") or "")
    ctx, grouped = quick_context(handle)
    mention_id = str(state.get("mention_id") or "")
    project_id = str(state.get("project_id") or "")
    if not ctx or not mention_id:
        return quick_render(handle, state, "No current prediction to save. Load the database and draw a random one first.")
    if scoreable not in {"yes", "no", "unclear"}:
        return quick_render(handle, state, "Choose whether this message can be scored.")
    if scoreable == "yes" and human_score is None:
        result = list(quick_render(handle, state, "You chose Yes. Please choose your human score from 0 to 5."))
        result[6] = gr.update(value=scoreable)
        result[7] = gr.update(value=None)
        result[8] = gr.update(value=project_verdict or "unclear")
        result[9] = gr.update(value=argument_verdict or "unclear")
        result[10] = reviewer or ""
        result[11] = notes or ""
        return tuple(result)
    prediction_markdown, predictions = render_prediction_summary(ctx, grouped, mention_id, project_id)
    record = ctx["messages"][mention_id]
    human_score_value = int(human_score) if scoreable == "yes" and human_score is not None else None
    with sqlite3.connect(AUDIT_DB) as con:
        con.execute(
            """INSERT INTO quick_reviews
               (prediction_db_sha256,source_sha256,mention_id,project_id,record_json,predictions_json,
                scoreable,human_score,project_verdict,argument_verdict,reviewer,notes,reviewed_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(prediction_db_sha256,mention_id,project_id) DO UPDATE SET
                source_sha256=excluded.source_sha256,record_json=excluded.record_json,
                predictions_json=excluded.predictions_json,scoreable=excluded.scoreable,
                human_score=excluded.human_score,project_verdict=excluded.project_verdict,
                argument_verdict=excluded.argument_verdict,reviewer=excluded.reviewer,
                notes=excluded.notes,reviewed_at=excluded.reviewed_at""",
            (ctx["prediction_sha"], ctx["source_sha"], mention_id, project_id,
             json.dumps(record, ensure_ascii=False), json.dumps(predictions, ensure_ascii=False),
             scoreable, human_score_value, project_verdict or "unclear", argument_verdict or "unclear",
             (reviewer or "").strip(), (notes or "").strip(), utc_now()),
        )
    if advance:
        state["seen"] = [list(item) for item in {tuple(x) for x in state.get("seen", [])} | {(mention_id, project_id)}]
        state["mention_id"], state["project_id"] = mention_id, project_id
        return quick_pick_next(handle, state, notice="Saved your review. Here is the next random message.")
    return quick_render(handle, state, "Saved your human review. You can edit it and save again, or draw the next random message.")


def quick_export(state: dict[str, Any] | None):
    handle = str((state or {}).get("handle") or "")
    ctx = DATA_CACHE.get(handle)
    if not ctx:
        return "Load a database first.", None, None
    with sqlite3.connect(AUDIT_DB) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(row) for row in con.execute("SELECT * FROM quick_reviews WHERE prediction_db_sha256=? ORDER BY reviewed_at", (ctx["prediction_sha"],))]
    if not rows:
        return "There are no saved reviews to export yet.", None, None
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"quick_validation_{stamp}"
    md_path, csv_path = EXPORT_DIR / f"{stem}.md", EXPORT_DIR / f"{stem}.csv"
    yes_rows = [row for row in rows if row["scoreable"] == "yes" and row["human_score"] is not None]
    score_errors = []
    if yes_rows:
        for row in yes_rows:
            predicted = json.loads(row["predictions_json"]).get("support_score", {}).get("prediction", {}).get("support_score_0_5")
            if predicted is not None:
                score_errors.append(abs(int(row["human_score"]) - int(predicted)))
    scoreable_counts = Counter(row["scoreable"] for row in rows)
    project_counts = Counter(row["project_verdict"] for row in rows)
    argument_counts = Counter(row["argument_verdict"] for row in rows)
    report = (
        "# Quick prediction review report\n\n"
        f"Generated: {utc_now()}\n\n"
        f"- Model: `{ctx['model_version']}`\n- Predictions SQLite: `{ctx['prediction_path'].name}`\n"
        f"- Predictions SHA-256: `{ctx['prediction_sha']}`\n- Source dataset: `{ctx['source_path'].name}`\n"
        f"- Reviews saved: **{len(rows)}**\n- Human scoreability: {dict(scoreable_counts)}\n"
        f"- Project verdicts: {dict(project_counts)}\n- Argument verdicts: {dict(argument_counts)}\n"
        f"- Rows human-marked scoreable with a score: **{len(yes_rows)}**\n"
    )
    if score_errors:
        exact = sum(error == 0 for error in score_errors) / len(score_errors)
        mae = statistics.mean(score_errors)
        within_one = sum(error <= 1 for error in score_errors) / len(score_errors)
        report += f"- Score exact match: **{exact:.1%}** · MAE: **{mae:.3f}** · within one point: **{within_one:.1%}**.\n"
    else:
        report += "- No scoreable reviews with both human and model scores yet.\n"
    report += "\nReported probabilities are raw model outputs, not calibrated confidence. This is a review sample, not a gold-standard independent benchmark.\n"
    md_path.write_text(report, encoding="utf-8")
    fields = ["mention_id", "project_id", "title", "content", "published_at", "source_name", "source_url", "model_version", "project_predictions_json", "support_prediction_json", "argument_predictions_json", "human_scoreable", "human_support_score", "model_support_score", "support_confidence", "project_verdict", "argument_verdict", "reviewer", "notes", "reviewed_at"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle_out:
        writer = csv.DictWriter(handle_out, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record, predictions = json.loads(row["record_json"]), json.loads(row["predictions_json"])
            support = predictions.get("support_score", {})
            support_pred = support.get("prediction", {})
            writer.writerow({
                "mention_id": row["mention_id"], "project_id": row["project_id"], "title": record.get("title", ""),
                "content": record.get("content", ""), "published_at": record.get("published_at", ""),
                "source_name": record.get("source_name", ""), "source_url": record.get("url", ""),
                "model_version": predictions.get("model_version", ""),
                "project_predictions_json": json.dumps(predictions.get("project_index", {}).get("prediction", {}), ensure_ascii=False),
                "support_prediction_json": json.dumps(support_pred, ensure_ascii=False),
                "argument_predictions_json": json.dumps(predictions.get("argument_types", {}).get("prediction", {}), ensure_ascii=False),
                "human_scoreable": row["scoreable"], "human_support_score": row["human_score"],
                "model_support_score": support_pred.get("support_score_0_5"), "support_confidence": support.get("confidence"),
                "project_verdict": row["project_verdict"], "argument_verdict": row["argument_verdict"],
                "reviewer": row["reviewer"], "notes": row["notes"], "reviewed_at": row["reviewed_at"],
            })
    return f"Exported {len(rows)} reviewed messages.", str(md_path), str(csv_path)


init_audit_db()
DEFAULT_PRED_FILE = str(DEFAULT_PREDICTIONS) if DEFAULT_PREDICTIONS.is_file() else None
DEFAULT_SOURCE_FILE = str(DEFAULT_SOURCE) if DEFAULT_SOURCE.is_file() else None


with gr.Blocks(title="Cable Car NLP · Quick Prediction Review") as demo:
    gr.Markdown(
        "# 缆车模型：随机抽查一条\n"
        "点一次按钮，页面会从预测数据库中抽一条**还没核验过**的消息，并合并显示项目、支持分数和论点预测。"
    )
    quick_state = gr.State({"handle": "", "summary": "", "seen": []})
    with gr.Accordion("数据库文件（默认已选最新的 Predictions SQLite 和本地原始数据）", open=False):
        with gr.Row():
            prediction_file = gr.File(label="预测结果 SQLite（可保留默认文件，也可换一个）", file_types=[".sqlite3", ".sqlite", ".db"], type="filepath", value=DEFAULT_PRED_FILE)
            source_file = gr.File(label="可选：上传原始 JSON / CSV / TSV", file_types=[".json", ".csv", ".tsv"], type="filepath")
        source_path = gr.Textbox(label="原始数据路径（SQLite 不含标题和正文）", value=DEFAULT_SOURCE_FILE or "")
    load_button = gr.Button("加载数据库并随机抽一条", variant="primary", size="lg")
    load_summary = gr.Markdown()
    record_info = gr.Markdown()
    title = gr.Textbox(label="标题 / 话题", interactive=False, lines=2)
    content = gr.Textbox(label="正文", interactive=False, lines=14, show_copy_button=True)
    predictions_view = gr.Markdown()
    gr.Markdown(
        "**分数说明：** 0 = 强烈反对，1 = 反对，2 = 倾向反对，3 = 倾向支持，4 = 支持，5 = 强烈支持。\n\n"
        "**注意：** 模型总会尝试给出一个分数；它当前没有单独判断“这条内容是否适合评分”的模型。下面的“能否评分”由你根据原文判断。置信度是模型的原始概率，尚未校准。"
    )
    with gr.Group():
        scoreable = gr.Radio(label="针对上方项目，这条内容能否给出明确的支持 / 反对分数？", choices=QUICK_SCOREABLE, value=None)
        human_score = gr.Dropdown(
            label="如果能评分，你的人工分数（0–5）",
            choices=[(f"{i} · {label}", i) for i, label in enumerate(SCORE_LABELS)],
            value=None,
        )
    with gr.Row():
        project_verdict = gr.Radio(label="项目预测对吗？", choices=QUICK_VERDICTS, value="unclear")
        argument_verdict = gr.Radio(label="论点预测对吗？", choices=QUICK_VERDICTS, value="unclear")
    reviewer = gr.Textbox(label="核验人（可选，名字或缩写）")
    notes = gr.Textbox(label="备注（可选）", lines=2)
    review_message = gr.Markdown()
    with gr.Row():
        save_button = gr.Button("保存本条", variant="primary")
        save_next_button = gr.Button("保存并抽下一条", variant="primary")
        next_button = gr.Button("跳过，抽下一条")
    allow_repeats = gr.Checkbox(label="没有未核验记录时，允许重新抽已核验记录", value=False)

    gr.Markdown("---\n## 导出核验结果")
    export_button = gr.Button("导出统计报告和逐条 CSV")
    export_status = gr.Markdown()
    with gr.Row():
        report_file = gr.File(label="统计报告（Markdown）")
        rows_file = gr.File(label="逐条核验数据（CSV）")

    quick_outputs = [quick_state, load_summary, record_info, title, content, predictions_view,
                     scoreable, human_score, project_verdict, argument_verdict, reviewer, notes, review_message]
    load_button.click(quick_load_and_pick, [prediction_file, source_file, source_path], quick_outputs)
    next_button.click(lambda s, repeat: quick_pick_next(str((s or {}).get("handle") or ""), s, allow_reviewed=repeat, notice="Skipped without saving."), [quick_state, allow_repeats], quick_outputs)
    save_inputs = [quick_state, scoreable, human_score, project_verdict, argument_verdict, reviewer, notes]
    save_button.click(lambda *args: quick_save(*args, advance=False), save_inputs, quick_outputs)
    save_next_button.click(lambda *args: quick_save(*args, advance=True), save_inputs, quick_outputs)
    export_button.click(quick_export, [quick_state], [export_status, report_file, rows_file])


if __name__ == "__main__":
    host = os.environ.get("AUDIT_HOST", "127.0.0.1")
    port = int(os.environ.get("AUDIT_PORT", "7862"))
    username = os.environ.get("AUDIT_USERNAME", "")
    password = os.environ.get("AUDIT_PASSWORD", "")
    if bool(username) != bool(password):
        raise RuntimeError("Set both AUDIT_USERNAME and AUDIT_PASSWORD, or neither.")
    auth = (username, password) if username and password else None
    if host not in {"127.0.0.1", "localhost"} and auth is None:
        raise RuntimeError("For a non-local bind, set AUDIT_USERNAME and AUDIT_PASSWORD and use HTTPS through a reverse proxy.")
    demo.queue().launch(server_name=host, server_port=port, share=False, auth=auth)
