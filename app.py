from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
from collections import Counter
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Event, Thread
from typing import Any

import gradio as gr
import pandas as pd

import store
from dashboard_charts import build_dashboard
from model_pipeline import infer_records, load_latest_models, pack_prediction_rows, save_models, train_models
from workflow import LEGACY, Workflow


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
EXPORT_DIR = APP_DIR / "data" / "exports"
SCORE_LABELS = [
    ("0 · Strongly oppose", 0),
    ("1 · Oppose", 1),
    ("2 · Lean against", 2),
    ("3 · Lean in favor", 3),
    ("4 · Support", 4),
    ("5 · Strongly support", 5),
]
STANCE_OPTIONS = [
    ("Explicit position · assign 0–5", "scored"),
    ("Neutral / no clear preference", "neutral"),
    ("Mixed or conditional position", "mixed_or_conditional"),
    ("Mentions the project, no personal position", "no_stance"),
    ("Cannot determine / needs review", "unclear"),
]
DIRECTION_OPTIONS = [
    ("Supports the project", "pro"),
    ("Opposes the project", "con"),
    ("Mixed / conditional", "mixed"),
    ("Neutral framing", "neutral"),
    ("Unclear", "unclear"),
]


def find_default_dataset() -> Path | None:
    configured = os.getenv("LABELER_SOURCE_PATH")
    if configured:
        return Path(configured).expanduser()
    folder = ROOT_DIR / "Participants - Future City Hackathon" / "Stadt Heilbronn"
    if not folder.exists():
        return None
    candidates = sorted(folder.glob("*.json"))
    preferred = [p for p in candidates if "FSI_Heilbronn" in p.name]
    return (preferred or candidates or [None])[0]


def normalize_records(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        for key in ("data", "records", "mentions", "items"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError("Expected a JSON array, or an object containing a data/records/items array.")
    output: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        base_id = str(item.get("mention_id") or item.get("id") or f"row_{i + 1}")
        seen[base_id] += 1
        mention_id = base_id if seen[base_id] == 1 else f"{base_id}#{seen[base_id]}"
        content = str(item.get("content") or item.get("text") or item.get("message") or "")
        title = str(item.get("title") or "")
        published = str(item.get("published_at") or item.get("date") or item.get("timestamp") or "")
        if not title and "\n\n" in content and item.get("timestamp"):
            title = content.split("\n\n", 1)[0]
        if not item.get("published_at") and item.get("timestamp"):
            try:
                published = parsedate_to_datetime(published).isoformat()
            except (ValueError, TypeError, OverflowError):
                pass
        output.append({
            "mention_id": mention_id,
            "source_mention_id": base_id,
            "title": title,
            "content": content,
            "published_at": published,
            "language_code": str(item.get("language_code") or item.get("language") or "unknown"),
            "media_type_name": str(item.get("media_type_name") or item.get("media_type") or "unknown"),
            "source_name": str(item.get("source_name") or item.get("source") or ""),
            "url": str(item.get("url") or ""),
            "country_code": str(item.get("country_code") or ""),
            "sentiment": item.get("sentiment"),
            "reach": item.get("reach"),
            "engagements": item.get("engagements"),
        })
    if not output:
        raise ValueError("No message records found.")
    return output


def parse_upload(filename: str, data: bytes) -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.casefold()
    if suffix == ".json":
        return normalize_records(json.loads(data.decode("utf-8-sig")))
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        return normalize_records(list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), delimiter=delimiter)))
    raise ValueError("Upload a JSON, CSV or TSV file.")


def fingerprint(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        for value in (row["mention_id"], row["url"], row["published_at"], row["title"], row["content"][:128]):
            digest.update(str(value).encode("utf-8", errors="ignore"))
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def model_key(dataset_key: str) -> str:
    return dataset_key[:16]


def project_choices() -> list[tuple[str, str]]:
    return [(p["project_name"], p["project_id"]) for p in store.list_projects()]


def argument_choices() -> list[tuple[str, str]]:
    return [(a["class_name"], a["class_code"]) for a in store.list_argument_classes()]


def as_content(record: dict[str, Any]) -> str:
    return record["content"] or "[No content text in source]"


def get_existing_message_annotations(dataset_key: str, mention_id: str) -> list[dict[str, Any]]:
    return store.get_annotations_for_message(dataset_key, mention_id)


def render_current(state: dict[str, Any] | None, notice: str = "", editing: str | None = None) -> tuple[Any, ...]:
    state = dict(state or {"ids": [], "idx": 0})
    ids = state.get("ids", [])
    if not ids:
        return (
            state, "No message in this queue", "", "", "0 / 0", gr.update(), "", "unclear", None, [], "unclear", "", "", "", False,
            {}, notice or "No messages match these filters.", gr.update(choices=[], value=None),
        )
    state["idx"] = max(0, min(int(state.get("idx", 0)), len(ids) - 1))
    record = RECORD_BY_ID[ids[state["idx"]]]
    message_id = record["mention_id"]
    saved = [row for row in get_existing_message_annotations(DATASET_KEY, message_id) if row["annotator"] != LEGACY]
    suggestion = WORKFLOW.keyword_for(message_id) or {}
    state["saved_projects"] = [row["project_id"] for row in saved]
    editing = editing or (saved[0]["project_id"] if saved else "__new__")
    annotation = next((row for row in saved if row["project_id"] == editing), None)
    use_suggestion = not saved and not annotation
    project = annotation["project_id"] if annotation else (suggestion.get("project_id", "__unclear__") if use_suggestion else "__unclear__")
    saved_args = (store.get_arguments_for_annotation(DATASET_KEY, message_id, project) if annotation
                  else suggestion.get("arguments", []) if use_suggestion else [])
    state["original_project_id"] = annotation["project_id"] if annotation else None
    state["original_updated_at"] = annotation["updated_at"] if annotation else None
    state["original_arguments"] = saved_args
    annotation = annotation or (suggestion if use_suggestion else {})
    stance_status = annotation["stance_status"] if annotation else "unclear"
    support_score = annotation.get("support_score")
    notes = annotation.get("notes", "") if not use_suggestion else ""
    needs_review = bool(annotation.get("needs_review")) if not use_suggestion else False
    classes = [a["class_code"] for a in saved_args]
    direction = saved_args[0]["direction"] if saved_args else "unclear"
    quote = saved_args[0]["evidence_quote"] if saved_args else ""
    abstract = saved_args[0]["argument_summary"] if saved_args else ""
    meta_parts = [record["published_at"], record["language_code"], record["media_type_name"], record["source_name"]]
    metadata = " · ".join(x for x in meta_parts if x) + (f"\n\nSource: {record['url']}" if record["url"] else "")
    progress = f"Message {state['idx'] + 1:,} / {len(ids):,} · ID `{message_id}`"
    existing_str = "; ".join(a["project_name"] for a in saved)
    display_notice = notice or (f"Already labeled for: {existing_str}" if existing_str else "Unlabeled message")
    return (
        state, record["title"] or "(No title)", as_content(record), metadata, progress,
        gr.Dropdown(choices=project_choices(), value=project), "", stance_status, support_score,
        classes, direction, quote, abstract, notes, needs_review,
        {"saved_project_annotations": saved, "keyword_suggestions": suggestion},
        display_notice, gr.update(choices=[(row["project_name"], row["project_id"]) for row in saved]
                                  + [("Add a project label", "__new__")], value=editing),
    )


def build_queue(queue_name: str, language: str, media_type: str, query: str, project_filter: str) -> tuple[Any, ...]:
    labeled_ids, review_ids = WORKFLOW.queue_ids()
    if queue_name == "Unlabeled":
        candidates = [r for r in RECORDS if r["mention_id"] not in labeled_ids and r["mention_id"] not in review_ids]
    elif queue_name == "Needs review":
        candidates = [r for r in RECORDS if r["mention_id"] in review_ids]
        candidates.sort(key=lambda r: "import_file" not in r)
    elif queue_name == "Labeled":
        candidates = [r for r in RECORDS if r["mention_id"] in labeled_ids]
    else:
        candidates = list(RECORDS)
    if language == "French (all variants)":
        candidates = [r for r in candidates if str(r["language_code"]).casefold().startswith("fr")]
    elif language == "German (all variants)":
        candidates = [r for r in candidates if str(r["language_code"]).casefold().startswith("de")]
    elif language != "All languages":
        candidates = [r for r in candidates if r["language_code"] == language]
    if media_type != "All media types":
        candidates = [r for r in candidates if r["media_type_name"] == media_type]
    if project_filter != "All projects":
        project_id = next((p["project_id"] for p in store.list_projects() if p["project_name"] == project_filter), "")
        assigned = WORKFLOW.project_mentions(project_id)
        candidates = [r for r in candidates if r["mention_id"] in assigned]
    query = query.casefold().strip()
    if query:
        candidates = [r for r in candidates if query in (r["title"] + " " + r["content"]).casefold()]
    state = {"ids": [r["mention_id"] for r in candidates], "idx": 0}
    if not candidates:
        return render_current(state, "No messages match. Change filters or search terms.")
    return render_current(state, f"Loaded {len(candidates):,} messages into the queue.")


def move(state: dict[str, Any], step: int) -> tuple[Any, ...]:
    state = state or {"ids": [], "idx": 0}
    state["idx"] = max(0, min(int(state.get("idx", 0)) + step, max(0, len(state.get("ids", [])) - 1)))
    return render_current(state)


def save_current(
    state: dict[str, Any], project_id: str, other_project_name: str, stance_status: str,
    support_score: int | None, argument_types: list[str] | None, direction: str,
    evidence_quote: str, argument_summary: str, annotator: str, notes: str,
    needs_review: bool, advance: bool,
) -> tuple[Any, ...]:
    if not state or not state.get("ids"):
        return render_current(state, "No message selected.")
    record = RECORD_BY_ID[state["ids"][state["idx"]]]
    saved_project_id = project_id
    actual_project_name = ""
    if project_id == "__other_named__":
        if not (other_project_name or "").strip():
            raise gr.Error("Enter the new project's name before saving.")
        saved_project_id = store.add_project(other_project_name.strip())
        actual_project_name = other_project_name.strip()
    if saved_project_id == "__general__":
        relevance = "general"
    elif saved_project_id == "__unrelated__":
        relevance = "unrelated"
    elif saved_project_id == "__unclear__":
        relevance = "unclear"
    else:
        relevance = "project_specific"
    score = int(support_score) if stance_status == "scored" and support_score is not None else None
    if stance_status == "scored" and score is None:
        raise gr.Error("Choose a support score, or select a status without a score.")
    original_args = {arg["class_code"]: arg for arg in state.get("original_arguments", [])}
    first = next(iter(original_args.values()), {})
    unchanged = (direction, evidence_quote or "", argument_summary or "") == (
        first.get("direction", "unclear"), first.get("evidence_quote", ""), first.get("argument_summary", ""),
    )
    arguments = []
    for code in argument_types or []:
        if unchanged and code in original_args:
            arguments.append(original_args[code])
            continue
        arguments.append({
            "class_code": code,
            "direction": direction,
            "evidence_quote": evidence_quote or "",
            "argument_summary": argument_summary or "",
        })
    try:
        store.save_annotation({
            "dataset_key": DATASET_KEY,
            "mention_id": record["mention_id"],
            "project_id": saved_project_id,
            "relevance": relevance,
            "stance_status": stance_status,
            "support_score": score,
            "other_project_name": actual_project_name or None,
            "annotator": (annotator or "").strip(),
            "notes": notes or "",
            "needs_review": bool(needs_review),
            "replace_rule_seed": True,
            "original_project_id": state.get("original_project_id"),
            "expected_updated_at": state.get("original_updated_at"),
        }, arguments)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    if actual_project_name:
        PROJECT_NAMES[saved_project_id] = actual_project_name
    project_name = next((p["project_name"] for p in store.list_projects() if p["project_id"] == saved_project_id), saved_project_id)
    notice = f"Saved {record['mention_id']} → {project_name}."
    if advance and state["idx"] < len(state["ids"]) - 1:
        state["idx"] += 1
        notice = "Saved; moved to the next message."
    rendered = list(render_current(state, notice, None if advance else saved_project_id))
    return tuple(rendered)


def train_and_save() -> tuple[str, dict[str, Any]]:
    annotations, arguments = store.get_training_data(DATASET_KEY)
    models, summary = train_models(RECORDS, annotations, arguments, PROJECT_NAMES)
    if not models:
        return "No trainable task yet. Add project labels, per-class examples, and at least two support-score levels.", summary
    path, version = save_models(models, DATASET_KEY, summary)
    store.save_model_run(DATASET_KEY, version, path, json.dumps(summary, ensure_ascii=False))
    return f"Saved model bundle `{version}` to `{path}`. Training success is not an accuracy score; evaluate on held-out human labels.", summary


def predict_unlabeled(project_threshold: float, argument_threshold: float, support_threshold: float) -> tuple[str, dict[str, Any]]:
    bundle = load_latest_models(DATASET_KEY)
    if not bundle:
        return "Train a model before predicting.", {}
    if "project" not in bundle["models"]:
        return (
            "The Project Index model is not trainable yet. Label at least two examples for each named project and add human-confirmed general/unrelated examples or examples of other named projects as negatives, then train again.",
            bundle.get("summary", {}),
        )
    labeled_ids, _, _ = store.annotation_overview(DATASET_KEY)
    remaining = [r for r in RECORDS if r["mention_id"] not in labeled_ids]
    if not remaining:
        return "All messages already have at least one human project annotation.", {"predictions": 0}
    predictions = infer_records(
        remaining, bundle["models"], PROJECT_NAMES,
        project_threshold=project_threshold,
        argument_threshold=argument_threshold,
        support_review_threshold=support_threshold,
    )
    rows = pack_prediction_rows(DATASET_KEY, bundle["version"], predictions)
    store.save_predictions(DATASET_KEY, bundle["version"], rows)
    return f"Saved {len(rows):,} task suggestions for {len(remaining):,} unlabeled messages. Predictions remain separate from human labels.", {
        "messages": len(remaining), "suggestion_rows": len(rows), "model_version": bundle["version"],
    }


def current_progress() -> str:
    labeled, needs_review, annotation_count = store.annotation_overview(DATASET_KEY)
    annotations, args = store.get_training_data(DATASET_KEY)
    predictions = store.prediction_rows(DATASET_KEY)
    confirmed_ids = {a["mention_id"] for a in annotations}
    project_counts = Counter(a["project_id"] for a in annotations)
    score_counts = Counter(str(a["support_score"]) for a in annotations if a["support_score"] is not None)
    return (
        f"### Progress\n"
        f"- Messages with any label, including provisional labels: **{len(labeled):,} / {len(RECORDS):,}**\n"
        f"- Confirmed messages eligible for training: **{len(confirmed_ids):,}**\n"
        f"- All message × project rows: **{annotation_count:,}** · **{len(needs_review):,}** messages pending review\n"
        f"- Confirmed argument rows eligible for training: **{len(args):,}**\n"
        f"- Model suggestion records: **{len(predictions):,}**\n"
        f"- Score counts: `{dict(score_counts)}`\n"
        f"- Project counts: `{dict(project_counts)}`"
    )


def _records_for_jsonl(labels: list[dict[str, Any]], args: list[dict[str, Any]]) -> bytes:
    args_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in args:
        args_by_key.setdefault((row["mention_id"], row["project_id"]), []).append({
            "class_code": row["class_code"], "class_name": row["class_name"],
            "direction": row["direction"], "evidence_quote": row["evidence_quote"],
            "argument_summary": row["argument_summary"],
        })
    lines = []
    for row in labels:
        record = RECORD_BY_ID.get(row["mention_id"], {})
        item = {
            "chunk_id": f"{row['mention_id']}::{row['project_id']}",
            "mention_id": row["mention_id"], "project_id": row["project_id"], "project_name": row["project_name"],
            "title": record.get("title", ""), "text": record.get("content", ""),
            "language": record.get("language_code", "unknown"), "published_at": record.get("published_at", ""),
            "source_name": record.get("source_name", ""), "source_url": record.get("url", ""),
            "media_type": record.get("media_type_name", "unknown"),
            "relevance": row["relevance"], "stance_status": row["stance_status"],
            "support_score_0_5": row["support_score"],
            "arguments": args_by_key.get((row["mention_id"], row["project_id"]), []),
            "review_status": "needs_review" if row["needs_review"] else "human_saved",
            "annotator": row["annotator"], "updated_at": row["updated_at"],
        }
        lines.append(json.dumps(item, ensure_ascii=False))
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def export_file(kind: str) -> str:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labels, args = store.export_rows(DATASET_KEY)
    if kind == "labels_csv":
        rows = []
        for label in labels:
            meta = RECORD_BY_ID.get(label["mention_id"], {})
            rows.append({**label, "title": meta.get("title", ""), "content": meta.get("content", ""),
                         "language_code": meta.get("language_code", "unknown"), "published_at": meta.get("published_at", ""),
                         "media_type_name": meta.get("media_type_name", "unknown"), "source_name": meta.get("source_name", ""),
                         "url": meta.get("url", "")})
        path = EXPORT_DIR / f"message_project_labels_{stamp}.csv"
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    elif kind == "arguments_csv":
        path = EXPORT_DIR / f"message_arguments_{stamp}.csv"
        pd.DataFrame(args).to_csv(path, index=False, encoding="utf-8-sig")
    elif kind == "rag_jsonl":
        path = EXPORT_DIR / f"annotations_rag_{stamp}.jsonl"
        path.write_bytes(_records_for_jsonl(labels, args))
    elif kind == "predictions_csv":
        rows = []
        for row in store.prediction_rows(DATASET_KEY):
            rows.append({**row, "result": row.pop("result_json")})
        path = EXPORT_DIR / f"model_suggestions_{stamp}.csv"
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    elif kind == "sqlite_backup":
        path = EXPORT_DIR / f"annotations_backup_{stamp}.sqlite3"
        store.backup_database(path)
    else:
        raise ValueError(kind)
    return str(path)


def add_argument_category(name: str) -> tuple[str, Any]:
    if not (name or "").strip():
        return "Enter a category name first.", gr.Dropdown(choices=argument_choices())
    code = store.add_argument_class(name.strip())
    return f"Added `{name.strip()}` (`{code}`).", gr.Dropdown(choices=argument_choices(), value=[code])


def add_project_category(name: str) -> tuple[str, Any]:
    if not (name or "").strip():
        return "Enter a project name first.", gr.Dropdown(choices=project_choices())
    project_id = store.add_project(name.strip())
    PROJECT_NAMES[project_id] = name.strip()
    return f"Added `{name.strip()}` (`{project_id}`).", gr.Dropdown(choices=project_choices(), value=project_id)


def export_progress() -> str:
    return current_progress()


def dashboard_projects():
    return [("All topics", "__all__")] + [
        (project["project_name"], project["project_id"])
        for project in store.list_projects() if not project["is_special"]
    ]


def refresh_dashboard(project_id, provenance, granularity, previous=None):
    with WORKFLOW.lock:
        revision = (*WORKFLOW.revision(), project_id, provenance, granularity)
        if previous and tuple(previous) == revision:
            return (gr.skip(),) * 7
        labels, arguments = WORKFLOW.dashboard_rows()
        _, pending = WORKFLOW.queue_ids()
        figures = build_dashboard(RECORDS, labels, arguments, store.list_projects(),
                                  store.list_argument_classes(), project_id, provenance, granularity)
        coverage = figures[0] + f"\n\n**{len(pending):,}** messages in the manual review queue."
    return (revision, gr.update(choices=dashboard_projects()), coverage, *figures[1:])


def sync_inputs(upload=None):
    added, duplicates, errors = WORKFLOW.sync([upload] if upload else ())
    status = f"Added **{added:,}** messages. Skipped **{duplicates:,}** duplicates. **{len(RECORDS):,}** messages available."
    if errors:
        status += "\n\n" + "\n".join(f"- {error}" for error in errors)
    elif not upload and not any(path.is_file() for path in WORKFLOW._paths()):
        status += "\n\nWaiting for a crawler JSON file. You can also upload one here."
    return status, gr.update(choices=language_choices()), gr.update(choices=media_choices())


def language_choices():
    return ["All languages", "French (all variants)", "German (all variants)"] + sorted({r["language_code"] for r in RECORDS})


def media_choices():
    return ["All media types"] + sorted({r["media_type_name"] for r in RECORDS})


store.initialize()
DEFAULT_DATASET = find_default_dataset()
if DEFAULT_DATASET is None:
    raise FileNotFoundError("Default dataset not found; place the app under the FCH26 project or edit DEFAULT_DATASET in app.py.")
with open(DEFAULT_DATASET, "r", encoding="utf-8-sig") as _handle:
    RECORDS = normalize_records(json.load(_handle))
DATASET_KEY = fingerprint(RECORDS)
WORKFLOW = Workflow(DATASET_KEY, RECORDS, parse_upload, ROOT_DIR)
WORKFLOW.sync()
RECORD_BY_ID = WORKFLOW.by_id
PROJECT_NAMES = {p["project_id"]: p["project_name"] for p in store.list_projects()}
QUEUE_NAMES = ["Needs review", "Unlabeled", "Labeled", "All"]
PROJECT_FILTERS = ["All projects"] + [p["project_name"] for p in store.list_projects() if not p["is_special"]]


with gr.Blocks(title="Cable Car Debate Atlas", theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown(
        "# Cable Car Debate Atlas\n"
        "Explore how projects are discussed in the source dataset and incoming coverage."
    )
    with gr.Tabs() as tabs:
        with gr.Tab("Overview", id="overview"):
            coverage = gr.Markdown("Loading labels from SQLite…")
            with gr.Row():
                chart_project = gr.Dropdown(dashboard_projects(), value="__all__", label="Topic for arguments and scores")
                chart_origin = gr.Dropdown([("Human and keyword", "all"), ("Human confirmed", "human"),
                                            ("Keyword only", "keyword")], value="all", label="Label source")
                chart_period = gr.Radio([("Month", "month"), ("Quarter", "quarter")], value="month", label="Time interval")
                open_queue = gr.Button("Open manual review", variant="primary")
            with gr.Row():
                topic_plot = gr.Plot()
                argument_plot = gr.Plot()
            time_plot = gr.Plot()
            source_plot = gr.Plot()

        with gr.Tab("Manual labeling", id="manual") as manual_tab:
            queue_state = gr.State({"ids": [], "idx": 0})
            with gr.Row():
                queue = gr.Dropdown(choices=QUEUE_NAMES, value="Needs review", label="Queue")
                language = gr.Dropdown(choices=language_choices(), value="All languages", label="Language")
                media = gr.Dropdown(choices=media_choices(), value="All media types", label="Media type")
                project_filter = gr.Dropdown(choices=PROJECT_FILTERS, value="All projects", label="Project filter")
            with gr.Row():
                search = gr.Textbox(label="Search title + content", placeholder="Search a phrase, city or project")
                load_button = gr.Button("Load / refresh queue", variant="primary")
            with gr.Row():
                previous_button = gr.Button("← Previous")
                next_button = gr.Button("Next →")
                position = gr.Markdown("Message queue has not loaded yet.")
            notice = gr.Markdown("")
            with gr.Row():
                with gr.Column(scale=5):
                    title = gr.Textbox(label="Title", interactive=False, lines=2)
                    metadata = gr.Markdown()
                    content = gr.Textbox(label="Content", interactive=False, lines=20, show_copy_button=True)
                    with gr.Accordion("Keyword clues and saved labels", open=False):
                        suggestions = gr.JSON(value={})
                with gr.Column(scale=3):
                    annotator = gr.Textbox(label="Annotator initials / name")
                    edit_project = gr.Dropdown(choices=[], label="Saved label to edit")
                    project = gr.Dropdown(choices=project_choices(), value="__unclear__", label="Project Index")
                    other_project_name = gr.Textbox(label="New project name (only for ‘Other named project’)")
                    stance_status = gr.Radio(choices=STANCE_OPTIONS, value="unclear", label="Support Score status")
                    support_score = gr.Radio(choices=SCORE_LABELS, value=None, label="Support Score · 0 = strongly oppose, 5 = strongly support")
                    argument_types = gr.Dropdown(choices=argument_choices(), multiselect=True, label="Argument Type(s) · multi-select")
                    direction = gr.Dropdown(choices=DIRECTION_OPTIONS, value="unclear", label="Direction for selected argument type(s)")
                    evidence_quote = gr.Textbox(label="Evidence quote (short exact excerpt)", lines=3)
                    argument_summary = gr.Textbox(label="Short faithful argument abstraction (optional)", lines=2)
                    notes = gr.Textbox(label="Annotator notes", lines=2)
                    needs_review = gr.Checkbox(label="Flag for second human review", value=False)
                    with gr.Row():
                        save_button = gr.Button("Save", variant="primary")
                        save_next_button = gr.Button("Save & next")
            view_outputs = [queue_state, title, content, metadata, position, project, other_project_name,
                            stance_status, support_score, argument_types, direction, evidence_quote,
                            argument_summary, notes, needs_review, suggestions, notice, edit_project]
            load_button.click(build_queue, [queue, language, media, search, project_filter], view_outputs)
            previous_button.click(lambda s: move(s, -1), [queue_state], view_outputs)
            next_button.click(lambda s: move(s, 1), [queue_state], view_outputs)
            edit_project.input(lambda s, editing: render_current(s, editing=editing), [queue_state, edit_project], view_outputs)
            save_inputs = [queue_state, project, other_project_name, stance_status, support_score,
                           argument_types, direction, evidence_quote, argument_summary, annotator, notes, needs_review]
            save_event = save_button.click(lambda *xs: save_current(*xs, advance=False), save_inputs, view_outputs)
            save_next_event = save_next_button.click(lambda *xs: save_current(*xs, advance=True), save_inputs, view_outputs)
            open_queue.click(lambda: gr.Tabs(selected="manual"), outputs=tabs).then(
                build_queue, [queue, language, media, search, project_filter], view_outputs,
            )
            manual_tab.select(lambda: (gr.update(choices=language_choices()), gr.update(choices=media_choices())),
                              outputs=[language, media])
            demo.load(lambda: build_queue("Needs review", "All languages", "All media types", "", "All projects"), None, view_outputs)

        with gr.Tab("Crawler import"):
            gr.Markdown("Drop the crawler's JSON here or place it at the configured crawler output path. New files at that path are picked up automatically.")
            crawler_file = gr.File(label="Crawler JSON", file_types=[".json"], type="filepath")
            with gr.Row():
                upload_import = gr.Button("Import uploaded JSON", variant="primary")
                scan_import = gr.Button("Check crawler output now")
            import_status = gr.Markdown("Crawler output is checked every 15 seconds while the app is running.")

        with gr.Tab("Model tools"):
            gr.Markdown("### Train on saved human labels\nThe local baseline uses TF-IDF + scikit-learn. It runs without an API key or pretrained model download. Predictions are suggestions and never overwrite human labels.")
            training_status = gr.Markdown()
            training_summary = gr.JSON(label="Training task summary")
            train_button = gr.Button("Train / update all available models", variant="primary")
            train_button.click(train_and_save, None, [training_status, training_summary])
            gr.Markdown("### Apply models to unlabeled messages")
            with gr.Row():
                project_threshold = gr.Slider(0.1, 0.9, value=0.40, step=0.05, label="Project candidate threshold")
                argument_threshold = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="Argument threshold")
                support_threshold = gr.Slider(0.1, 0.9, value=0.55, step=0.05, label="Support review threshold")
            predict_button = gr.Button("Predict remaining unlabeled messages", variant="primary")
            prediction_status = gr.Markdown()
            prediction_summary = gr.JSON(label="Prediction run summary")
            predict_button.click(predict_unlabeled, [project_threshold, argument_threshold, support_threshold], [prediction_status, prediction_summary])
            gr.Markdown("Thresholds are review controls, not validated accuracy guarantees. Review the suggestions from the message queue.")

        with gr.Tab("Progress & export"):
            progress = gr.Markdown(current_progress())
            refresh_progress = gr.Button("Refresh progress")
            refresh_progress.click(export_progress, None, progress)
            with gr.Row():
                labels_file = gr.File(label="Message-project labels CSV")
                args_file = gr.File(label="Argument labels CSV")
                rag_file = gr.File(label="RAG-ready JSONL")
                preds_file = gr.File(label="Model suggestions CSV")
                sqlite_file = gr.File(label="Consistent SQLite backup (for SFTP)")
            with gr.Row():
                gr.Button("Export labels CSV").click(lambda: export_file("labels_csv"), None, labels_file)
                gr.Button("Export arguments CSV").click(lambda: export_file("arguments_csv"), None, args_file)
                gr.Button("Export RAG JSONL").click(lambda: export_file("rag_jsonl"), None, rag_file)
                gr.Button("Export predictions CSV").click(lambda: export_file("predictions_csv"), None, preds_file)
                gr.Button("Create SQLite backup").click(lambda: export_file("sqlite_backup"), None, sqlite_file)
            gr.Markdown(f"SQLite database: `{store.DB_PATH}`\n\nModel files: `{APP_DIR / 'models'}`")

        with gr.Tab("Guide & taxonomy"):
            gr.Markdown("""
### How to label
- **Project Index:** select one concrete project. If a message discusses multiple projects, save one project row, change the Project Index while the message stays open, then save again for the next project.
- **Support Score:** `0` strongly opposes · `1` opposes · `2` leans against · `3` leans in favor · `4` supports · `5` strongly supports. Use a non-scored status for neutral, mixed/conditional, no position, or unclear text.
- **Argument Type(s):** select the argument categories present. The current form applies one shared direction, evidence quote and short abstraction to the selected categories; keep the selection to categories that share that treatment. These fields help later review and argument abstraction.
- **Other named project:** choose that option and enter its name; saving adds it to the project index for reuse.
- **Human review:** uncertain labels can be flagged for a second reviewer.
- **Provisional seed labels:** choose the `Needs review` queue to inspect the 1,000 rule-based pre-labels. They are not confirmed and are excluded from training until a human verifies them and clears the review flag.
- **Language groups:** `French (all variants)` includes `fr`, `fr-FR`, `fr-CH`, `fr-CA`, and other French locale codes. Use the search box for `Heilbronn` or `Stuttgart` to inspect city-name candidates.

### What the model learns
Project Index is a multi-label prediction over named project IDs. Argument types and direction are also multi-label. Support is target-conditioned and trained as an ordinal score using cumulative thresholds. When a text has multiple projects, the model proposes each project and predicts argument/support separately per project.

The local model is a quick baseline, not a validated production classifier. It requires enough positive and negative human examples. The model's raw probabilities are not calibrated confidence. Reserve a source/time-separated human validation sample before using performance claims.

`Argument abstraction` is currently a human-reviewed short paraphrase linked to the original quote. Automatic abstraction is a separate text-generation task and needs examples of verified quote → faithful summary → argument type/direction. The app does not generate summaries automatically.

### Data boundary
The app runs on `127.0.0.1`; it does not publish data or call a remote LLM. The original dataset remains in its JSON file. SQLite stores labels and predictions. CSV/JSONL exports are generated under the app's `data/exports` folder.
""")
            with gr.Row():
                new_project = gr.Textbox(label="Add a project to the index")
                add_project_button = gr.Button("Add project")
                project_added = gr.Markdown()
            add_project_button.click(add_project_category, new_project, [project_added, project])
            with gr.Row():
                new_argument = gr.Textbox(label="Add an argument type")
                add_argument_button = gr.Button("Add argument type")
                taxonomy_status = gr.Markdown()
            add_argument_button.click(add_argument_category, new_argument, [taxonomy_status, argument_types])

    dashboard_revision = gr.State()
    dashboard_inputs = [chart_project, chart_origin, chart_period, dashboard_revision]
    dashboard_outputs = [dashboard_revision, chart_project, coverage, topic_plot, argument_plot, time_plot, source_plot]
    demo.load(refresh_dashboard, dashboard_inputs, dashboard_outputs)
    for control in (chart_project, chart_origin, chart_period):
        control.input(refresh_dashboard, dashboard_inputs, dashboard_outputs)
    for event in (save_event, save_next_event):
        event.success(refresh_dashboard, dashboard_inputs, dashboard_outputs)
    import_outputs = [import_status, language, media]
    upload_import.click(sync_inputs, crawler_file, import_outputs).success(refresh_dashboard, dashboard_inputs, dashboard_outputs)
    scan_import.click(sync_inputs, None, import_outputs).success(refresh_dashboard, dashboard_inputs, dashboard_outputs)
    gr.Timer(15).tick(refresh_dashboard, dashboard_inputs, dashboard_outputs, show_progress="hidden")


def watch_crawler(stop):
    while not stop.wait(15):
        try:
            WORKFLOW.sync()
        except Exception:
            logging.exception("Could not sync crawler output")


if __name__ == "__main__":
    host = os.getenv("LABELER_HOST", "127.0.0.1")
    port = int(os.getenv("LABELER_PORT", "7861"))
    username = os.getenv("LABELER_USERNAME")
    password = os.getenv("LABELER_PASSWORD")
    if host not in {"127.0.0.1", "localhost"} and not (username and password):
        raise RuntimeError("For non-local access, set LABELER_USERNAME and LABELER_PASSWORD; protect the connection with HTTPS.")
    app_auth = (username, password) if username and password else None
    stop = Event()
    Thread(target=watch_crawler, args=(stop,), daemon=True).start()
    try:
        demo.queue(default_concurrency_limit=1).launch(
            server_name=host, server_port=port, auth=app_auth, share=False, show_error=True,
        )
    finally:
        stop.set()

