"""SQLite persistence for the Seilbahn dashboard."""

from __future__ import annotations

import re
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "annotations.sqlite3"

PROJECTS = [
    ("heilbronn_cable_car", "Heilbronn Cable Car", 0),
    ("bonn_cable_car", "Bonn Cable Car", 0),
    ("passau_oberhaus_cable_car", "Passau Oberhaus Cable Car", 0),
    ("stuttgart_urban_cable_car", "Stuttgart Urban Cable Car Proposal", 0),
    ("stuttgart_waldfriedhof_funicular", "Stuttgart Waldfriedhof Funicular (existing)", 0),
    ("paris_cable_c1", "Paris Câble C1", 0),
    ("__general__", "General cable-car topic / no single project", 1),
    ("__other_named__", "Other named cable-car project (enter its name)", 1),
    ("__unclear__", "Unclear project / needs review", 1),
    ("__unrelated__", "Unrelated to a cable-car proposal", 1),
]

ARGUMENTS = [
    ("cost_finance", "Cost / funding / opportunity cost"),
    ("time_convenience", "Travel time / convenience"),
    ("accessibility_connectivity", "Accessibility / network connections"),
    ("capacity_reliability", "Capacity / reliability"),
    ("climate_emissions", "Climate / emissions"),
    ("nature_landscape", "Nature / landscape / trees"),
    ("noise_vibration", "Noise / vibration"),
    ("privacy_visual", "Privacy / visual impact"),
    ("economic_development", "Economic growth / jobs / tourism"),
    ("equity_distribution", "Equity / who benefits or bears the impact"),
    ("safety", "Safety"),
    ("construction_disruption", "Construction disruption"),
    ("governance_participation", "Governance / transparency / participation"),
    ("technical_feasibility", "Technical / operational feasibility"),
    ("other", "Other argument"),
]


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


@contextmanager
def db_session():
    con = connect()
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def initialize() -> None:
    with db_session() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                project_name TEXT NOT NULL UNIQUE,
                is_special INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS argument_classes (
                class_code TEXT PRIMARY KEY,
                class_name TEXT NOT NULL UNIQUE,
                is_active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS annotations (
                dataset_key TEXT NOT NULL,
                mention_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                relevance TEXT NOT NULL,
                stance_status TEXT NOT NULL,
                support_score INTEGER,
                other_project_name TEXT,
                annotator TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                needs_review INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (dataset_key, mention_id, project_id),
                FOREIGN KEY(project_id) REFERENCES projects(project_id)
            );
            CREATE TABLE IF NOT EXISTS argument_labels (
                dataset_key TEXT NOT NULL,
                mention_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                class_code TEXT NOT NULL,
                direction TEXT NOT NULL,
                evidence_quote TEXT NOT NULL DEFAULT '',
                argument_summary TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (dataset_key, mention_id, project_id, class_code),
                FOREIGN KEY(project_id) REFERENCES projects(project_id),
                FOREIGN KEY(class_code) REFERENCES argument_classes(class_code)
            );
            CREATE TABLE IF NOT EXISTS predictions (
                dataset_key TEXT NOT NULL,
                mention_id TEXT NOT NULL,
                task TEXT NOT NULL,
                project_id TEXT NOT NULL DEFAULT '',
                result_json TEXT NOT NULL,
                confidence REAL,
                review_required INTEGER NOT NULL DEFAULT 1,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(dataset_key, mention_id, task, project_id)
            );
            CREATE TABLE IF NOT EXISTS model_runs (
                dataset_key TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_path TEXT NOT NULL,
                trained_at TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                PRIMARY KEY(dataset_key, model_version)
            );
            CREATE TABLE IF NOT EXISTS imported_messages (
                dataset_key TEXT NOT NULL,
                mention_id TEXT NOT NULL,
                record_json TEXT NOT NULL,
                PRIMARY KEY(dataset_key, mention_id)
            );
            CREATE TABLE IF NOT EXISTS keyword_labels (
                dataset_key TEXT NOT NULL,
                mention_id TEXT NOT NULL,
                rule_version TEXT NOT NULL,
                result_json TEXT NOT NULL,
                needs_review INTEGER NOT NULL,
                PRIMARY KEY(dataset_key, mention_id)
            );
            CREATE TABLE IF NOT EXISTS deleted_messages (
                dataset_key TEXT NOT NULL,
                mention_id TEXT NOT NULL,
                record_json TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                deleted_at TEXT NOT NULL,
                PRIMARY KEY (dataset_key, mention_id)
            );
            CREATE TABLE IF NOT EXISTS workspace_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            );
            """
        )
        for project_id, name, special in PROJECTS:
            con.execute(
                "INSERT OR IGNORE INTO projects(project_id, project_name, is_special) VALUES (?, ?, ?)",
                (project_id, name, special),
            )
        for code, name in ARGUMENTS:
            con.execute(
                "INSERT OR IGNORE INTO argument_classes(class_code, class_name) VALUES (?, ?)",
                (code, name),
            )


def list_projects() -> list[dict[str, Any]]:
    with db_session() as con:
        return [dict(r) for r in con.execute("SELECT * FROM projects ORDER BY is_special, project_name")]


def list_argument_classes() -> list[dict[str, Any]]:
    with db_session() as con:
        return [dict(r) for r in con.execute("SELECT * FROM argument_classes WHERE is_active=1 ORDER BY class_name")]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug or "project"


def add_project(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Project name cannot be empty")
    with db_session() as con:
        existing = con.execute("SELECT project_id FROM projects WHERE project_name=?", (name,)).fetchone()
        if existing:
            return str(existing["project_id"])
        base = _slug(name)
        candidate = base
        suffix = 2
        while con.execute("SELECT 1 FROM projects WHERE project_id=?", (candidate,)).fetchone():
            candidate = f"{base}_{suffix}"
            suffix += 1
        con.execute("INSERT INTO projects(project_id, project_name, is_special) VALUES (?, ?, 0)", (candidate, name))
        return candidate


def add_argument_class(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Argument name cannot be empty")
    with db_session() as con:
        existing = con.execute("SELECT class_code FROM argument_classes WHERE class_name=?", (name,)).fetchone()
        if existing:
            return str(existing["class_code"])
        base = _slug(name)
        candidate = base
        suffix = 2
        while con.execute("SELECT 1 FROM argument_classes WHERE class_code=?", (candidate,)).fetchone():
            candidate = f"{base}_{suffix}"
            suffix += 1
        con.execute("INSERT INTO argument_classes(class_code, class_name) VALUES (?, ?)", (candidate, name))
        return candidate


def get_or_set_dataset_key(proposed: str) -> str:
    """Keep a stable active dataset identity if its original source file is absent."""
    with db_session() as con:
        row = con.execute(
            "SELECT setting_value FROM workspace_settings WHERE setting_key='active_dataset_key'"
        ).fetchone()
        if row:
            return str(row[0])
        candidates = {r[0] for r in con.execute(
            "SELECT dataset_key FROM annotations UNION SELECT dataset_key FROM imported_messages"
        )}
        value = next(iter(candidates)) if len(candidates) == 1 else proposed
        con.execute(
            "INSERT INTO workspace_settings(setting_key, setting_value) VALUES('active_dataset_key', ?)",
            (value,),
        )
        return value


def deleted_message_ids(dataset_key: str) -> set[str]:
    with db_session() as con:
        return {str(r[0]) for r in con.execute(
            "SELECT mention_id FROM deleted_messages WHERE dataset_key=?", (dataset_key,)
        )}


def get_deleted_messages(dataset_key: str) -> list[dict[str, Any]]:
    with db_session() as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM deleted_messages WHERE dataset_key=? ORDER BY deleted_at DESC", (dataset_key,)
        )]


def delete_message(dataset_key: str, mention_id: str, record: dict[str, Any], reason: str = "") -> None:
    """Hide a message and its labels from active views, retaining a restorable snapshot."""
    with db_session() as con:
        con.execute(
            """INSERT INTO deleted_messages(dataset_key, mention_id, record_json, reason, deleted_at)
               VALUES(?,?,?,?,?) ON CONFLICT(dataset_key, mention_id) DO UPDATE SET
               record_json=excluded.record_json, reason=excluded.reason, deleted_at=excluded.deleted_at""",
            (dataset_key, mention_id, json.dumps(record, ensure_ascii=False), reason.strip(),
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )


def restore_message(dataset_key: str, mention_id: str) -> dict[str, Any] | None:
    with db_session() as con:
        row = con.execute(
            "SELECT record_json FROM deleted_messages WHERE dataset_key=? AND mention_id=?",
            (dataset_key, mention_id),
        ).fetchone()
        if not row:
            return None
        record = json.loads(row[0])
        con.execute(
            "INSERT OR IGNORE INTO imported_messages(dataset_key, mention_id, record_json) VALUES(?,?,?)",
            (dataset_key, mention_id, json.dumps(record, ensure_ascii=False)),
        )
        con.execute("DELETE FROM deleted_messages WHERE dataset_key=? AND mention_id=?", (dataset_key, mention_id))
        return record


def get_annotation(dataset_key: str, mention_id: str, project_id: str) -> dict[str, Any] | None:
    with db_session() as con:
        row = con.execute(
            "SELECT * FROM annotations WHERE dataset_key=? AND mention_id=? AND project_id=?",
            (dataset_key, mention_id, project_id),
        ).fetchone()
        return dict(row) if row else None


def get_annotations_for_message(dataset_key: str, mention_id: str) -> list[dict[str, Any]]:
    with db_session() as con:
        rows = con.execute(
            "SELECT a.*, p.project_name FROM annotations a JOIN projects p USING(project_id) "
            "WHERE dataset_key=? AND mention_id=? ORDER BY a.updated_at DESC",
            (dataset_key, mention_id),
        ).fetchall()
        return [dict(r) for r in rows]


def get_arguments_for_annotation(dataset_key: str, mention_id: str, project_id: str) -> list[dict[str, Any]]:
    with db_session() as con:
        rows = con.execute(
            "SELECT al.*, ac.class_name FROM argument_labels al JOIN argument_classes ac USING(class_code) "
            "WHERE dataset_key=? AND mention_id=? AND project_id=? ORDER BY ac.class_name",
            (dataset_key, mention_id, project_id),
        ).fetchall()
        return [dict(r) for r in rows]


def save_annotation(data: dict[str, Any], arguments: list[dict[str, Any]]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    with db_session() as con:
        if con.execute(
            "SELECT 1 FROM deleted_messages WHERE dataset_key=? AND mention_id=?",
            (data["dataset_key"], data["mention_id"]),
        ).fetchone():
            raise ValueError("This message was removed from the active dataset. Restore it before editing labels.")
        if data.get("replace_rule_seed"):
            con.execute("BEGIN IMMEDIATE")
            original = data.get("original_project_id")
            key = (data["dataset_key"], data["mention_id"], original or data["project_id"])
            current = con.execute(
                """SELECT updated_at FROM annotations WHERE dataset_key=? AND mention_id=? AND project_id=?
                   AND annotator<>'codex_rule_prelabel_v1'""", key,
            ).fetchone()
            if (current[0] if current else None) != data.get("expected_updated_at"):
                raise ValueError("This label changed since you opened it. Reload the queue before saving.")
        if data.get("replace_rule_seed"):
            key = (data["dataset_key"], data["mention_id"])
            con.execute(
                """DELETE FROM argument_labels WHERE dataset_key=? AND mention_id=? AND project_id IN (
                       SELECT project_id FROM annotations WHERE dataset_key=? AND mention_id=?
                       AND annotator='codex_rule_prelabel_v1')""", (*key, *key),
            )
            con.execute(
                """DELETE FROM annotations WHERE dataset_key=? AND mention_id=?
                   AND annotator='codex_rule_prelabel_v1'""", key,
            )
        old_project = data.get("original_project_id")
        if old_project and old_project != data["project_id"]:
            if con.execute(
                "SELECT 1 FROM annotations WHERE dataset_key=? AND mention_id=? AND project_id=?",
                (data["dataset_key"], data["mention_id"], data["project_id"]),
            ).fetchone():
                raise ValueError("This project already has a saved label. Select it in the editing menu.")
            key = (data["dataset_key"], data["mention_id"], old_project)
            con.execute("DELETE FROM argument_labels WHERE dataset_key=? AND mention_id=? AND project_id=?", key)
            con.execute("DELETE FROM annotations WHERE dataset_key=? AND mention_id=? AND project_id=?", key)
        con.execute(
            """INSERT INTO annotations(
                dataset_key, mention_id, project_id, relevance, stance_status, support_score,
                other_project_name, annotator, notes, needs_review, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(dataset_key, mention_id, project_id) DO UPDATE SET
                relevance=excluded.relevance, stance_status=excluded.stance_status,
                support_score=excluded.support_score, other_project_name=excluded.other_project_name,
                annotator=excluded.annotator, notes=excluded.notes, needs_review=excluded.needs_review,
                updated_at=excluded.updated_at""",
            (
                data["dataset_key"], data["mention_id"], data["project_id"], data["relevance"],
                data["stance_status"], data["support_score"], data.get("other_project_name"),
                data.get("annotator", ""), data.get("notes", ""), int(data.get("needs_review", False)), now,
            ),
        )
        con.execute(
            "DELETE FROM argument_labels WHERE dataset_key=? AND mention_id=? AND project_id=?",
            (data["dataset_key"], data["mention_id"], data["project_id"]),
        )
        for arg in arguments:
            con.execute(
                """INSERT INTO argument_labels(
                    dataset_key, mention_id, project_id, class_code, direction, evidence_quote, argument_summary
                ) VALUES(?,?,?,?,?,?,?)""",
                (
                    data["dataset_key"], data["mention_id"], data["project_id"], arg["class_code"],
                    arg.get("direction", "unclear"), arg.get("evidence_quote", ""), arg.get("argument_summary", ""),
                ),
            )


def annotation_overview(dataset_key: str) -> tuple[set[str], set[str], int]:
    with db_session() as con:
        rows = con.execute(
            """SELECT a.mention_id, a.needs_review FROM annotations a WHERE a.dataset_key=?
               AND NOT EXISTS (SELECT 1 FROM deleted_messages d
                               WHERE d.dataset_key=a.dataset_key AND d.mention_id=a.mention_id)""",
            (dataset_key,),
        ).fetchall()
        labeled = {str(r["mention_id"]) for r in rows}
        review = {str(r["mention_id"]) for r in rows if r["needs_review"]}
        return labeled, review, len(rows)


def get_training_data(dataset_key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with db_session() as con:
        annotations = [dict(r) for r in con.execute(
            """SELECT a.* FROM annotations a WHERE a.dataset_key=? AND a.needs_review=0
               AND a.annotator <> 'codex_rule_prelabel_v1'
               AND NOT EXISTS (SELECT 1 FROM deleted_messages d
                               WHERE d.dataset_key=a.dataset_key AND d.mention_id=a.mention_id)
               ORDER BY a.mention_id, a.project_id""", (dataset_key,)
        )]
        arguments = [dict(r) for r in con.execute(
            """SELECT l.* FROM argument_labels l
               JOIN annotations a USING(dataset_key, mention_id, project_id)
               WHERE l.dataset_key=? AND a.needs_review=0 AND a.annotator <> 'codex_rule_prelabel_v1'
               AND NOT EXISTS (SELECT 1 FROM deleted_messages d
                               WHERE d.dataset_key=l.dataset_key AND d.mention_id=l.mention_id)
               ORDER BY l.mention_id, l.project_id, l.class_code""",
            (dataset_key,),
        )]
    return annotations, arguments


def save_model_run(dataset_key: str, model_version: str, model_path: str, summary_json: str) -> None:
    with db_session() as con:
        con.execute(
            "INSERT OR REPLACE INTO model_runs VALUES(?,?,?,?,?)",
            (dataset_key, model_version, model_path, datetime.now(timezone.utc).isoformat(timespec="seconds"), summary_json),
        )


def save_predictions(dataset_key: str, model_version: str, rows: list[dict[str, Any]]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db_session() as con:
        con.execute("DELETE FROM predictions WHERE dataset_key=?", (dataset_key,))
        con.executemany(
            """INSERT INTO predictions(
                dataset_key, mention_id, task, project_id, result_json, confidence,
                review_required, model_version, created_at
            ) VALUES(?,?,?,?,?,?,?,?,?)""",
            [
                (dataset_key, r["mention_id"], r["task"], r.get("project_id", ""), r["result_json"],
                 r.get("confidence"), int(r.get("review_required", True)), model_version, now)
                for r in rows
            ],
        )


def predictions_for_message(dataset_key: str, mention_id: str) -> list[dict[str, Any]]:
    with db_session() as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM predictions WHERE dataset_key=? AND mention_id=? ORDER BY task, project_id",
            (dataset_key, mention_id),
        )]


def prediction_rows(dataset_key: str) -> list[dict[str, Any]]:
    with db_session() as con:
        return [dict(r) for r in con.execute(
            """SELECT p.* FROM predictions p WHERE p.dataset_key=?
               AND NOT EXISTS (SELECT 1 FROM deleted_messages d
                               WHERE d.dataset_key=p.dataset_key AND d.mention_id=p.mention_id)
               ORDER BY p.mention_id, p.task, p.project_id""",
            (dataset_key,),
        )]


def backup_database(destination: str | Path) -> None:
    """Create a consistent SQLite snapshot, including committed WAL contents."""
    source = connect()
    target = sqlite3.connect(str(destination), timeout=30)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def export_rows(dataset_key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with db_session() as con:
        labels = [dict(r) for r in con.execute(
            """SELECT a.*, p.project_name FROM annotations a JOIN projects p USING(project_id)
               WHERE a.dataset_key=? AND NOT EXISTS (SELECT 1 FROM deleted_messages d
                   WHERE d.dataset_key=a.dataset_key AND d.mention_id=a.mention_id)
               ORDER BY a.mention_id, a.project_id""", (dataset_key,)
        )]
        args = [dict(r) for r in con.execute(
            """SELECT al.*, ac.class_name, p.project_name FROM argument_labels al
               JOIN argument_classes ac USING(class_code) JOIN projects p USING(project_id)
               WHERE al.dataset_key=? AND NOT EXISTS (SELECT 1 FROM deleted_messages d
                   WHERE d.dataset_key=al.dataset_key AND d.mention_id=al.mention_id)
               ORDER BY al.mention_id, al.project_id, al.class_code""", (dataset_key,)
        )]
    return labels, args

