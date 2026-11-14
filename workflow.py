"""Persist incoming messages and keyword labels in the labeler's SQLite database."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import store
from keyword_labels import classify


LEGACY = "codex_rule_prelabel_v1"
DEFAULT_CRAWLER_NAME = "heilbronn_cable_car_data.json"
RULE_VERSION = "cable-keywords-1"


def article_url(record):
    media = str(record.get("media_type_name", "")).casefold().replace("_", " ")
    if not any(word in media for word in ("news", "press", "newspaper", "article")):
        return ""
    try:
        parts = urlsplit(record.get("url") or "")
    except ValueError:
        return ""
    if not parts.netloc:
        return ""
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
    return urlunsplit(("", parts.netloc.casefold().removeprefix("www."), parts.path.rstrip("/"),
                       urlencode(sorted(query)), ""))


def content_key(record):
    values = [" ".join(str(record.get(key) or "").casefold().split())
              for key in ("title", "content", "published_at", "url", "source_name")]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


class Workflow:
    def __init__(self, dataset_key, records, parse, root):
        self.dataset_key = dataset_key
        self.records = records
        self.by_id = {row["mention_id"]: row for row in records}
        self.urls = {url for row in records if (url := article_url(row))}
        self.content_keys = {content_key(row) for row in records}
        self.parse = parse
        self.root = root
        self.lock = RLock()
        self.file_versions = {}
        self.file_errors = {}
        self.previous_taxonomy = None
        self.last_import_rowid = 0
        self._load_imported()

    def _load_imported(self):
        with store.db_session() as con:
            rows = con.execute(
                "SELECT rowid, record_json FROM imported_messages WHERE dataset_key=? AND rowid>? ORDER BY rowid",
                (self.dataset_key, self.last_import_rowid),
            ).fetchall()
        for rowid, raw in rows:
            record = json.loads(raw)
            self.last_import_rowid = rowid
            if record["mention_id"] not in self.by_id:
                self.by_id[record["mention_id"]] = record
                self.records.append(record)
                self.content_keys.add(content_key(record))
                if url := article_url(record):
                    self.urls.add(url)

    def _paths(self):
        configured = os.getenv("LABELER_CRAWLER_IMPORT_PATH")
        paths = [Path(configured).expanduser()] if configured else [
            self.root / "Crawler" / DEFAULT_CRAWLER_NAME,
            self.root / DEFAULT_CRAWLER_NAME,
            store.APP_DIR / DEFAULT_CRAWLER_NAME,
        ]
        for path in paths:
            if path.is_dir():
                yield from sorted(path.glob("*.json"))
            else:
                yield path

    @staticmethod
    def _version(projects, classes):
        taxonomy = ([(p["project_id"], p["project_name"]) for p in projects],
                    [a["class_code"] for a in classes])
        return RULE_VERSION + ":" + hashlib.sha256(repr(taxonomy).encode()).hexdigest()[:16]

    def _prepare(self, records, projects, classes, version):
        result = []
        for record in records:
            label = classify(record, projects, classes)
            result.append((self.dataset_key, record["mention_id"], version,
                           json.dumps(label, ensure_ascii=False), int(label["needs_review"])))
        return result

    @staticmethod
    def _save_keywords(con, rows):
        con.executemany(
            """INSERT INTO keyword_labels VALUES(?,?,?,?,?)
               ON CONFLICT(dataset_key, mention_id) DO UPDATE SET
               rule_version=excluded.rule_version, result_json=excluded.result_json,
               needs_review=excluded.needs_review""",
            rows,
        )

    def _import(self, path, projects, classes, version):
        raw = path.read_bytes()
        if json.loads(raw.decode("utf-8-sig")) == []:
            return 0, 0
        records = self.parse(path.name, raw)
        added = []
        duplicates = 0
        keys = set()
        urls = set()
        with store.db_session() as con:
            con.execute("BEGIN IMMEDIATE")
            self._load_imported()
            for record in records:
                key = content_key(record)
                url = article_url(record)
                source_id = record["mention_id"]
                if key in self.content_keys or key in keys or (url and (url in self.urls or url in urls)):
                    duplicates += 1
                    continue
                if not record["title"].strip() and not record["content"].strip():
                    continue
                if source_id.startswith("row_") or source_id in self.by_id:
                    source_id = f"import:{key}"
                if source_id in self.by_id:
                    duplicates += 1
                    continue
                added.append(dict(record, mention_id=source_id, import_file=path.name))
                keys.add(key)
                if url:
                    urls.add(url)
            con.executemany("INSERT INTO imported_messages VALUES(?,?,?)", [
                (self.dataset_key, r["mention_id"], json.dumps(r, ensure_ascii=False)) for r in added
            ])
            self._save_keywords(con, self._prepare(added, projects, classes, version))
        self._load_imported()
        return len(added), duplicates

    def sync(self, uploads=(), scan=True):
        with self.lock:
            self._load_imported()
            projects, classes = store.list_projects(), store.list_argument_classes()
            version = self._version(projects, classes)
            if self.previous_taxonomy != version:
                with store.db_session() as con:
                    cached = {row[0] for row in con.execute(
                        "SELECT mention_id FROM keyword_labels WHERE dataset_key=? AND rule_version=?",
                        (self.dataset_key, version),
                    )}
                pending = [record for record in self.records if record["mention_id"] not in cached]
                labels = self._prepare(pending, projects, classes, version)
                with store.db_session() as con:
                    self._save_keywords(con, labels)
                self.previous_taxonomy = version
            added = duplicates = 0
            for path in [*(self._paths() if scan else []), *map(Path, uploads)]:
                if not path.is_file() or path.suffix.casefold() != ".json":
                    continue
                try:
                    stat = path.stat()
                    signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                    if self.file_versions.get(path) == signature:
                        continue
                    fresh, known = self._import(path, projects, classes, version)
                except (OSError, ValueError) as exc:
                    self.file_errors[path] = f"{path.name}: {exc}"
                else:
                    self.file_errors.pop(path, None)
                    added += fresh
                    duplicates += known
                    self.file_versions[path] = signature
            return added, duplicates, list(self.file_errors.values())

    def keyword_for(self, mention_id):
        with store.db_session() as con:
            row = con.execute("SELECT result_json FROM keyword_labels WHERE dataset_key=? AND mention_id=?",
                              (self.dataset_key, mention_id)).fetchone()
        return json.loads(row[0]) if row else None

    def revision(self):
        with store.db_session() as con:
            human = tuple(con.execute(
                "SELECT count(*), max(updated_at) FROM annotations WHERE dataset_key=?",
                (self.dataset_key,),
            ).fetchone())
        return (*human, self.previous_taxonomy, len(self.records))

    def queue_ids(self):
        with store.db_session() as con:
            human = con.execute("SELECT mention_id, needs_review, annotator FROM annotations WHERE dataset_key=?",
                                (self.dataset_key,)).fetchall()
            keywords = con.execute("SELECT mention_id, needs_review FROM keyword_labels WHERE dataset_key=?",
                                   (self.dataset_key,)).fetchall()
        saved = {r["mention_id"] for r in human if r["annotator"] != LEGACY}
        pending = {r["mention_id"] for r in human if r["needs_review"] or r["annotator"] == LEGACY}
        pending.update(r["mention_id"] for r in keywords if r["needs_review"] and r["mention_id"] not in saved)
        labeled = saved | {r["mention_id"] for r in keywords if not r["needs_review"]}
        return labeled, pending

    def project_mentions(self, project_id):
        labels, _ = self.dashboard_rows()
        return {row["mention_id"] for row in labels if row["project_id"] == project_id}

    def dashboard_rows(self):
        with store.db_session() as con:
            con.execute("BEGIN")
            labels = [dict(row) for row in con.execute(
                "SELECT * FROM annotations WHERE dataset_key=? AND annotator<>?", (self.dataset_key, LEGACY),
            )]
            args = [dict(row) for row in con.execute(
                """SELECT l.* FROM argument_labels l JOIN annotations a USING(dataset_key,mention_id,project_id)
                   WHERE l.dataset_key=? AND a.annotator<>?""", (self.dataset_key, LEGACY),
            )]
            keywords = con.execute(
                """SELECT k.mention_id,k.result_json FROM keyword_labels k
                   WHERE k.dataset_key=? AND NOT EXISTS (
                       SELECT 1 FROM annotations a WHERE a.dataset_key=k.dataset_key
                       AND a.mention_id=k.mention_id AND a.annotator<>?)""", (self.dataset_key, LEGACY),
            ).fetchall()
        for mention_id, raw in keywords:
            result = json.loads(raw)
            identity = dict(dataset_key=self.dataset_key, mention_id=mention_id,
                            project_id=result["project_id"], label_source="keyword")
            args.extend(dict(arg, **identity) for arg in result.pop("arguments"))
            labels.append(dict(result, **identity))
        return labels, args
