"""Create an auditable, provisional 1,000-message labeling queue.

This is a deterministic rule-based pre-labeler, not a trained model and not a
human gold-labeling pass. Every inserted annotation is flagged needs_review=1
and therefore must be checked in app.py before it is eligible for training.
Existing annotations are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any

import store


APP_DIR = Path(__file__).resolve().parent
SOURCE = APP_DIR.parent / "Participants - Future City Hackathon" / "Stadt Heilbronn" / "FSI_Heilbronn_Medienbeiträge (1).json"
OUT = APP_DIR / "data" / "exports"
SEED = 20261114
LIMIT = 1000
ANNOTATOR = "codex_rule_prelabel_v1"

CABLE_TERMS = [
    "seilbahn", "seilbahnen", "kabelbahn", "luftseilbahn", "standseilbahn", "gondelbahn",
    "téléphérique", "telepherique", "télécabine", "telecabine", "funiculaire",
    "câble c1", "cable c1", "transport par câble", "transport par cable",
    "cable car", "cable-car", "urban cable car", "ropeway", "gondola",
    "teleférico", "teleferico", "cabinovia", "cablebús", "cablebus",
]

ARGUMENT_RULES = {
    "cost_finance": [r"\bkosten?\b", r"kostenexplosion", r"teuer", r"finanz\w*", r"budget", r"investition\w*", r"milliard\w*", r"preis\w*", r"\bcoût\w*", r"\bcouteux\b", r"\bcher\b", r"\bbudget\b", r"\bfinancement\b", r"\binvestissement\w*", r"\bprix\b"],
    "time_convenience": [r"fahrzeit", r"zeiterspar\w*", r"schnell\w*", r"direktverbindung", r"bequem\w*", r"\btemps de trajet\b", r"\bgain de temps\b", r"\brapide\w*", r"\bpratique\b"],
    "accessibility_connectivity": [r"anbindung", r"anschluss", r"barrierefrei\w*", r"zugänglich", r"erreichbar", r"\baccessibilité\b", r"\baccessible\b", r"\bconnexion\b", r"\bcorrespondance\b", r"\bmobilité\b", r"\bdesserte\b"],
    "capacity_reliability": [r"kapazität", r"zuverläss\w*", r"betriebsausfall", r"windanfällig", r"wetterbedingt", r"\bcapacité\b", r"\bfiable\b", r"\bpanne\b", r"\bvent\b", r"maintenance", r"fréquence"],
    "climate_emissions": [r"\bco2\b", r"emission\w*", r"klimaschutz", r"klimafreund\w*", r"umweltfreund\w*", r"\bécolog\w*", r"\bémission\w*", r"\bclimat\b", r"\bpollution\b"],
    "nature_landscape": [r"\bnatur\w*", r"bäum\w*", r"wald\w*", r"landschaft\w*", r"biodivers\w*", r"\bnature\b", r"\barbre\w*", r"\bpaysage\w*", r"\bfaune\b", r"\bflore\b", r"\bmontagne\b"],
    "noise_vibration": [r"lärm\w*", r"geräusch\w*", r"vibration\w*", r"\bbruit\w*", r"\bnuisance sonore\w*", r"\bsonore\b"],
    "privacy_visual": [r"privatsphäre", r"einblick\w*", r"aussicht\w*", r"sichtachse", r"\bvue\b", r"\bvisuel\w*", r"\besthétique\w*", r"\bfenêtre\w*", r"\bprivacy\b"],
    "economic_development": [r"arbeitsplätze", r"wirtschaft\w*", r"tourism\w*", r"\bjobs?\b", r"\bemploi\w*", r"\béconomie\w*", r"\bcommerces?\b", r"\bretombées économiques\b"],
    "equity_distribution": [r"gerecht\w*", r"ungleich\w*", r"gleichberechtig\w*", r"sozial\w* gerecht", r"\béquit\w*", r"\binégal\w*", r"\bjustice sociale\b", r"\bqui bénéficie\b"],
    "safety": [r"sicher\w*", r"unfall\w*", r"gefähr\w*", r"\bsécurit\w*", r"\baccident\w*", r"\bdanger\w*", r"\brisque\w*"],
    "construction_disruption": [r"baustell\w*", r"bauarbeiten", r"bauphase", r"sperrung\w*", r"\btravaux\b", r"\bchantier\w*", r"\bconstruction\b", r"\bperturbation\w*"],
    "governance_participation": [r"bürgerbeteilig\w*", r"bürgerentscheid", r"abstimmung\w*", r"transparenz", r"mitbestimm\w*", r"\bconsultation\b", r"\bconcertation\b", r"\bparticipation\b", r"\bdécision\b", r"\bdémocratie\b"],
    "technical_feasibility": [r"machbar\w*", r"realisierbar", r"technisch\w*", r"umsetzbar", r"\bfaisabilité\b", r"\bréalisable\b", r"\btechnique\b", r"\bfeasible\b"],
}

PRO_PATTERNS = [
    r"\bich bin dafür\b", r"\bwir sind dafür\b", r"\bich unterstütz\w*\b", r"\bbefürwort\w*\b",
    r"\bsinnvoll\b", r"\bgute idee\b", r"\bnotwendig\b", r"\bverbessert den öpnv\b",
    r"\bje suis pour\b", r"\bje soutiens\b", r"\bbonne idée\b", r"\butile\b", r"\bnécessaire\b", r"\bfavorable\b",
]
CON_PATTERNS = [
    r"\bich bin dagegen\b", r"\bwir sind dagegen\b", r"\bgegen die seilbahn\b", r"\bseilbahn.*\bunsinnig\b",
    r"\bsinnlos\b", r"\bunnötig\b", r"\bzu teuer\b", r"\bverschwendung\b", r"\bmuss verhindert werden\b",
    r"\bje suis contre\b", r"\bmauvaise idée\b", r"\binutile\b", r"\btrop cher\b", r"\bje m'oppose\b", r"\bà éviter\b",
]


def normalize(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        for key in ("data", "records", "mentions", "items"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError("Expected a JSON array in the source file.")
    seen: Counter[str] = Counter()
    output = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        base_id = str(item.get("mention_id") or item.get("id") or f"row_{index + 1}")
        seen[base_id] += 1
        mention_id = base_id if seen[base_id] == 1 else f"{base_id}#{seen[base_id]}"
        output.append({
            "mention_id": mention_id,
            "source_mention_id": base_id,
            "title": str(item.get("title") or ""),
            "content": str(item.get("content") or item.get("text") or item.get("message") or ""),
            "published_at": str(item.get("published_at") or item.get("date") or ""),
            "language_code": str(item.get("language_code") or "unknown"),
            "media_type_name": str(item.get("media_type_name") or item.get("media_type") or "unknown"),
            "source_name": str(item.get("source_name") or ""),
            "url": str(item.get("url") or ""),
            "country_code": str(item.get("country_code") or ""),
            "sentiment": item.get("sentiment"),
            "reach": item.get("reach"),
            "engagements": item.get("engagements"),
        })
    return output


def dataset_fingerprint(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        for value in (row["mention_id"], row["url"], row["published_at"], row["title"], row["content"][:128]):
            digest.update(str(value).encode("utf-8", errors="ignore"))
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def normalized_text(row: dict[str, Any]) -> tuple[str, str]:
    title = unescape(re.sub(r"<[^>]+>", " ", row["title"])).casefold()
    content = unescape(re.sub(r"<[^>]+>", " ", row["content"])).casefold()
    return title, title + " " + content


def has_cable_term(text: str) -> bool:
    return any(term in text for term in CABLE_TERMS)


def sample_pool(rows: list[dict[str, Any]], n: int, rng: random.Random, exclude: set[str]) -> list[tuple[dict[str, Any], str]]:
    pool = [row for row in rows if row["mention_id"] not in exclude]
    rng.shuffle(pool)
    return [(row, "") for row in pool[:n]]


def select_records(records: list[dict[str, Any]], labeled: set[str], limit: int = LIMIT) -> list[tuple[dict[str, Any], str]]:
    rng = random.Random(SEED)
    unlabelled = [row for row in records if row["mention_id"] not in labeled]
    selected: list[tuple[dict[str, Any], str]] = []
    used: set[str] = set()

    def add(pool: list[dict[str, Any]], tag: str, maximum: int | None = None) -> None:
        available = [row for row in pool if row["mention_id"] not in used]
        rng.shuffle(available)
        if maximum is not None:
            available = available[:maximum]
        for row in available:
            if len(selected) >= limit:
                break
            selected.append((row, tag))
            used.add(row["mention_id"])

    he = []
    st = []
    french = []
    cable_other = []
    for row in unlabelled:
        title, all_text = normalized_text(row)
        cable = has_cable_term(all_text)
        if cable and "heilbronn" in all_text:
            he.append(row)
        elif cable and "stuttgart" in all_text:
            st.append(row)
        elif cable and row["language_code"].casefold().startswith("fr"):
            french.append(row)
        elif cable:
            cable_other.append(row)

    add(he, "heilbronn_city_and_cablecar_keyword")
    add(st, "stuttgart_city_and_cablecar_keyword")
    add(french, "french_language_and_cablecar_keyword", maximum=200)
    add(cable_other, "other_language_or_project_cablecar_keyword")
    if len(selected) < limit:
        add([row for row in unlabelled if row["mention_id"] not in used], "random_unlabelled_fallback")
    if len(selected) < limit:
        raise ValueError(f"Only {len(selected)} unique unlabeled messages were available; {limit} requested.")
    return selected[:limit]


def first_match(text: str, patterns: list[str]) -> tuple[str, str] | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(0), pattern
    return None


def stance_label(text: str, media_type: str) -> tuple[str, int | None, str]:
    pro = first_match(text, PRO_PATTERNS)
    con = first_match(text, CON_PATTERNS)
    if pro:
        matched = re.search(pro[1], text, re.I)
        before = text[max(0, matched.start() - 18):matched.start()] if matched else ""
        if re.search(r"\b(?:nicht|kein\w*|not|no|pas|never|nie)\s*$", before, re.I):
            pro = None
    if pro and con:
        return "mixed_or_conditional", None, f"both stance cue {pro[0]!r} and {con[0]!r}"
    if pro:
        score = 5 if any(x in pro[0] for x in ("absolut", "voll", "sehr", "je soutiens")) else 4
        return "scored", score, f"pro cue {pro[0]!r}"
    if con:
        score = 0 if any(x in con[0] for x in ("absolut", "völlig", "komplett", "je m'oppose")) else 1
        return "scored", score, f"con cue {con[0]!r}"
    # In news-like source types, absence of a clear first-person stance is usually no_stance.
    media = media_type.casefold()
    if any(token in media for token in ("news", "newspaper", "newspaper", "radio", "television", "press")):
        return "no_stance", None, "no explicit stance phrase detected in a news/press item"
    return "unclear", None, "no high-precision stance cue detected"


def argument_labels(text: str, score: int | None, stance_status: str) -> tuple[list[dict[str, str]], list[str]]:
    labels = []
    hits = []
    direction = "pro" if score is not None and score >= 3 else ("con" if score is not None else "unclear")
    for code, patterns in ARGUMENT_RULES.items():
        hit = first_match(text, patterns)
        if not hit:
            continue
        evidence = hit[0][:180]
        labels.append({"class_code": code, "direction": direction, "evidence_quote": evidence, "argument_summary": ""})
        hits.append(f"{code}:{evidence}")
    return labels, hits


def assign_project(row: dict[str, Any], stratum: str) -> tuple[str, str, str]:
    title, text = normalized_text(row)
    if "heilbronn" in text:
        if any(term in title for term in ("seilbahn", "kabelbahn", "gondel")) or "heilbronn" in title and has_cable_term(title):
            return "heilbronn_cable_car", "project_specific", "Heilbronn name and cable-car term co-occur in title"
        return "__unclear__", "unclear", "Heilbronn appears, but the title does not clearly identify the cable-car proposal"
    if "stuttgart" in text:
        if any(term in text for term in ("waldfriedhof", "südheimer platz", "suedheimer platz", "standseilbahn")):
            return "stuttgart_waldfriedhof_funicular", "project_specific", "existing Stuttgart Waldfriedhof funicular keywords"
        if has_cable_term(title) and "stuttgart" in title:
            return "stuttgart_urban_cable_car", "project_specific", "Stuttgart and cable-car terms co-occur in title"
        return "__unclear__", "unclear", "Stuttgart appears, but title-level cable-car project match is weak"
    if row["language_code"].casefold().startswith("fr"):
        if any(token in text for token in ("câble c1", "cable c1", "créteil", "creteil", "villeneuve-saint-georges")):
            return "paris_cable_c1", "project_specific", "Paris Câble C1 corridor anchor term"
        return "__unclear__", "unclear", "French cable-car-related item; specific project not resolved by rule"
    if has_cable_term(text):
        return "__general__", "general", "cable-car term found without a mapped project"
    return "__unclear__", "unclear", "no confident project match"


def write_seed(records: list[dict[str, Any]], dataset_key: str, apply: bool) -> tuple[list[dict[str, Any]], Path]:
    store.initialize()
    existing, _, _ = store.annotation_overview(dataset_key)
    selected = select_records(records, existing)
    # Add this named project only if French C1 records in this sample require it.
    project_rows = {p["project_id"]: p["project_name"] for p in store.list_projects()}
    if any("câble c1" in normalized_text(row)[1] or "cable c1" in normalized_text(row)[1] or "créteil" in normalized_text(row)[1] or "creteil" in normalized_text(row)[1] for row, _ in selected) and "paris_cable_c1" not in project_rows:
        project_id = store.add_project("Paris Câble C1")
        project_rows[project_id] = "Paris Câble C1"

    manifest = []
    for row, stratum in selected:
        _, text = normalized_text(row)
        project_id, relevance, project_reason = assign_project(row, stratum)
        stance_status, score, stance_reason = stance_label(text, row["media_type_name"])
        args, arg_hits = argument_labels(text, score, stance_status)
        note = (
            "WEAK PRE-LABEL ONLY; NOT HUMAN-CONFIRMED. "
            f"stratum={stratum}; project_rule={project_reason}; stance_rule={stance_reason}; "
            f"argument_hits={'; '.join(arg_hits) if arg_hits else 'none matched'}; "
            "source sentiment was deliberately ignored. Verify and correct in app.py before training."
        )
        data = {
            "dataset_key": dataset_key,
            "mention_id": row["mention_id"],
            "project_id": project_id,
            "relevance": relevance,
            "stance_status": stance_status,
            "support_score": score,
            "other_project_name": None,
            "annotator": ANNOTATOR,
            "notes": note,
            "needs_review": True,
        }
        manifest.append({
            "mention_id": row["mention_id"], "stratum": stratum,
            "language_code": row["language_code"], "project_id": project_id,
            "relevance": relevance, "stance_status": stance_status,
            "support_score": score if score is not None else "",
            "argument_labels_json": json.dumps(args, ensure_ascii=False),
            "title": row["title"], "content": row["content"],
            "notes": note, "annotator": ANNOTATOR, "needs_review": 1,
        })

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"weak_label_seed_{LIMIT}_{SEED}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    if apply:
        # Fail closed if any selected item acquired a human annotation after selection.
        current, _, _ = store.annotation_overview(dataset_key)
        collisions = current.intersection(item["mention_id"] for item in manifest)
        if collisions:
            raise RuntimeError(f"Refusing to overwrite {len(collisions)} newly annotated messages.")
        rows_by_id = {row["mention_id"]: row for row, _ in selected}
        for item in manifest:
            row = rows_by_id[item["mention_id"]]
            args = json.loads(item["argument_labels_json"])
            store.save_annotation({
                "dataset_key": dataset_key, "mention_id": item["mention_id"],
                "project_id": item["project_id"], "relevance": item["relevance"],
                "stance_status": item["stance_status"], "support_score": item["support_score"] if item["support_score"] != "" else None,
                "other_project_name": None, "annotator": ANNOTATOR, "notes": item["notes"], "needs_review": True,
            }, args)
        print(f"Inserted {len(manifest)} provisional annotations into {store.DB_PATH}")
    else:
        print(f"Dry run: prepared {len(manifest)} provisional annotations; pass --apply to write {store.DB_PATH}")

    print("By selection stratum:", dict(Counter(row["stratum"] for row in manifest)))
    print("By project:", dict(Counter(row["project_id"] for row in manifest)))
    print("By stance status:", dict(Counter(row["stance_status"] for row in manifest)))
    print("Argument label counts:", dict(Counter(arg["class_code"] for item in manifest for arg in json.loads(item["argument_labels_json"]))))
    print("Manifest:", path)
    return manifest, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write these pre-labels into app.py's SQLite database")
    args = parser.parse_args()
    if not SOURCE.is_file():
        print(f"Source dataset not found: {SOURCE}", file=sys.stderr)
        return 2
    raw = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    records = normalize(raw)
    key = dataset_fingerprint(records)
    existing_keys = {row[0] for row in sqlite3.connect(store.DB_PATH).execute("SELECT DISTINCT dataset_key FROM annotations")}
    if existing_keys and key not in existing_keys:
        print(f"Dataset key {key} does not match annotations DB keys {sorted(existing_keys)}; refusing to write.", file=sys.stderr)
        return 3
    write_seed(records, key, args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
