"""Fast, local baseline models for project, argument and stance labels.

The first version deliberately uses scikit-learn TF-IDF so it works without
downloading a neural checkpoint. A multilingual SetFit encoder can replace the
vectorizer once enough human labels and an approved model download are available.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"


def make_text(record: dict[str, Any], project_name: str | None = None) -> str:
    title = str(record.get("title") or "").strip()
    content = str(record.get("content") or "").strip()
    text = f"Title: {title}\nContent: {content}".strip()
    if project_name:
        text = f"Target project: {project_name}\n{text}"
    # Avoid one huge news article dominating runtime. Keep its beginning and end;
    # both commonly carry the topic and conclusion in this monitoring corpus.
    limit = 30000
    if len(text) > limit:
        text = text[:15000] + "\n[... middle omitted for baseline model ...]\n" + text[-15000:]
    return text


def _vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=180000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    lowercase=True,
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=220000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


class MultiLabelTextModel:
    """Shared TF-IDF representation with one binary head per label."""

    def __init__(self) -> None:
        self.vectorizer = _vectorizer()
        self.classifiers: dict[str, LogisticRegression] = {}

    def fit(self, texts: list[str], label_sets: list[set[str]], min_positive: int = 2) -> "MultiLabelTextModel":
        x = self.vectorizer.fit_transform(texts)
        all_labels = sorted(set().union(*label_sets)) if label_sets else []
        for label in all_labels:
            y = np.asarray([int(label in labels) for labels in label_sets], dtype=int)
            positives = int(y.sum())
            negatives = int(len(y) - positives)
            if positives < min_positive or negatives < 2:
                continue
            clf = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")
            clf.fit(x, y)
            self.classifiers[label] = clf
        if not self.classifiers:
            raise ValueError("Not enough examples to train any label. Add more reviewed examples per class.")
        return self

    @property
    def labels(self) -> list[str]:
        return sorted(self.classifiers)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        labels = self.labels
        if not labels:
            return np.zeros((len(texts), 0), dtype=float)
        return np.column_stack([self.classifiers[label].predict_proba(x)[:, 1] for label in labels])


class OrdinalTextModel:
    """Cumulative binary model for scores 0..5; adjacent scores stay ordered."""

    def __init__(self) -> None:
        self.vectorizer = _vectorizer()
        self.threshold_models: list[LogisticRegression | float] = []

    def fit(self, texts: list[str], scores: list[int]) -> "OrdinalTextModel":
        if len(set(scores)) < 2:
            raise ValueError("Support model needs examples from at least two score levels.")
        x = self.vectorizer.fit_transform(texts)
        for threshold in range(1, 6):
            y = np.asarray([int(score >= threshold) for score in scores], dtype=int)
            if len(np.unique(y)) == 1:
                self.threshold_models.append(float(y[0]))
            else:
                clf = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")
                clf.fit(x, y)
                self.threshold_models.append(clf)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        ge = []
        for model in self.threshold_models:
            if isinstance(model, float):
                ge.append(np.full(len(texts), model, dtype=float))
            else:
                ge.append(model.predict_proba(x)[:, 1])
        # The five probabilities P(score >= k) should decrease as k increases.
        q = np.minimum.accumulate(np.column_stack(ge), axis=1)
        probs = np.zeros((len(texts), 6), dtype=float)
        probs[:, 0] = 1.0 - q[:, 0]
        for score in range(1, 5):
            probs[:, score] = q[:, score - 1] - q[:, score]
        probs[:, 5] = q[:, 4]
        probs = np.clip(probs, 0.0, 1.0)
        probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        return probs


def train_models(
    records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    argument_rows: list[dict[str, Any]],
    project_names: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_id = {str(r["mention_id"]): r for r in records}
    project_labels: dict[str, set[str]] = {}
    for row in annotations:
        pid = str(row["project_id"])
        if not pid.startswith("__") and row.get("relevance") == "project_specific":
            project_labels.setdefault(str(row["mention_id"]), set()).add(pid)
        elif row.get("relevance") in {"general", "unrelated"}:
            # Human-confirmed general/off-topic records are useful negatives
            # when only one named project has been labeled so far.
            project_labels.setdefault(str(row["mention_id"]), set())

    project_ids = sorted({label for labels in project_labels.values() for label in labels})
    project_texts = []
    project_y = []
    for mention_id, labels in project_labels.items():
        record = by_id.get(mention_id)
        if record:
            project_texts.append(make_text(record))
            project_y.append(labels)

    models: dict[str, Any] = {}
    stats: dict[str, Any] = {"project": {}, "arguments": {}, "support": {}}
    if len(project_ids) >= 1 and project_texts:
        try:
            model = MultiLabelTextModel()
            model.fit(project_texts, project_y, min_positive=2)
            models["project"] = model
            stats["project"] = {"examples": len(project_texts), "classes": model.labels}
        except ValueError as exc:
            stats["project"] = {"skipped": str(exc)}
    else:
        stats["project"] = {"skipped": "Need human labels for at least one named project."}

    args_by_pair: dict[tuple[str, str], set[str]] = {}
    for row in argument_rows:
        direction = str(row.get("direction") or "unclear")
        args_by_pair.setdefault((str(row["mention_id"]), str(row["project_id"])), set()).add(
            f"{row['class_code']}::{direction}"
        )
    argument_texts: list[str] = []
    argument_y: list[set[str]] = []
    for row in annotations:
        mention_id, project_id = str(row["mention_id"]), str(row["project_id"])
        record = by_id.get(mention_id)
        if record and row.get("relevance") == "project_specific":
            argument_texts.append(make_text(record, project_names.get(project_id, project_id)))
            argument_y.append(args_by_pair.get((mention_id, project_id), set()))
    if argument_texts:
        try:
            model = MultiLabelTextModel()
            model.fit(argument_texts, argument_y, min_positive=2)
            models["arguments"] = model
            stats["arguments"] = {"examples": len(argument_texts), "classes": model.labels}
        except ValueError as exc:
            stats["arguments"] = {"skipped": str(exc)}
    else:
        stats["arguments"] = {"skipped": "Need project-specific human annotations with argument labels."}

    support_texts: list[str] = []
    support_scores: list[int] = []
    for row in annotations:
        score = row.get("support_score")
        mention_id, project_id = str(row["mention_id"]), str(row["project_id"])
        record = by_id.get(mention_id)
        if record and row.get("relevance") == "project_specific" and score is not None:
            support_texts.append(make_text(record, project_names.get(project_id, project_id)))
            support_scores.append(int(score))
    if support_texts and len(set(support_scores)) >= 2:
        model = OrdinalTextModel().fit(support_texts, support_scores)
        models["support"] = model
        stats["support"] = {
            "examples": len(support_texts),
            "score_counts": {str(i): support_scores.count(i) for i in range(6) if i in support_scores},
        }
    else:
        stats["support"] = {"skipped": "Need scored project-specific examples from at least two score levels."}

    return models, stats


def save_models(models: dict[str, Any], dataset_key: str, summary: dict[str, Any]) -> tuple[str, str]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = f"tfidf-baseline-{stamp}"
    path = MODEL_DIR / f"{dataset_key[:16]}-{stamp}.joblib"
    bundle = {"version": version, "dataset_key": dataset_key, "models": models, "summary": summary}
    joblib.dump(bundle, path, compress=3)
    (MODEL_DIR / f"{dataset_key[:16]}-latest.json").write_text(
        json.dumps({"version": version, "path": str(path), "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path), version


def load_latest_models(dataset_key: str) -> dict[str, Any] | None:
    latest_path = MODEL_DIR / f"{dataset_key[:16]}-latest.json"
    if not latest_path.exists():
        return None
    metadata = json.loads(latest_path.read_text(encoding="utf-8"))
    path = Path(metadata["path"])
    if not path.exists():
        return None
    return joblib.load(path)


def infer_records(
    records: list[dict[str, Any]],
    models: dict[str, Any],
    project_names: dict[str, str],
    project_threshold: float = 0.40,
    argument_threshold: float = 0.45,
    support_review_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    """Create separate, reviewable predictions. Never changes human labels."""
    output: list[dict[str, Any]] = []
    pmodel = models.get("project")
    if not pmodel:
        return output
    project_probabilities: list[np.ndarray] = []
    batch_size = 256
    for start in range(0, len(records), batch_size):
        project_texts = [make_text(r) for r in records[start : start + batch_size]]
        project_probabilities.extend(pmodel.predict_proba(project_texts))
    target_items: list[tuple[dict[str, Any], str, float]] = []
    for row_idx, record in enumerate(records):
        probabilities = project_probabilities[row_idx]
        if len(probabilities) == 0:
            continue
        project_candidates = [
            (pid, float(probabilities[j])) for j, pid in enumerate(pmodel.labels)
            if probabilities[j] >= project_threshold
        ]
        if not project_candidates:
            best = int(np.argmax(probabilities))
            project_candidates = [(pmodel.labels[best], float(probabilities[best]))]
        project_candidates.sort(key=lambda x: x[1], reverse=True)
        top_project, top_conf = project_candidates[0]
        output.append({
            "mention_id": str(record["mention_id"]), "task": "project_index", "project_id": "",
            "result": {
                "project_candidates": [
                    {"project_id": pid, "project_name": project_names.get(pid, pid), "confidence": conf}
                    for pid, conf in project_candidates
                ]
            },
            "confidence": top_conf, "review_required": top_conf < project_threshold or len(project_candidates) > 1,
        })
        for project_id, pconf in project_candidates:
            target_items.append((record, project_id, pconf))

    # Keep working memory bounded when the unlabeled pool is large.
    for start in range(0, len(target_items), batch_size):
        batch_items = target_items[start : start + batch_size]
        target_texts = [make_text(record, project_names.get(project_id, project_id)) for record, project_id, _ in batch_items]
        if "support" in models:
            support_matrix = models["support"].predict_proba(target_texts)
            for (record, project_id, _), distribution in zip(batch_items, support_matrix):
                expected = float(np.dot(distribution, np.arange(6)))
                score = int(np.clip(np.rint(expected), 0, 5))
                confidence = float(distribution[score])
                output.append({
                    "mention_id": str(record["mention_id"]), "task": "support_score", "project_id": project_id,
                    "result": {"support_score_0_5": score, "score_distribution": [float(x) for x in distribution]},
                    "confidence": confidence, "review_required": confidence < support_review_threshold,
                })
        if "arguments" in models:
            model = models["arguments"]
            argument_matrix = model.predict_proba(target_texts)
            for (record, project_id, _), probs in zip(batch_items, argument_matrix):
                predicted = [
                    {
                        "class_code": label.rsplit("::", 1)[0],
                        "direction": label.rsplit("::", 1)[1] if "::" in label else "unclear",
                        "class_name": label.rsplit("::", 1)[0].replace("_", " "),
                        "confidence": float(prob),
                    }
                    for label, prob in zip(model.labels, probs) if prob >= argument_threshold
                ]
                max_conf = float(np.max(probs)) if len(probs) else 0.0
                output.append({
                    "mention_id": str(record["mention_id"]), "task": "argument_types", "project_id": project_id,
                    "result": {"arguments": predicted}, "confidence": max_conf,
                    "review_required": not predicted or max_conf < argument_threshold,
                })
    return output


def pack_prediction_rows(dataset_key: str, model_version: str, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for prediction in predictions:
        rows.append({
            "mention_id": prediction["mention_id"],
            "task": prediction["task"],
            "project_id": prediction.get("project_id", ""),
            "result_json": json.dumps(prediction["result"], ensure_ascii=False),
            "confidence": prediction.get("confidence"),
            "review_required": prediction.get("review_required", True),
        })
    return rows

