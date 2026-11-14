"""Build the topic and argument overview from stored label records."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re

import plotly.graph_objects as go


COLORS = (
    "#007c70", "#5854a4", "#c55338", "#256da6", "#9d4c82",
    "#668119", "#b5771f", "#396a78", "#aa454d", "#3a8367",
)
GREY = "#526173"
GRID = "#e3eaf1"
GENERAL = "__general__"
ALL_PROJECTS = "__all__"


def _figure(title: str, height: int = 360) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=dict(text=title, x=0.025, xanchor="left", font=dict(size=18)),
        template="plotly_white",
        height=height,
        margin=dict(l=24, r=24, t=62, b=35),
        font=dict(family="Inter, system-ui, sans-serif", color="#243348", size=13),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hoverlabel=dict(align="left"),
    )
    return figure


def _empty(title: str, message: str) -> go.Figure:
    figure = _figure(title, 280)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper",
        text=message, showarrow=False, font=dict(size=14, color=GREY),
    )
    return figure


def _source_group(record: dict) -> str:
    media = str(record.get("media_type_name") or record.get("media_type") or "").casefold()
    media = media.replace("_", " ").replace("-", " ")
    source = str(record.get("source_name") or record.get("source") or "").casefold()
    url = str(record.get("url") or "").casefold()
    youtube = any("youtube" in item or "youtu.be" in item for item in (media, source, url))
    if youtube:
        if "comment" in media or "kommentar" in media or "lc=" in url:
            return "YouTube comments"
        return "YouTube videos"
    if "press release" in media or "pressemitteilung" in media:
        return "Press releases"
    if "news" in media or media in {"journal", "magazine", "newspaper", "media"}:
        return "News / media"
    if "forum" in media:
        return "Forums"
    if "blog" in media and "microblog" not in media:
        return "Blogs"
    if any(word in media for word in ("social", "microblog", "instagram", "facebook", "tiktok")):
        return "Social media"
    if "comment" in media or "kommentar" in media:
        return "Other comments"
    if media in {"video", "videos", "pictures", "images", "photos"}:
        return "Other audio / visual"
    if media in {"", "unknown", "none", "nan"}:
        return "Unknown source type"
    return "Other sources"


def _date_bucket(raw, granularity: str) -> str | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    iso_date = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", raw.replace("Z", "+00:00"))
    try:
        date = datetime.fromisoformat(iso_date)
    except ValueError:
        try:
            date = parsedate_to_datetime(raw)
        except (ValueError, OverflowError, TypeError):
            return None
    if date is None:
        return None
    if date.tzinfo is not None:
        try:
            date = date.astimezone(timezone.utc)
        except OverflowError:
            return None
    month = ((date.month - 1) // 3) * 3 + 1 if granularity == "quarter" else date.month
    return f"{date.year:04d}-{month:02d}-01"


def _label_source(annotation: dict) -> str:
    if annotation.get("annotator") == "codex_rule_prelabel_v1":
        return "legacy_rule"
    return str(annotation.get("label_source") or "human")


def _valid_score(annotation: dict) -> int | None:
    if annotation.get("stance_status") != "scored":
        return None
    try:
        number = float(annotation["support_score"])
    except (KeyError, ValueError, TypeError):
        return None
    return int(number) if 0 <= number <= 5 and number.is_integer() else None


def _share_chart(
    title: str, grouped: dict, names: dict, denominator: int, unit: str,
) -> go.Figure:
    if not denominator:
        return _empty(title, f"No messages with {unit} labels match the current filters.")
    entries = sorted(
        ((names.get(key, key), len(messages)) for key, messages in grouped.items()),
        key=lambda item: (item[1], item[0].casefold()),
    )
    labels, counts = zip(*entries)
    percentages = [100 * n / denominator for n in counts]
    figure = _figure(title, max(295, len(entries) * 27 + 105))
    figure.add_bar(
        x=percentages, y=labels, orientation="h", marker_color=COLORS[0],
        text=[f"{p:.1f}%" for p in percentages], textposition="outside", cliponaxis=False,
        customdata=[(n, denominator) for n in counts],
        hovertemplate=(
            "%{y}<br>%{x:.1f}% · %{customdata[0]:,} of %{customdata[1]:,} messages"
            "<extra></extra>"
        ),
    )
    figure.update_xaxes(
        range=[0, min(116, max(10, max(percentages) * 1.14))],
        ticksuffix="%", gridcolor=GRID, title="Share of labeled messages",
    )
    figure.update_yaxes(automargin=True)
    return figure


def _mean_and_counts(items: list[tuple[int, str]]) -> tuple[float, int, int, int]:
    count = len(items)
    human = sum(source == "human" for _, source in items)
    return sum(score for score, _ in items) / count, count, human, count - human


def _month_index(period: str) -> int:
    return int(period[:4]) * 12 + int(period[5:7]) - 1


def _score_time_chart(grouped: dict, project_names: dict, granularity: str) -> go.Figure:
    title = "Average support score over time"
    if not grouped:
        return _empty(title, "No dated support scores match the current filters.")
    figure = _figure(title, 415)
    step = 3 if granularity == "quarter" else 1
    for palette_index, project_id in enumerate(sorted(grouped, key=lambda key: project_names.get(key, key).casefold())):
        label = project_names.get(project_id, project_id)
        periods = grouped[project_id]
        x, y, counts = [], [], []
        previous = None
        for period, items in sorted(periods.items()):
            month_index = _month_index(period)
            if previous is not None and month_index - previous > step:
                year, month = divmod(previous + step, 12)
                x.append(f"{year:04d}-{month + 1:02d}-01")
                y.append(None)
                counts.append((0, 0, 0))
            mean, count, human, keyword = _mean_and_counts(items)
            x.append(period)
            y.append(mean)
            counts.append((count, human, keyword))
            previous = month_index
        color = COLORS[palette_index % len(COLORS)]
        figure.add_scatter(
            x=x, y=y, customdata=counts, name=label, mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=7, color=color),
            connectgaps=False,
            hovertemplate=(
                f"{label}<br>%{{x|%b %Y}}<br>Mean %{{y:.2f}} / 5"
                "<br>n=%{customdata[0]:,} · human %{customdata[1]:,} · keyword %{customdata[2]:,}"
                "<extra></extra>"
            ),
        )
    figure.update_xaxes(
        type="date", title="Quarter starting" if granularity == "quarter" else "Month",
        tickformat="%b %Y", gridcolor=GRID,
    )
    figure.update_yaxes(title="Mean support score", range=[-0.15, 5.15], dtick=1, gridcolor=GRID)
    figure.update_layout(legend=dict(orientation="h", y=-0.22, x=0), margin=dict(b=80))
    return figure


def _score_source_chart(grouped: dict, project_names: dict) -> go.Figure:
    title = "Average support score by source type"
    if not grouped:
        return _empty(title, "No support scores match the current filters.")
    project_ids = sorted(grouped, key=lambda key: project_names.get(key, key).casefold())
    sources = sorted({source for items in grouped.values() for source in items})
    z, labels, counts = [], [], []
    for project_id in project_ids:
        project_means, project_labels, project_counts = [], [], []
        for source in sources:
            items = grouped[project_id].get(source)
            if items:
                mean, count, human, keyword = _mean_and_counts(items)
                project_means.append(mean)
                project_labels.append(f"{mean:.2f}<br>n={count:,}")
                project_counts.append((count, human, keyword))
            else:
                project_means.append(None)
                project_labels.append("")
                project_counts.append((0, 0, 0))
        z.append(project_means)
        labels.append(project_labels)
        counts.append(project_counts)
    figure = _figure(title, max(310, len(project_ids) * 46 + 140))
    figure.add_heatmap(
        x=sources, y=[project_names.get(key, key) for key in project_ids],
        z=z, zmin=0, zmax=5, xgap=3, ygap=3, text=labels, texttemplate="%{text}",
        customdata=counts, hoverongaps=False,
        colorscale=[[0, "#c86851"], [0.5, "#f6f1e2"], [1, "#59a790"]],
        colorbar=dict(title="Mean<br>0–5", tickvals=[0, 1, 2, 3, 4, 5]),
        hovertemplate=(
            "%{y} · %{x}<br>Mean %{z:.2f} / 5"
            "<br>n=%{customdata[0]:,} · human %{customdata[1]:,} · keyword %{customdata[2]:,}"
            "<extra></extra>"
        ),
    )
    figure.update_xaxes(automargin=True, tickangle=-28, showgrid=False)
    figure.update_yaxes(automargin=True, autorange="reversed", showgrid=False)
    return figure


def build_dashboard(
    records, annotations, arguments, projects, argument_classes,
    project_id: str = ALL_PROJECTS, provenance: str = "all", granularity: str = "month",
):
    """Return coverage and four interactive figures for the dashboard tab.

    Topic shares describe the full filtered label pool. The project selector
    applies to arguments and score charts so the topic distribution remains
    useful while inspecting a single project.
    """
    record_by_key = {}
    possible_keys = defaultdict(list)
    for raw_record in records:
        record = dict(raw_record)
        mention = str(record.get("mention_id") or "")
        if mention:
            key = (str(record.get("dataset_key") or ""), mention)
            if key not in record_by_key:
                possible_keys[mention].append(key)
            record_by_key[key] = record

    def find_message(row):
        row_key = (str(row.get("dataset_key") or ""), str(row.get("mention_id") or ""))
        if row_key in record_by_key:
            return row_key
        candidates = possible_keys.get(row_key[1], [])
        if len(candidates) == 1 and (not row_key[0] or not candidates[0][0]):
            return candidates[0]
        return None

    project_names = {str(row["project_id"]): str(row["project_name"]) for row in projects}
    argument_names = {str(row["class_code"]): str(row["class_name"]) for row in argument_classes}
    topics = defaultdict(set)
    argument_groups = defaultdict(set)
    eligible_projects = {}
    time_groups = defaultdict(lambda: defaultdict(list))
    source_groups = defaultdict(lambda: defaultdict(list))
    score_origins = defaultdict(int)
    undated = 0

    for raw_annotation in annotations:
        annotation = dict(raw_annotation)
        source = _label_source(annotation)
        if source not in {"human", "keyword"} or (provenance != "all" and provenance != source):
            continue
        if source == "human" and annotation.get("needs_review") not in (0, "0", None):
            continue
        message = find_message(annotation)
        target = str(annotation.get("project_id") or "")
        relevance = annotation.get("relevance")
        named = bool(target and not target.startswith("__") and relevance == "project_specific")
        general = target == GENERAL and relevance == "general"
        if message is None:
            continue
        if named or general:
            topics[target].add(message)
        if project_id not in {ALL_PROJECTS, "", None, target}:
            continue
        if not (named or general or (source == "keyword" and target == "__unclear__")):
            continue
        eligible_projects[(message, target)] = source
        score = _valid_score(annotation)
        if not named or score is None:
            continue
        record = record_by_key[message]
        source_groups[target][_source_group(record)].append((score, source))
        score_origins[source] += 1
        period = _date_bucket(record.get("published_at") or record.get("timestamp"), granularity)
        if period:
            time_groups[target][period].append((score, source))
        else:
            undated += 1

    for raw_argument in arguments:
        argument = dict(raw_argument)
        message = find_message(argument)
        target = str(argument.get("project_id") or "")
        code = str(argument.get("class_code") or "")
        source = eligible_projects.get((message, target))
        if source and code and argument.get("label_source", source) in (source, None):
            argument_groups[code].add(message)

    topic_total = len(set().union(*topics.values())) if topics else 0
    argument_total = len(set().union(*argument_groups.values())) if argument_groups else 0
    corpus_total = len(record_by_key)
    score_total = sum(score_origins.values())
    topic_scope = "all topics" if project_id in {ALL_PROJECTS, "", None} else project_names.get(project_id, project_id)
    coverage = (
        f"**{corpus_total:,}** messages in the combined dataset · "
        f"**{topic_total:,}** with a usable topic label · "
        f"**{argument_total:,}** messages with an argument label · "
        f"**{score_total:,}** support ratings ({score_origins['human']:,} human, {score_origins['keyword']:,} keyword)"
        "\n\n"
        "Topic and argument percentages use unique messages with a label in that chart. "
        f"A message can have several labels, so the percentages can add up to more than 100%. "
        f"Arguments and ratings are scoped to **{topic_scope}**. "
        "Score charts average available ratings on the 0–5 scale and show the sample size for each group."
    )
    if undated:
        coverage += f" The time chart omits {undated:,} ratings without a usable date."
    return (
        coverage,
        _share_chart("Topic share", topics, project_names, topic_total, "topic"),
        _share_chart("Argument share", argument_groups, argument_names, argument_total, "argument"),
        _score_time_chart(time_groups, project_names, granularity),
        _score_source_chart(source_groups, project_names),
    )
