"""Keyword suggestions for the cable car workbench."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from html import unescape
from typing import Any


def words(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)


CABLE = words(
    r"[a-zäöüß]{0,24}seilbahn\w*|kabinenbahn\w*|gondelbahn\w*|kabelbahn\w*|gondel[ns]?|"
    r"t[ée]l[ée]ph[ée]riqu\w*|t[ée]l[ée]cabin\w*|funiculair\w*|funiculars?|transport par c[âa]ble|"
    r"c[âa]ble[\s-]*c\s*1|cable[\s-]*cars?|cableways?|ropeways?|"
    r"gondola(?:[\s-]*lifts?)?|aerial[\s-]*(?:tramways?|trams?|lifts?)"
)
HTML_TAG = re.compile(r"</?[A-Za-z][^<>\n]{0,300}>")
URL = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
WORDS_TO_IGNORE = re.compile(r"\b(?:impressum|datenschutz|privacy policy|politique de confidentialité)\b", re.I)
NEWS_MEDIA = re.compile(r"news|zeitung|press|presse|magazine|radio|television|\bprint\b|\barticle\b", re.I)

# Specific routes take precedence over the default project for that city.
PLACES = re.compile(
    r"\b(?P<heilbronn>heilbronn\w*|ipai|neckarbogen)\b|"
    r"\b(?P<stuttgart>stuttgart\w*|heslach|südheimer platz|suedheimer platz)\b|"
    r"\b(?P<bonn>bonn(?:er|ers)?|venusberg)\b|"
    r"\b(?P<paris>c[âa]ble[\s-]*c\s*1|cr[ée]teil|villeneuve[\s-]+saint[\s-]+georges|"
    r"limeil[\s-]+br[ée]vannes)\b|"
    r"\b(?P<cologne>köln\w*|koeln\w*|cologne)\b|"
    r"\b(?P<hamburg>hamburg\w*|oberbillwerder)\b|"
    r"\b(?P<wuppertal>wuppertal\w*)\b|"
    r"\b(?P<passau>passau\w*|veste oberhaus)\b|"
    r"\b(?P<koblenz>koblenz\w*|ehrenbreitstein)\b|"
    r"\b(?P<duisburg>duisburg\w*)\b|"
    r"\b(?P<herne>herne(?:r)?|blumenthal)\b|"
    r"\b(?P<kassel>kassel\w*|bergpark wilhelmshöhe)\b",
    re.IGNORECASE,
)
PROJECT_DEFAULTS = {
    "heilbronn": "heilbronn_cable_car",
    "stuttgart": "stuttgart_urban_cable_car",
    "bonn": "bonn_cable_car",
    "paris": "paris_cable_c1",
    "wuppertal": "wuppertal_cable_car",
    "passau": "passau_oberhaus_cable_car",
    "koblenz": "koblenz_cable_car",
    "duisburg": "duisburg_urban_cable_car",
    "herne": "herne_blumenthal_cable_car",
    "kassel": "kassel_herkules_cable_car",
}
ROUTES = {
    "stuttgart": (
        (words(r"waldfriedhof\w*|heslach|südheimer platz|suedheimer platz"), "stuttgart_waldfriedhof_funicular"),
    ),
    "bonn": (
        (words(r"nordbrück\w*|friedrich[\s-]+ebert[\s-]+brück\w*"), "bonn_north_bridge_temporary_cable_car"),
    ),
    "cologne": (
        (words(r"rheinseilbahn\w*|rhein[\s-]+seilbahn\w*|river rhine|zoobrück\w*"), "cologne_rhine_cable_car"),
        (words(r"projekt\w*|geplant\w*|vorschlag\w*|projektier\w*|proposal|urban\w*"), "cologne_urban_cable_car"),
    ),
    "hamburg": (
        (words(r"oberbillwerder"), "hamburg_oberbillwerder_cable_car"),
        (words(r"elbe|elbseilbahn\w*|hafencity|st\.? pauli"), "hamburg_elbe_cable_car"),
    ),
    "wuppertal": (
        (words(r"buga|bundesgartenschau|2031"), "wuppertal_buga_2031_cable_car"),
        (words(r"universit[äa]t\w*|campus grifflenberg|university campus"), "wuppertal_university_cable_car"),
    ),
}

ARGUMENT_KEYWORDS = {
    "cost_finance": words(r"kost(?:en\w*|et|spielig\w*)|budget\w*|finanzier\w*|zu teuer|fahrpreis\w*|"
                           r"co[ûu]t\w*|financement\w*|trop cher|prix du billet|costs?|funding|fares?|too expensive"),
    "time_convenience": words(r"fahrzeit\w*|zeiterspar\w*|reisezeit\w*|schneller|umweg\w*|bequem\w*|"
                               r"temps de trajet|gain de temps|plus rapide|plus pratique|travel time|faster|convenien\w*"),
    "accessibility_connectivity": words(r"barrierefrei\w*|anbindung\w*|umstieg\w*|rollstuhl\w*|verkehrsnetz\w*|"
                                         r"accessibilit[ée]|accessible\w*|correspondance\w*|desserte|"
                                         r"wheelchair\w*|step[\s-]*free|accessib\w*|transit connections?"),
    "capacity_reliability": words(r"kapazit[äa]t\w*|zuverl[äa]ssig\w*|betriebsausfall\w*|windanf[äa]llig\w*|"
                                    r"betriebsunterbrech\w*|capacit[ée]\w*|fiabilit[ée]|panne\w*|fr[ée]quence\w*|"
                                    r"capacity|reliab\w*|breakdowns?|wind closures?"),
    "climate_emissions": words(r"klimaschutz|klimafreund\w*|co[2₂]|emission\w*|treibhausgas\w*|"
                                 r"[ée]mission\w*|gaz [àa] effet de serre|empreinte carbone|"
                                 r"climate impact|carbon emissions?|greenhouse gas\w*"),
    "nature_landscape": words(r"umweltschutz|umweltbelast\w*|naturschutz|artenschutz|baumf[äa]ll\w*|b[äa]ume|"
                                r"landschaftsbild|waldrodung\w*|biodiversit\w*|protection de l[’']environnement|"
                                r"abattage d[’']arbres|arbres?|abattre des arbres|impact environnemental|"
                                r"habitat loss|environmental impact|conservation area|tree removal|cutting down trees"),
    "noise_vibration": words(r"l[äa]rm\w*|ger[äa]uschbelast\w*|schall\w*|ersch[üu]tter\w*|bruit\w*|"
                               r"nuisances? sonores?|vibrations?|noise|noisy"),
    "privacy_visual": words(r"privatsph[äa]re|sichtachse\w*|einblick\w*|verschandel\w*|sichtschutz|"
                              r"vie priv[ée]e|vis[àa][\s-]*vis|impact visuel|atteinte visuelle|"
                              r"privacy|visual impact|overlook(?:s|ing)? (?:houses|homes|gardens)"),
    "economic_development": words(r"arbeitspl[äa]tz\w*|touris\w*|wirtschaftswachstum|"
                                    r"retomb[ée]es? [ée]conomiques?|emplois?|commerces?|"
                                    r"job creation|economic growth|local business\w*"),
    "equity_distribution": words(r"verteilungsgerechtigkeit|sozial\w* gerechtigkeit|ungleichheit\w*|"
                                 r"benachteil\w*|[ée]quit[ée]|in[ée]galit[ée]\w*|justice sociale|"
                                 r"social equity|unequal access|lower[\s-]+income"),
    "safety": words(r"sicherheit\w*|unf[äa]ll\w*|unfall\w*|rettungskonzept\w*|"
                     r"s[ée]curit[ée]|accident\w*|[ée]vacu\w*|risque de collision|safety|evacu\w*|accidents?"),
    "construction_disruption": words(r"baustellen?\w*|bauarbeiten|bauphase\w*|verkehrssperr\w*|"
                                      r"chantier\w*|travaux|perturbations? li[ée]es? aux travaux|"
                                      r"construction disruption|road closures?|construction works?"),
    "governance_participation": words(r"b[üu]rgerbeteilig\w*|b[üu]rgerentscheid\w*|b[üu]rgerbefrag\w*|"
                                       r"transparenz|gemeinderat|abstimmung\w*|consultation publique|"
                                       r"concertation\w*|participation citoyenne|r[ée]f[ée]rendum|"
                                       r"public consultation|public participation|referendums?"),
    "technical_feasibility": words(r"technisch\w* machbar\w*|machbarkeitsstudie\w*|trassenf[üu]hrung\w*|"
                                   r"st[üu]tzenh[öo]he\w*|faisabilit[ée]|[ée]tude technique|"
                                   r"feasibility|technical constraints?|route alignment"),
}
DIRECTION_CUES = {
    "cost_finance": (words(r"kosteng[üu]nstig\w*|spart geld|[ée]conomise\w* de l[’']argent|cost[\s-]+effective"),
                     words(r"zu teuer|kostenexplosion\w*|trop cher|too expensive|cost overrun\w*")),
    "nature_landscape": (words(r"schont die natur|sch[üu]tzt die umwelt|prot[èe]ge l[’']environnement|protects? habitats?"),
                          words(r"b[äa]ume f[äa]llen|waldrodung\w*|abattage d[’']arbres|tree removal|habitat loss")),
    "noise_vibration": (words(r"weniger l[äa]rm|moins de bruit|reduces? noise|quieter"),
                       words(r"l[äa]rmbelastung\w*|mehr l[äa]rm|nuisances? sonores?|more noise|noise pollution")),
    "time_convenience": (words(r"fahrzeitverk[üu]rz\w*|zeiterspar\w*|schneller|gain de temps|saves? time|faster"),
                         words(r"l[äa]ngere fahrzeit|plus lent|slower|longer travel time")),
}

STANCE_OPPOSE = words(
    r"ich\s+(?:bin\s+(?:(?:eher|klar|strikt|entschieden|absolut)\s+)?gegen|lehne)|"
    r"je\s+(?:suis\s+(?:(?:plut[ôo]t|fermement|totalement)\s+)?(?:contre|oppos[ée]e?\s+[àa])|m[’']oppose\s+[àa])|"
    r"i\s+(?:(?:strongly|firmly|somewhat)\s+)?(?:oppose|am\s+(?:(?:strongly|firmly|somewhat)\s+)?(?:against|opposed to))"
)
STANCE_SUPPORT = words(
    r"ich\s+(?:bin\s+(?:(?:eher|klar|strikt|entschieden|absolut)\s+)?f[üu]r|unterst[üu]tze|bef[üu]rworte)|"
    r"je\s+(?:suis\s+(?:(?:plut[ôo]t|fermement|totalement)\s+)?(?:pour|favorable\s+[àa])|soutiens)|"
    r"i\s+(?:(?:strongly|firmly|somewhat)\s+)?(?:support|am\s+(?:(?:strongly|firmly|somewhat)\s+)?(?:in favou?r of|for))"
)
STANCE_INTENSE = words(r"strikt|entschieden|absolut|fermement|totalement|strongly|firmly")
STANCE_HEDGED = words(r"eher|plut[ôo]t|somewhat")
STANCE_CONDITIONAL = words(r"falls|sofern|vorausgesetzt|nur wenn|wenn|[àa] condition|seulement si|only if|provided that|if")
STANCE_SARCASM = re.compile(r"[🙄🤡]|/s\b|\b(?:sarcasm|satire|sarkasmus|ironie|irony)\b", re.I)


def _project_ids(projects: Mapping | Iterable) -> set[str]:
    if isinstance(projects, Mapping):
        return set(projects)
    return {project if isinstance(project, str) else project["project_id"] for project in projects}


def _argument_codes(argument_codes: Iterable) -> set[str]:
    return {code if isinstance(code, str) else code["class_code"] for code in argument_codes}


def _clean(text: Any) -> str:
    # Preserve line breaks between body paragraphs and discard URL fragments.
    text = unescape(str(text or ""))
    return URL.sub(" ", HTML_TAG.sub(" ", text))


def _projects_in_context(context: str) -> set[str]:
    cities = {match.lastgroup for match in PLACES.finditer(context)}
    result = set()
    for city in cities:
        routes = {project for pattern, project in ROUTES.get(city, ()) if pattern.search(context)}
        if routes:
            result.update(routes)
        elif city in PROJECT_DEFAULTS:
            result.add(PROJECT_DEFAULTS[city])
    return result


def _stance(sentence: str) -> tuple[str, int | None]:
    if any(mark in sentence for mark in ('"', '„', '“', '”', '«', '»')) or STANCE_SARCASM.search(sentence):
        return "unclear", None
    positions = []
    for pattern, direction in ((STANCE_OPPOSE, "con"), (STANCE_SUPPORT, "pro")):
        for expression in pattern.finditer(sentence):
            target = sentence[expression.end():expression.end() + 62]
            cable = CABLE.search(target)
            if not cable or re.search(r"[,;:!?]|\b(?:nicht|pas|not|never)\b", target[:cable.start()], re.I):
                continue
            if STANCE_CONDITIONAL.search(sentence):
                return "mixed_or_conditional", None
            intensity = expression.group()
            if intensity.casefold().endswith("lehne") and not re.search(r"\bab\b", target[cable.end():], re.I):
                continue
            following_clause = re.split(r"[.;:!?]", target[cable.end():], 1)[0]
            if re.search(r"\b(?:nicht|pas|not|never)\b", following_clause, re.I):
                continue
            if direction == "con":
                score = 0 if STANCE_INTENSE.search(intensity) else 2 if STANCE_HEDGED.search(intensity) else 1
            else:
                score = 5 if STANCE_INTENSE.search(intensity) else 3 if STANCE_HEDGED.search(intensity) else 4
            positions.append(score)
    if any(score <= 2 for score in positions) and any(score >= 3 for score in positions):
        return "mixed_or_conditional", None
    if positions:
        return "scored", positions[0]
    return "unclear", None


def classify(record: Mapping[str, Any], projects: Mapping | Iterable, argument_codes: Iterable) -> dict[str, Any]:
    """Return a single project's keyword suggestions and identify incomplete items."""
    text = "\n".join((_clean(record.get("title")), _clean(record.get("content"))))
    known_projects = _project_ids(projects)
    allowed_arguments = _argument_codes(argument_codes)
    media_type = str(record.get("media_type_name") or record.get("media_type") or "")
    may_score = not NEWS_MEDIA.search(media_type)
    targets = set()
    stance_scores = []
    conditional_stance = False
    argument_evidence: dict[str, tuple[str, set[str]]] = {}
    has_cable_context = False

    for cable in CABLE.finditer(text):
        start, end = cable.span()
        context = text[max(0, start - 120): min(len(text), end + 120)]
        if WORDS_TO_IGNORE.search(context):
            continue
        has_cable_context = True
        # A city farther away in the document may describe a comparison or a byline.
        targets.update(_projects_in_context(text[max(0, start - 80): end + 80]))
        for code, keyword in ARGUMENT_KEYWORDS.items():
            if code not in allowed_arguments:
                continue
            match = keyword.search(context)
            if not match:
                continue
            _, directions = argument_evidence.setdefault(code, (match.group(), set()))
            if code in DIRECTION_CUES:
                positive, negative = DIRECTION_CUES[code]
                local = context[max(0, match.start() - 24):match.end() + 24]
                if positive.search(local):
                    directions.add("pro")
                if negative.search(local):
                    directions.add("con")
        if may_score:
            # Only explicit positions tied to a cable car in the same sentence get a score.
            merged = text[max(0, start - 125): end + 65]
            for sentence in SENTENCE_BREAK.split(merged):
                if not CABLE.search(sentence) or not (STANCE_SUPPORT.search(sentence) or STANCE_OPPOSE.search(sentence)):
                    continue
                status, score = _stance(sentence)
                if score is not None:
                    stance_scores.append(score)
                conditional_stance = conditional_stance or status == "mixed_or_conditional"

    project_id = "__unclear__"
    if len(targets) == 1:
        candidate = next(iter(targets))
        if candidate in known_projects:
            project_id = candidate

    status = "unclear"
    score = None
    if project_id != "__unclear__":
        if conditional_stance or (stance_scores and min(stance_scores) <= 2 < max(stance_scores)):
            status = "mixed_or_conditional"
        elif stance_scores:
            status, score = "scored", stance_scores[0]

    arguments = []
    for code, (evidence, directions) in argument_evidence.items():
        direction = "mixed" if len(directions) > 1 else next(iter(directions), "unclear")
        arguments.append({"class_code": code, "direction": direction, "evidence_quote": evidence, "argument_summary": ""})

    gaps = []
    if project_id == "__unclear__":
        if len(targets) > 1:
            gaps.append("multiple project candidates: " + ", ".join(sorted(targets)))
        elif targets:
            gaps.append("project candidate is missing from the project index: " + next(iter(targets)))
        else:
            gaps.append("no project keyword close to a cable car mention" if has_cable_context else "no cable car keyword")
    if not arguments:
        gaps.append("no argument keyword near a cable car mention")
    if score is None:
        gaps.append("support score needs interpretation")
    return {
        "project_id": project_id,
        "relevance": "unclear" if project_id == "__unclear__" else "project_specific",
        "stance_status": status,
        "support_score": score,
        "arguments": arguments,
        "needs_review": bool(gaps),
        "notes": "Keyword suggestions. " + ("Review: " + "; ".join(gaps) + "." if gaps else "Project, arguments and explicit stance found."),
    }
