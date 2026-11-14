# Seilbahn

**Seilbahn** is a collaborative dashboard for exploring cable-car project discussions, reviewing message labels, and tracking how arguments and expressed support change over time. It combines the existing labeling workbench with keyword-assisted imports, an optional local crawler, charts, model tools, and dataset export.

## What it does

- Shows project and argument coverage, support scores over time, and differences by source/media type.
- Imports crawler-format JSON datasets and adds one-off records through a form.
- Assigns conservative project, argument, and support suggestions with multilingual keyword rules. Ambiguous items go to **Manual labeling → Needs review**.
- Runs the bundled Heilbronn crawler in the background while the web app is running. It runs once at startup and then on a configurable interval.
- Saves records, labels, keyword results, and removal history in SQLite. Imported records therefore survive an app restart.
- Lets reviewers remove empty or irrelevant messages from the active dataset and restore them later.
- Exports labels, arguments, RAG-friendly JSONL, a combined active JSON dataset, and a consistent SQLite backup.
- Includes a local TF-IDF model baseline and a separate prediction-review utility.

Keyword labels are rule-based suggestions, not verified public-opinion measurements. They are shown separately from human-confirmed labels. The support score describes the position expressed in a message; it is not a representative poll of the population.

## Repository layout

```text
Seilbahn/
├── app.py                      # Main Gradio dashboard
├── store.py                    # SQLite schema and persistence
├── workflow.py                 # Import, deduplication and keyword-label flow
├── keyword_labels.py           # Conservative multilingual rules
├── dashboard_charts.py         # Plotly figures
├── model_pipeline.py           # Local TF-IDF training and predictions
├── crawler/
│   └── scraper.py              # Bundled Heilbronn news and public-source crawler
├── tools/
│   ├── review_predictions.py   # Separate blind prediction-audit UI
│   └── seed_provisional_labels.py # Optional legacy rule-based seed utility
├── data/                       # Created locally; database and data are ignored by Git
├── models/                     # Local model artifacts; ignored by Git
├── requirements.txt
└── .env.example
```

The source corpus, SQLite databases, crawler snapshots, imports, exports, credentials, and trained model files are intentionally excluded from Git. Each team should provide its own data and keep its database backed up.

## Quick start

Use Python 3.10 or newer.

```bash
git clone <repository-url> Seilbahn
cd Seilbahn
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:7861>. The app can start without the original source corpus. Import a JSON dataset in **Datasets & crawler**, or set `LABELER_SOURCE_PATH` before starting the app. If the original source is available next to the repository in `../Participants - Future City Hackathon/Stadt Heilbronn/`, the app will find the Heilbronn JSON automatically on its first run.

For local settings, copy the example and export its values into the shell before starting the app:

```bash
cp .env.example .env
set -a
source ./.env
set +a
python app.py
```

Python does not load `.env` automatically; the shell commands above do. `.env` is git-ignored.

## Dataset format and importing

Upload a JSON array whose entries follow the crawler's message format. The app accepts these standard fields:

```json
[
  {
    "mention_id": "source:123",
    "title": "Example title",
    "published_at": "2026-11-14T10:00:00+01:00",
    "media_type_name": "news_article",
    "source_name": "Example source",
    "language_code": "de",
    "url": "https://example.org/article",
    "content": "Full message or article text"
  }
]
```

Crawler aliases `timestamp`, `media_type`, `source`, and `language` are accepted as well. In **Datasets & crawler**, upload JSON to append its records, or use **Add one record manually**. New items pass through the same deduplication and keyword-label pipeline. The app skips blank records and known duplicates. When imported rows need review, the dashboard opens the **Manual labeling** queue automatically. Use **Progress & export → Export active dataset JSON** to download the current combined dataset in crawler-compatible format.

Imported records are stored in `data/annotations.sqlite3`; uploaded source files and manual-entry snapshots are also kept under the ignored `data/imports/` folder. A stable workspace dataset key lets a server continue using the same label set when the original JSON file is not mounted.

## Crawler behavior

The included crawler checks recent Heilbronn cable-car coverage from SWR, the City of Heilbronn, and Google News search feeds. The app runs it once at startup and repeats it every six hours by default. Each completed snapshot is written atomically to `data/incoming/heilbronn_cable_car_data.json`; the app imports new records and skips duplicates. Crawling and imports use the same local environment and database as the dashboard.

To change the interval, set `SEILBAHN_CRAWL_INTERVAL_SECONDS` (minimum five minutes). To disable scheduled crawler runs while retaining manual JSON import, set `SEILBAHN_DISABLE_CRAWLER=1`. You can also press **Run crawler now**. If crawler packages are missing, install the root `requirements.txt`; the dashboard still starts and displays the crawler error. For a crawler running on a different machine, set `LABELER_CRAWLER_IMPORT_PATH` to a JSON file or a directory of JSON snapshots; the dashboard polls it automatically.

## Run on a team server

Run one app process with one shared local SQLite database. Keep the app bound to localhost behind a TLS reverse proxy when possible. For direct binding to a private network, configure a username and password and firewall the port:

```bash
LABELER_HOST=0.0.0.0 \
LABELER_PORT=7861 \
LABELER_USERNAME='team-user' \
LABELER_PASSWORD='use-a-long-secret' \
python app.py
```

Do not expose plain HTTP to the public internet. The built-in login is basic access protection; use HTTPS and a firewall or an authenticated reverse proxy. Do not run multiple app processes against the same SQLite file. The app queue serializes writes for collaborators using one server instance.

## Label-review workflow

1. Open **Overview** to inspect the current charts and pending review count.
2. Open **Manual labeling** and load **Needs review**. Newly imported items with missing or ambiguous rule labels appear here.
3. Select project, support-status/score, and all applicable argument classes. Save labels as confirmed or flag them for another reviewer.
4. For empty, duplicate, or irrelevant rows, open **Remove an empty or irrelevant message**, enter an optional reason, check the confirmation box, and remove it. Removal hides the row and its labels from active charts, queues, training, and exports; the **Progress & export** tab can restore it.
5. Use **Progress & export** to download the combined dataset and SQLite backup.

The app saves human labels by message and project. If a message discusses more than one project, save one project label and then add the next project using the saved-label selector. Argument labels are multi-label. In this version the selected argument classes share the same direction and evidence/abstraction fields for each save.

## Models and optional tools

The in-app baseline uses local word and character TF-IDF features with scikit-learn. It needs human-reviewed examples and a held-out evaluation set before its predictions should be treated as reliable. Keyword suggestions are excluded from model training unless a human reviews and confirms them.

The prediction-audit UI can be started separately when inspecting a predictions SQLite file:

```bash
python tools/review_predictions.py
```

It writes human audit labels to a separate local database. `tools/seed_provisional_labels.py` is a legacy utility for creating provisional rule-based seed rows from the supplied Heilbronn source file; those rows are review-only. Neither tool is needed to run the main Seilbahn dashboard.

## Back up and move the dataset

Use **Create SQLite backup** before downloading or transferring the database. It includes SQLite's committed WAL contents. Save the original corpus separately if it is still needed; uploaded records are included in the database and in the combined JSON export. Keep the backup and exports out of Git and transfer them securely (for example, over SFTP).

## Configuration

See [.env.example](.env.example) for the supported host, port, crawler interval, source path, and credentials. The main defaults are localhost on port `7861`, automatic crawl every six hours, and the built-in source file if it is present beside the repository.
