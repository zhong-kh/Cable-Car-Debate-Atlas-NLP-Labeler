# Cable Car Debate Atlas — Labeling Workbench

This is a local Gradio app for building a human-reviewed training dataset from the supplied media-monitoring JSON. It shows each message's title and content, stores labels in SQLite, exports CSV/JSONL, and trains a first-pass model to suggest labels for the remaining records.

## Start the dashboard

From a terminal:

```bash
cd Cable-Car-Debate-Atlas-NLP-Labeler
python --version  # use Python 3.9 or newer
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

The app runs locally at `http://127.0.0.1:7861`; sharing is disabled. It automatically finds the supplied file under `Participants - Future City Hackathon/Stadt Heilbronn/` when this project folder is in its current location. To point it to another JSON/CSV/TSV dataset, edit `find_default_dataset()` in `app.py`; it expects at least `title` and `content` fields.

## Run on a team server

The app serializes label/save requests through Gradio's queue and stores them in one SQLite database on the server. All annotators visiting the same server instance therefore see the same label progress. Keep the source JSON at the same relative location next to the cloned `NLP_Labeler` folder, or update `find_default_dataset()` for the server's dataset path. The source dataset is intentionally not part of this code repository.

For direct network binding, set a username and password and listen on the server interface:

```bash
LABELER_HOST=0.0.0.0 \
LABELER_PORT=7861 \
LABELER_USERNAME='team-user' \
LABELER_PASSWORD='use-a-long-secret' \
python app.py
```

Use a TLS reverse proxy (for example Caddy or Nginx with HTTPS) and firewall the Gradio port so it is not exposed as plain HTTP. Do not commit credentials or the SQLite database. If the server is behind an authenticated reverse proxy, keep the Gradio port private to localhost and configure the proxy to require HTTPS and authentication.

At the end of annotation, use **Create SQLite backup** in the Progress & export tab, then SFTP-download the generated `annotations_backup_*.sqlite3` from `data/exports/`. This uses SQLite's backup API to create a consistent snapshot even when WAL journaling is enabled. Also download the original source JSON and any CSV/JSONL exports you need. The backup and export folder are git-ignored.

## What is stored

- Human annotations: `data/annotations.sqlite3`
- Local model files: `models/`
- The original 200 MB JSON remains the source of text; its full contents are not copied into SQLite.
- Each saved row identifies a `mention_id × project_id`. To annotate a message about two projects, save one project row and then save the other.
- Argument labels are stored as separate `mention_id × project_id × argument_type` records, including direction, an optional verbatim evidence quote, and an optional short abstraction. In this first UI, the same direction/quote/abstraction is applied to all categories selected in one save; select together only categories that share that context.
- Model suggestions are stored separately from human annotations and never overwrite them.

Back up the SQLite file together with the source dataset. CSV and RAG-friendly JSONL snapshots are available from the **Progress & export** tab.

## Annotation scale

The numeric scale means:

| Score | Meaning |
|---:|---|
| 0 | Strongly opposes |
| 1 | Opposes |
| 2 | Leans against |
| 3 | Leans in favor |
| 4 | Supports |
| 5 | Strongly supports |

Neutral, mixed/conditional, no expressed position and unclear are separate statuses without a numeric score. This prevents the model from inventing a neutral position on a scale that has no neutral value.

Argument types are multi-label. Their direction is recorded relative to the selected cable-car proposal. The current form applies one shared direction and one shared evidence/abstraction note to the selected categories in a save. These fields make later argument abstraction auditable.

## How the training pipeline works

### First baseline (available in the app)

The dashboard's first model is deliberately simple and fully local:

1. **Project Index:** a multi-label classifier predicts one or more named projects. General cable-car discussion and unrelated posts that a human explicitly labels are negative examples. Add any new concrete project through the project index before labeling it.
2. **Argument Type + direction:** a multi-label classifier predicts combinations such as `cost_finance::con` or `economic_development::pro`. It sees the target project name as part of its input.
3. **Support Score:** a cumulative ordinal classifier predicts the five thresholds `score ≥ 1` … `score ≥ 5`, then reconstructs probabilities for levels 0–5. This respects the ordering between score levels better than treating 0–5 as unrelated categories.
4. **Predict remaining records:** project candidates are predicted first; argument and support suggestions are then conditioned on each predicted project. The app writes these predictions to the predictions table and flags low-confidence cases for review. Human annotations remain untouched.

Models use word and character TF-IDF features with scikit-learn Logistic Regression. Character features are a useful fast baseline for German/French spellings, inflections and names. It needs no API key and no pretrained checkpoint download. The app uses the entire `Title + Content` string, capped at 30,000 characters for model computation; the full original text remains visible and is exported intact.

The current baseline does **not** claim validated accuracy. Raw Logistic Regression probabilities are not calibrated confidence. The prediction threshold controls which suggestions are flagged for review; tune it against a human-labeled validation set before presenting predictions as reliable.

### Recommended model progression

**Phase 1 — collect labels:** start with a diverse pilot across German/French, named projects, media types, years, clear/unclear position, and argument classes. Double-label at least 20–25% and adjudicate differences. Do not label all first examples from the same news source or duplicate story.

**Phase 2 — train the baseline:** use the dashboard button after there are reviewed examples from multiple projects or explicit general/unrelated negatives, at least two examples for each argument-class/direction pair, and more than one scored support level. Early models are for queue triage, not unattended publishing.

**Phase 3 — evaluate before relying on it:** reserve a human-labeled validation set from different URLs/sources, preferably from a later time period. Report project-level and language-level precision/recall/F1; argument-level macro-F1 and per-class precision/recall; and support-score MAE plus quadratic weighted kappa. Keep challenging and rare examples in the evaluation set. This dashboard version does not yet calculate formal holdout metrics, so the team should not infer accuracy from training success.

**Phase 4 — multilingual neural upgrade:** after the baseline and annotation guide are stable, compare a SetFit model using a German/French-capable sentence transformer (for example `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) against the TF-IDF baseline. Use SetFit's multi-label head for argument categories/directions and project candidates. For support, prepend the project target and compare a target-aware 6-level classifier with a cumulative ordinal head. Retain the same held-out evaluation set for a fair comparison. This requires installing the extra `setfit` and `sentence-transformers` packages and downloading a model checkpoint.

**Argument abstraction:** the app collects a short, human-written paraphrase linked to a quote. A model that generates argument summaries is a separate text-generation task; it needs examples of source text → verified evidence span → faithful abstract → class/direction. Once those pairs exist, pilot a constrained JSON/text-to-text model that cites the source span and abstains when evidence is missing. Review generated summaries for factual faithfulness and preservation of minority arguments before publication. Do not treat an automatically generated paraphrase as an independent source.

## Research informing the choices

These are primary research papers or official tool docs. The mobility-participation corpus is the closest task/domain match; SetFit is the candidate low-label multilingual upgrade; target-aware stance methods guide the support model; argument-mining/summarization papers inform the later abstraction phase.

1. **Most domain-relevant:** Romberg, Mark & Escher (LREC 2022), [A Corpus of German Citizen Contributions in Mobility Planning](https://aclanthology.org/2022.lrec-1.308/). It covers contributions to six planning processes in five German municipalities and annotates argument recognition, argument concreteness, location and themes. It is a strong reference for our codebook and evaluation dimensions. Check license and label compatibility before reusing data.
2. Vamvas & Sennrich (2020), [X-Stance: A Multilingual Multi-Target Dataset for Stance Detection](https://arxiv.org/abs/2003.08385). German/French/Italian comments are paired with issue targets; the paper provides a close methodological analogue for target-conditioned cable-car support scoring, with a domain-transfer caveat.
3. Du et al. (IJCAI 2017), [Stance Classification with Target-specific Neural Attention](https://www.ijcai.org/Proceedings/2017/557). Supports conditioning stance detection on the target instead of generic sentiment.
4. Tunstall et al. (NeurIPS 2022), [Efficient Few-Shot Learning Without Prompts (SetFit)](https://arxiv.org/abs/2209.11055). Proposes contrastive fine-tuning of sentence embeddings followed by a classification head; supports the low-label model upgrade option.
5. Hugging Face, [SetFit multi-label classification guide](https://huggingface.co/docs/setfit/how_to/multilabel); Sentence Transformers, [multilingual pretrained model list](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html). Official implementation documentation.
6. Kawarada et al. (EACL 2024), [Argument Mining as a Text-to-Text Generation Task](https://aclanthology.org/2024.eacl-long.121/). Shows a generative route for spans, argument components and relations. Consider after the team has verified structured annotations; it is more complex than tag prediction.
7. Syed et al. (SIGDIAL 2023), [Frame-oriented Summarization of Argumentative Discussions](https://aclanthology.org/2023.sigdial-1.10/). Relevant for aggregating many comments into frames/key points while retrieving representative arguments.
8. Syed et al. (COLING 2020), [News Editorials: Towards Summarizing Long Argumentative Texts](https://aclanthology.org/2020.coling-main.470/). Useful for judging concise summaries of individual long opinion articles.
9. Nadeem & Ostendorf (WASSA 2023), [Examining Bias in Opinion Summarisation through Opinion Diversity](https://aclanthology.org/2023.wassa-1.14/). Motivates reporting minority as well as majority arguments in any generated summary.
10. Cao et al. (2020), [Rank-consistent ordinal regression (CORAL)](https://arxiv.org/abs/1901.07884). Provides a principled neural ordinal alternative if we upgrade the support-score model.
11. Guo et al. (ICML 2017), [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). Motivates treating confidence calibration as a separate evaluation step.

## Known limits in this version

- The current model is a fast baseline, not a validated production classifier. It needs domain labels, distinct positive and negative examples, and separate evaluation.
- The current project classifier is multi-label; the human interface stores one project per annotation row so that two targets can receive different scores.
- Support is the text's expressed position, not a population approval percentage. News volume and platform coverage are not representative polling data.
- Automatic argument abstraction is not generated yet; this version captures the evidence and gold paraphrases needed to train or prompt it responsibly.
- This is a local single-user workbench. If multiple annotators need simultaneous access, move the database to a managed service and add user authentication/audit rules before deployment.
