# PHAK Study Assistant

A retrieval and practice tool for selected chapters of the FAA Pilot's Handbook of Aeronautical Knowledge (PHAK). PDF text and embeddings stay local; Gemini supplies the default LLM and requires an API key. Gemini usage may have quotas or charges.

**Study from the book as well as this tool.** Local models can make mistakes or miss nuance. Verify answers and grading against the PHAK. Figures and diagrams are not indexed; selectable text captions are retained when present, but diagrams themselves must be studied in the book.

## Setup

Requires Python 3.11+. Then:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gemini.txt
cp .env.example .env
```

Add your Gemini API key as `GEMINI_API_KEY` in `.env`. The default Gemini model is `gemini-3.8-flash`. Generate a key in [Google AI Studio](https://aistudio.google.com/apikey).

The local sentence-transformers model is `BAAI/bge-base-en-v1.5`. Its first use may download model files from Hugging Face; subsequent embedding runs are local and have no API quota. Embedding settings live in `config.yaml`. `.env` is ignored by Git.

## Get the PHAK PDF

Download the current PHAK from the [FAA's official handbook page](https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/phak). Save either the full handbook or chapter PDFs 4, 5, 6, 7, and 12 in `data/`. The default config filters chapter 7 to sections 7-1 through 7-19. Source citations use physical PDF page numbers. The `data/` folder is not committed.

## Commands

Run these from the project folder with the virtual environment active:

```sh
PYTHONPATH=src python -m phak_study.cli ingest
PYTHONPATH=src python -m phak_study.cli retrieve "Why does a wing produce lift?"
PYTHONPATH=src python -m phak_study.cli ask "Why does a wing produce lift?"
PYTHONPATH=src python -m phak_study.cli ask "What happens when angle of attack exceeds critical?" --hide-excerpts
PYTHONPATH=src python -m phak_study.cli quiz-prepare --chapter 5 --n 30
PYTHONPATH=src python -m phak_study.cli quiz --chapter 5 --n 10
PYTHONPATH=src python -m phak_study.cli quiz --weak --n 10
PYTHONPATH=src python -m phak_study.cli stats
PYTHONPATH=src python scripts/evaluate_retrieval.py
PYTHONPATH=src python scripts/check_grading.py
```

`ask` streams its answer and displays source excerpts by default. Each factual claim is prompted to cite a retrieved source label that maps to chapter, section, and PDF page. The CLI warns if a sentence lacks a valid source label. `--hide-excerpts` hides the text excerpts while retaining citations. Quiz questions can be prepared ahead and cached in SQLite. `quiz` uses cached questions when available and generates questions on demand when the matching cache is empty. Type `q` at an answer prompt to stop.

The grading sanity script makes three live model calls (good, partial, and wrong sample answers) and reports score ranges; run it after changing models to compare them. The retrieval evaluation reports hit rate at configured k. Adjust its expected section labels after downloading your PHAK edition because heading text can differ between PDFs.

SQLite attempt history and the persistent Chroma index are local (`study.sqlite3` and `.chroma/`). Re-running ingestion replaces each PDF's indexed chunks rather than duplicating them.

## Choosing another provider

Edit `config.yaml` to use another provider. Ollama is available as a local alternative; install Ollama and the base requirements, pull a model, then configure it:

```yaml
llm_provider: ollama
llm_model: qwen3.5:2b
llm_base_url: http://localhost:11434
llm_timeout_seconds: 600
llm_temperature: 0.1
```

Optional OpenAI-compatible service (for example Groq or OpenRouter): install `requirements-openai-compatible.txt`, set `llm_provider: openai_compatible`, set its model and API base URL in config, and put the key named by `llm_api_key_env` (default `OPENAI_API_KEY`) in `.env`.

Prompts are in `src/phak_study/prompts.py`. Quiz JSON uses Pydantic schemas, Ollama JSON Schema constrained output, validation, and a repair retry. Supporting quotes are checked against their source chunks after whitespace normalization; a quote that fails a retry is labeled unverified.
