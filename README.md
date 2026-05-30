# DataSense CSV Analysis Agent

DataSense is a local FastAPI web app and CLI for analyzing CSV files. It uses Gemini to suggest and run analyses, then falls back to local pandas/matplotlib summaries when Gemini quota is unavailable.

## Reproduce

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
4. Put one or more CSV files in `input/`.
5. Start the app:

```bash
python main.py
```

6. Open `http://127.0.0.1:8000`, choose a dataset, select or edit a goal, and run the analysis.

Outputs are written to `output/<dataset-name>/` as `report.json`, `report.md`, and PNG charts.

## CLI

```bash
python main.py --csv input/sample_data.csv --goal "Summarize trends and generate useful charts"
```

## Project Layout

```text
datasense/            Python package for the app and agent
datasense/agent.py    Gemini-guided analysis loop and local fallback
datasense/server.py   FastAPI backend
index.html            Single-file web UI
main.py               Root CLI/server launcher
input/                Local CSV datasets, ignored except .gitkeep
output/               Generated reports and charts, ignored
docs/                 Architecture diagrams and demo video
```

## Verify

```bash
python -m compileall main.py datasense
```
