# DataSense - Autonomous Data Analysis Agent

DataSense perceives CSV data, suggests analysis paths, plans selected analyses with Gemini, runs pandas/matplotlib code locally, retries failed steps, and produces charts plus a natural-language report.

## What It Does

DataSense is an agentic loop. You place CSV files in `input/`, choose a dataset in the UI, and pick one of the suggested analyses. The agent then:

1. Perceives the data structure, column types, statistics, and missing values.
2. Suggests up to three analysis goals based on the available columns.
3. Executes pandas/matplotlib analysis code.
4. Retries failed code with the error context.
5. Reports key findings and generated chart paths.

## Quick Start

### Prerequisites

- Python 3.10+
- Gemini API key from Google AI Studio

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Secrets

```bash
cp .env.example .env
```

Then edit `.env` and set:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3-pro-preview
```

`GEMINI_MODEL` is configurable, so you can replace it with a Gemini 3.5 model alias if your API account exposes one.

### Add Datasets

Put CSV files in the `input/` folder:

```text
input/
└── your_dataset.csv
```

Each analysis writes to a matching subfolder:

```text
output/
└── your_dataset/
    ├── report.json
    ├── report.md
    └── chart files
```

### Run the Web App

```bash
python main.py
```

Open `http://localhost:8000`.

### Run the Agent Directly

```bash
python main.py --csv input/sample_data.csv --goal "Summarize trends and generate useful charts"
```

## Project Structure

```text
datasense/
├── .env.example          # Required environment variables
├── agent.py              # Core agent loop
├── analysis_options.py   # Dataset discovery and suggested analysis goals
├── config.py             # Environment-driven runtime config
├── data_context.py       # CSV perception helpers
├── executor.py           # Sandboxed Python runner
├── llm_client.py         # Gemini client wrapper
├── main.py               # CLI and server orchestrator
├── prompts.py            # Agent prompts
├── server.py             # FastAPI backend
├── index.html            # Web UI
├── input/                # CSV datasets selected in the UI
├── output/               # Per-dataset reports and charts, auto-created
└── requirements.txt
```

## Agent Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Perception | CSV parsed into data context with shape, dtypes, samples, stats, and null counts |
| Suggestions | The app proposes up to three analysis goals from dataset column types |
| Decision-making | Gemini chooses analysis steps dynamically based on the selected goal |
| Action | Python analysis code is executed locally through the sandboxed executor |
| Memory | Completed steps, chart paths, errors, and retry counts are tracked across iterations |
| Retry | Execution errors are fed back into the loop for correction |
| Goal-directed loop | The agent continues until it emits `finish_report` or reaches the iteration limit |
| Safety | Code execution uses restricted builtins and patched chart output paths |
