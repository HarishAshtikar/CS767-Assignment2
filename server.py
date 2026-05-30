"""DataSense Agent - FastAPI backend."""

from pathlib import Path
import json
import shutil

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from agent import run_agent
from analysis_options import list_csv_datasets, resolve_dataset_path, safe_dataset_id, suggest_analysis_options
from config import DEFAULT_OUTPUTS_DIR, INPUT_DIR
from llm_client import GeminiQuotaError


app = FastAPI(title="DataSense Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INPUT_PATH = Path(INPUT_DIR)
OUTPUTS_DIR = Path(DEFAULT_OUTPUTS_DIR)
INPUT_PATH.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

app.mount("/output", StaticFiles(directory=str(OUTPUTS_DIR)), name="output")


class AnalyzeRequest(BaseModel):
    dataset: str
    goal: str


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/api/datasets")
def datasets():
    """Return CSV datasets available in the input folder."""
    return {"datasets": list_csv_datasets(INPUT_PATH)}


@app.get("/api/datasets/{dataset_name}/suggestions")
def suggestions(dataset_name: str):
    """Return up to five suggested analysis options for a dataset."""
    try:
        csv_path = resolve_dataset_path(INPUT_PATH, dataset_name)
        return {
            "dataset": csv_path.name,
            "options": suggest_analysis_options(csv_path, OUTPUTS_DIR),
        }
    except GeminiQuotaError as exc:
        raise HTTPException(429, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    """Run the selected dataset and analysis goal, then return persisted output."""
    if not request.goal.strip():
        raise HTTPException(400, "Analysis goal is required")

    try:
        csv_path = resolve_dataset_path(INPUT_PATH, request.dataset)
        dataset_stem = safe_dataset_id(csv_path.name)
        dataset_outputs = OUTPUTS_DIR / dataset_stem

        if dataset_outputs.exists():
            shutil.rmtree(dataset_outputs)
        dataset_outputs.mkdir(parents=True, exist_ok=True)

        report = run_agent(
            csv_path=str(csv_path),
            user_goal=request.goal.strip(),
            outputs_dir=str(dataset_outputs),
        )

        report_payload = {
            "dataset": csv_path.name,
            "output_folder": str(dataset_outputs),
            "goal": request.goal.strip(),
            **report,
        }
        (dataset_outputs / "report.json").write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        (dataset_outputs / "report.md").write_text(report.get("summary", ""), encoding="utf-8")

        chart_urls = []
        for chart_path in report.get("charts_generated", []):
            filename = Path(chart_path).name
            chart_urls.append(f"/output/{dataset_stem}/{filename}")

        return JSONResponse(
            {
                "success": True,
                "dataset": csv_path.name,
                "output_folder": f"output/{dataset_stem}",
                "summary": report["summary"],
                "key_findings": report["key_findings"],
                "charts": chart_urls,
                "steps_completed": report.get("steps_completed", []),
                "iterations": report.get("iterations", 0),
            }
        )

    except GeminiQuotaError as exc:
        raise HTTPException(429, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Agent error: {str(exc)}")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "agent": "DataSense v1.0",
        "input_dir": str(INPUT_PATH),
        "output_dir": str(OUTPUTS_DIR),
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
