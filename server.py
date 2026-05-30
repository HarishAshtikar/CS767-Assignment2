"""
DataSense Agent - FastAPI Backend
Handles file upload, runs agent, streams progress, returns results.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import uuid
import json
from pathlib import Path
from agent import run_agent
from llm_client import GeminiQuotaError

app = FastAPI(title="DataSense Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Serve generated chart files and the single-file UI.
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    goal: str = Form(...),
):
    """Upload CSV + goal → run agent → return report + chart URLs"""
    
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")

    # Save uploaded file
    session_id = str(uuid.uuid4())[:8]
    csv_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
    
    with open(csv_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run outputs to session-specific folder
    session_outputs = OUTPUTS_DIR / session_id
    session_outputs.mkdir(exist_ok=True)

    try:
        report = run_agent(
            csv_path=str(csv_path),
            user_goal=goal,
            outputs_dir=str(session_outputs),
        )
        
        # Convert chart paths to URLs
        chart_urls = []
        for chart_path in report.get("charts_generated", []):
            filename = Path(chart_path).name
            chart_urls.append(f"/outputs/{session_id}/{filename}")
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "summary": report["summary"],
            "key_findings": report["key_findings"],
            "charts": chart_urls,
            "steps_completed": report.get("steps_completed", []),
            "iterations": report.get("iterations", 0),
        })

    except GeminiQuotaError as e:
        raise HTTPException(429, str(e))

    except Exception as e:
        raise HTTPException(500, f"Agent error: {str(e)}")

    finally:
        # Clean up upload
        if csv_path.exists():
            os.remove(csv_path)


@app.get("/api/health")
def health():
    return {"status": "ok", "agent": "DataSense v1.0"}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
