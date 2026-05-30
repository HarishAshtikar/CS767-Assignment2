"""Dataset discovery, cached LLM analysis suggestions, and output naming."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re

import pandas as pd

from llm_client import build_client, generate_json


MAX_ANALYSIS_OPTIONS = 5
ANALYSIS_CACHE_VERSION = 2


def safe_dataset_id(filename: str) -> str:
    """Return a filesystem-safe dataset id based on the filename stem."""
    stem = Path(filename).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return safe or "dataset"


def list_csv_datasets(input_dir: Path) -> list[dict]:
    """List CSV datasets available in the input directory."""
    input_dir.mkdir(exist_ok=True)
    datasets = []
    for path in sorted(input_dir.glob("*.csv")):
        datasets.append(
            {
                "id": path.name,
                "name": path.name,
                "stem": safe_dataset_id(path.name),
                "size_bytes": path.stat().st_size,
            }
        )
    return datasets


def resolve_dataset_path(input_dir: Path, dataset_name: str) -> Path:
    """Resolve a dataset name while preventing path traversal."""
    dataset_path = (input_dir / Path(dataset_name).name).resolve()
    input_root = input_dir.resolve()
    if input_root not in dataset_path.parents or dataset_path.suffix.lower() != ".csv":
        raise ValueError("Invalid dataset path")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")
    return dataset_path


def suggest_analysis_options(csv_path: Path, output_dir: Path) -> list[dict]:
    """
    Suggest up to five analysis goals using Gemini and cache them.

    The cache is stored at output/analysis.json and invalidated when the dataset
    size or modification time changes.
    """
    output_dir.mkdir(exist_ok=True)
    cache_path = output_dir / "analysis.json"
    fingerprint = _dataset_fingerprint(csv_path)
    cache = _read_cache(cache_path)
    cached = cache.get("datasets", {}).get(csv_path.name)

    if _cache_entry_is_valid(cached, fingerprint):
        return _normalize_options(cached["options"])

    options = _generate_options(csv_path)
    cache.setdefault("datasets", {})[csv_path.name] = _cache_entry(fingerprint, options)
    _write_cache(cache_path, cache)
    return options


def ensure_analysis_cache(input_dir: Path, output_dir: Path) -> dict:
    """
    Ensure output/analysis.json contains analysis options for every CSV in input/.

    Cached entries are reused when the dataset fingerprint and cache version match.
    Missing or stale entries are generated with Gemini.
    """
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    cache_path = output_dir / "analysis.json"
    cache = _read_cache(cache_path)
    cache.setdefault("datasets", {})

    changed = False
    for csv_path in sorted(input_dir.glob("*.csv")):
        fingerprint = _dataset_fingerprint(csv_path)
        cached = cache["datasets"].get(csv_path.name)
        if _cache_entry_is_valid(cached, fingerprint):
            continue

        options = _generate_options(csv_path)
        cache["datasets"][csv_path.name] = _cache_entry(fingerprint, options)
        changed = True

    valid_names = {path.name for path in input_dir.glob("*.csv")}
    for dataset_name in list(cache["datasets"]):
        if dataset_name not in valid_names:
            del cache["datasets"][dataset_name]
            changed = True

    if changed or not cache_path.exists():
        _write_cache(cache_path, cache)
    return cache


def get_cached_analysis_options(csv_path: Path, output_dir: Path) -> list[dict]:
    """Read cached options for a dataset without calling Gemini."""
    cache = _read_cache(output_dir / "analysis.json")
    fingerprint = _dataset_fingerprint(csv_path)
    cached = cache.get("datasets", {}).get(csv_path.name)
    if not _cache_entry_is_valid(cached, fingerprint):
        return []
    return _normalize_options(cached["options"])


def _generate_options_with_llm(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path, nrows=10)
    sample_csv = df.to_csv(index=False)
    dtypes = df.dtypes.astype(str).to_dict()

    prompt = f"""You are helping a user choose useful data analysis goals for a CSV dataset.
Read only this dataset preview and propose at most {MAX_ANALYSIS_OPTIONS} distinct analyses.

Dataset filename: {csv_path.name}
Columns and inferred dtypes:
{json.dumps(dtypes, indent=2)}

First 10 rows as CSV:
{sample_csv}

Return only valid JSON with this exact shape:
{{
  "options": [
    {{
      "title": "short option title",
      "goal": "specific analysis goal the downstream data agent can execute"
    }}
  ]
}}

Rules:
- Return between 3 and {MAX_ANALYSIS_OPTIONS} options when the data supports it.
- Make each option meaningfully different.
- Mention relevant column names in each goal.
- Do not invent columns that are not in the preview.
"""

    response = generate_json(build_client(), prompt, "Generate practical CSV analysis options.")
    return _normalize_options(response.get("options", []))


def _generate_options(csv_path: Path) -> list[dict]:
    """Generate suggestions with Gemini, falling back to local heuristics."""
    try:
        return _generate_options_with_llm(csv_path)
    except Exception:
        return _generate_fallback_options(csv_path)


def _generate_fallback_options(csv_path: Path) -> list[dict]:
    """Create useful suggestions locally when Gemini quota is unavailable."""
    df = pd.read_csv(csv_path, nrows=10)
    columns = [str(column) for column in df.columns]
    if not columns:
        return [
            {
                "title": "Summarize Dataset",
                "goal": "Summarize the dataset structure, row count, columns, missing values, and any notable data quality issues.",
            }
        ]

    numeric_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_columns = [column for column in columns if column not in numeric_columns]

    options = [
        {
            "title": "Dataset Overview",
            "goal": (
                "Summarize the dataset structure, row count, column types, missing values, "
                f"and notable patterns across {', '.join(columns[:8])}."
            ),
        }
    ]

    if numeric_columns:
        options.append(
            {
                "title": "Numeric Trends",
                "goal": (
                    "Analyze distributions, outliers, correlations, and summary statistics "
                    f"for numeric columns including {', '.join(numeric_columns[:6])}."
                ),
            }
        )

    if categorical_columns:
        options.append(
            {
                "title": "Category Breakdown",
                "goal": (
                    "Compare frequencies, dominant groups, and relationships for categorical columns "
                    f"including {', '.join(categorical_columns[:6])}."
                ),
            }
        )

    if numeric_columns and categorical_columns:
        options.append(
            {
                "title": "Group Comparisons",
                "goal": (
                    f"Compare {', '.join(numeric_columns[:3])} across groups such as "
                    f"{', '.join(categorical_columns[:3])}, highlighting meaningful differences."
                ),
            }
        )

    options.append(
        {
            "title": "Chart Key Patterns",
            "goal": (
                "Generate charts that reveal the most important distributions, comparisons, "
                "relationships, and possible anomalies in the dataset."
            ),
        }
    )
    return _normalize_options(options)


def _normalize_options(options: list[dict]) -> list[dict]:
    normalized = []
    seen = set()
    for option in options:
        title = str(option.get("title", "")).strip()
        goal = str(option.get("goal", "")).strip()
        if not title or not goal or title.lower() in seen:
            continue
        seen.add(title.lower())
        normalized.append({"title": title, "goal": goal})
    return normalized[:MAX_ANALYSIS_OPTIONS]


def _dataset_fingerprint(csv_path: Path) -> dict:
    stat = csv_path.stat()
    return {
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def _cache_entry(fingerprint: dict, options: list[dict]) -> dict:
    return {
        "cache_version": ANALYSIS_CACHE_VERSION,
        "fingerprint": fingerprint,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "options": options,
    }


def _cache_entry_is_valid(cached: dict | None, fingerprint: dict) -> bool:
    return bool(
        cached
        and cached.get("cache_version") == ANALYSIS_CACHE_VERSION
        and cached.get("fingerprint") == fingerprint
        and cached.get("options")
    )


def _read_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {"datasets": {}}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"datasets": {}}


def _write_cache(cache_path: Path, cache: dict) -> None:
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
