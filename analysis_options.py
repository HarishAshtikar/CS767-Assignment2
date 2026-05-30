"""Dataset discovery and suggested analysis options."""

from pathlib import Path
import re

import pandas as pd


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


def suggest_analysis_options(csv_path: Path) -> list[dict]:
    """Suggest up to three analysis goals based on the dataset columns."""
    df = pd.read_csv(csv_path, nrows=200)
    numeric_cols = list(df.select_dtypes(include="number").columns)
    date_cols = _date_like_columns(df)
    categorical_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

    options = []

    if date_cols and numeric_cols:
        options.append(
            {
                "title": "Trend Analysis",
                "goal": (
                    f"Analyze trends over time using {', '.join(date_cols[:2])} and "
                    f"the numeric measures {', '.join(numeric_cols[:4])}. Identify changes, peaks, dips, and patterns."
                ),
            }
        )

    if categorical_cols and numeric_cols:
        options.append(
            {
                "title": "Group Comparison",
                "goal": (
                    f"Compare {', '.join(numeric_cols[:4])} across groups such as "
                    f"{', '.join(categorical_cols[:3])}. Identify top performers and meaningful differences."
                ),
            }
        )

    if len(numeric_cols) >= 2:
        options.append(
            {
                "title": "Relationships and Outliers",
                "goal": (
                    f"Explore relationships, correlations, distributions, and outliers among "
                    f"{', '.join(numeric_cols[:6])}."
                ),
            }
        )

    if len(options) < 3:
        options.append(
            {
                "title": "Dataset Overview",
                "goal": (
                    "Create a concise dataset overview covering column quality, missing values, "
                    "important distributions, and the strongest insights visible in the data."
                ),
            }
        )

    deduped = []
    seen = set()
    for option in options:
        if option["title"] not in seen:
            seen.add(option["title"])
            deduped.append(option)
    return deduped[:3]


def _date_like_columns(df: pd.DataFrame) -> list[str]:
    date_cols = []
    for col in df.columns:
        lower_col = str(col).lower()
        if "date" in lower_col or "time" in lower_col or "month" in lower_col or "year" in lower_col:
            date_cols.append(col)
            continue
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.7:
                date_cols.append(col)
    return date_cols
