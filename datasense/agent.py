"""DataSense agent loop."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import DEFAULT_OUTPUTS_DIR, GEMINI_MODEL, MAX_AGENT_ITERATIONS
from .data_context import build_data_context
from .executor import execute_code
from .llm_client import GeminiQuotaError, build_client, generate_agent_step
from .prompts import NEXT_STEP_PROMPT, SYSTEM_PROMPT


def run_agent(csv_path: str, user_goal: str, outputs_dir: str = DEFAULT_OUTPUTS_DIR) -> dict:
    """
    Run the autonomous data analysis loop.

    The agent perceives a CSV, asks Gemini for the next action, executes analysis
    code locally, feeds results back, and stops when Gemini emits finish_report.
    """
    print(f"\n{'=' * 60}")
    print("DataSense Agent Starting")
    print(f"Provider: Gemini")
    print(f"Model: {GEMINI_MODEL}")
    print(f"CSV: {csv_path}")
    print(f"Goal: {user_goal}")
    print(f"{'=' * 60}\n")

    client = build_client()
    df = pd.read_csv(csv_path)
    data_context = build_data_context(df)

    messages = [
        {
            "role": "user",
            "content": f"""Please analyze this dataset and address the following goal:

Goal: {user_goal}

Data Context:
{data_context}

The outputs directory is: {outputs_dir}
Save all charts to: {outputs_dir}/<name>.png

{NEXT_STEP_PROMPT}
""",
        }
    ]

    memory = {
        "steps_completed": [],
        "charts_generated": [],
        "errors_encountered": [],
        "retry_count": {},
    }
    final_report = None

    for iteration in range(MAX_AGENT_ITERATIONS):
        print(f"[Iteration {iteration + 1}] Calling Gemini...")
        try:
            step = generate_agent_step(client, messages, SYSTEM_PROMPT)
        except GeminiQuotaError:
            print("  Gemini quota exhausted. Running local fallback analysis.")
            return run_local_analysis(df, user_goal, outputs_dir)

        action = step.get("action")
        description = step.get("description", f"step_{iteration + 1}")

        if action == "execute_python":
            print(f"  [TOOL] execute_python: {description[:80]}")
            result = execute_code(step["code"], df, outputs_dir)

            if result["success"]:
                memory["steps_completed"].append(description)
                memory["charts_generated"].extend(result["saved_files"])
                print(f"  Success. Output: {result['stdout'][:100]}")
                tool_result_content = (
                    f"Executed step: {description}\n"
                    f"SUCCESS\nOutput:\n{result['stdout']}\nSaved files: {result['saved_files']}"
                )
            else:
                memory["errors_encountered"].append(result["error"])
                retry_count = memory["retry_count"].get(description, 0) + 1
                memory["retry_count"][description] = retry_count
                print(f"  Error (retry {retry_count}): {result['error'][:150]}")
                tool_result_content = (
                    f"Executed step: {description}\n"
                    f"ERROR (attempt {retry_count}):\n{result['error']}\n"
                    "Please fix the code and retry."
                )

            messages.append({"role": "assistant", "content": str(step)})
            messages.append({"role": "user", "content": f"{tool_result_content}\n\n{NEXT_STEP_PROMPT}"})
            continue

        if action == "finish_report":
            print("  Agent finished. Generating report...")
            final_report = {
                "summary": step.get("summary", ""),
                "key_findings": step.get("key_findings", []),
                "charts_generated": memory["charts_generated"] or step.get("charts_generated", []),
                "steps_completed": memory["steps_completed"],
                "iterations": iteration + 1,
            }
            break

        messages.append(
            {
                "role": "user",
                "content": f"Unknown action '{action}'. Return a valid action.\n\n{NEXT_STEP_PROMPT}",
            }
        )

    if not final_report:
        final_report = {
            "summary": "Analysis completed but report generation was interrupted.",
            "key_findings": memory["steps_completed"],
            "charts_generated": memory["charts_generated"],
            "steps_completed": memory["steps_completed"],
            "iterations": MAX_AGENT_ITERATIONS,
        }

    print(f"\n{'=' * 60}")
    print(f"Analysis complete. {len(memory['charts_generated'])} charts generated.")
    print(f"{'=' * 60}\n")

    return final_report


def run_local_analysis(df: pd.DataFrame, user_goal: str, outputs_dir: str = DEFAULT_OUTPUTS_DIR) -> dict:
    """Produce a useful deterministic report when Gemini is unavailable."""
    output_path = Path(outputs_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    numeric_columns = list(df.select_dtypes(include="number").columns)
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    charts_generated = []

    if numeric_columns:
        df[numeric_columns[:6]].hist(figsize=(12, 8), bins=20)
        figure = plt.gcf()
        figure.suptitle("Numeric Distributions")
        figure.tight_layout()
        chart_path = output_path / "numeric_distributions.png"
        figure.savefig(chart_path, dpi=150)
        plt.close(figure)
        charts_generated.append(str(chart_path))

    if len(numeric_columns) >= 2:
        figure = df[numeric_columns].corr(numeric_only=True).plot(
            kind="bar",
            figsize=(12, 7),
            title="Numeric Correlations",
        ).figure
        figure.tight_layout()
        chart_path = output_path / "numeric_correlations.png"
        figure.savefig(chart_path, dpi=150)
        plt.close(figure)
        charts_generated.append(str(chart_path))

    for column in categorical_columns[:1]:
        counts = df[column].astype(str).value_counts(dropna=False).head(12)
        figure = counts.plot(kind="bar", figsize=(10, 6), title=f"Top Values: {column}").figure
        figure.tight_layout()
        chart_path = output_path / f"top_values_{_safe_chart_name(column)}.png"
        figure.savefig(chart_path, dpi=150)
        plt.close(figure)
        charts_generated.append(str(chart_path))

    missing_total = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    key_findings = [
        f"The dataset contains {len(df):,} rows and {len(df.columns):,} columns.",
        f"Detected {len(numeric_columns):,} numeric columns and {len(categorical_columns):,} non-numeric columns.",
        f"Found {missing_total:,} missing values and {duplicate_rows:,} duplicate rows.",
    ]

    if numeric_columns:
        summary_stats = df[numeric_columns].describe().loc[["mean", "min", "max"]].round(3)
        key_findings.append(
            "Numeric summary: "
            + "; ".join(
                f"{column} mean={summary_stats.at['mean', column]}, min={summary_stats.at['min', column]}, max={summary_stats.at['max', column]}"
                for column in numeric_columns[:4]
            )
        )

    if categorical_columns:
        column = categorical_columns[0]
        top_value = df[column].astype(str).value_counts(dropna=False).head(1)
        if not top_value.empty:
            key_findings.append(f"The most common value in {column} is {top_value.index[0]} ({int(top_value.iloc[0]):,} rows).")

    summary = "\n\n".join(
        [
            "## Local analysis fallback",
            "Gemini quota was unavailable, so DataSense generated this deterministic pandas report locally.",
            f"**Goal:** {user_goal}",
            "\n".join(f"- {finding}" for finding in key_findings),
        ]
    )

    return {
        "summary": summary,
        "key_findings": key_findings,
        "charts_generated": charts_generated,
        "steps_completed": ["Ran local pandas summary", "Generated fallback charts"],
        "iterations": 0,
    }


def _safe_chart_name(value: object) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(value)).strip("_") or "column"


if __name__ == "__main__":
    raise SystemExit(
        "Run `python main.py` to start the UI, or `python main.py --csv <path>` for direct CLI analysis."
    )
