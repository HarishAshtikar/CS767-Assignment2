"""DataSense agent loop."""

import pandas as pd

from config import DEFAULT_OUTPUTS_DIR, GEMINI_MODEL, MAX_AGENT_ITERATIONS
from data_context import build_data_context
from executor import execute_code
from llm_client import build_client, generate_agent_step
from prompts import NEXT_STEP_PROMPT, SYSTEM_PROMPT


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
        step = generate_agent_step(client, messages, SYSTEM_PROMPT)
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


if __name__ == "__main__":
    raise SystemExit(
        "Run `python main.py` to start the UI, or `python main.py --csv <path>` for direct CLI analysis."
    )
