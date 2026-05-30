"""Command-line orchestrator for DataSense."""

import argparse
from pathlib import Path

from datasense.agent import run_agent
from datasense.analysis_options import safe_dataset_id
from datasense.config import DEFAULT_OUTPUTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DataSense CSV analysis agent.")
    parser.add_argument("--csv", help="Path to a CSV file to analyze directly. Omit to start the UI.")
    parser.add_argument(
        "--goal",
        default="Understand the dataset, identify important patterns, and generate useful charts.",
        help="Natural-language analysis goal.",
    )
    parser.add_argument(
        "--outputs-dir",
        help="Directory for generated charts. Defaults to output/<dataset-name>/ for direct CLI analysis.",
    )
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI web server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host for --serve.")
    parser.add_argument("--port", type=int, default=8000, help="Port for --serve.")
    args = parser.parse_args()

    if args.serve or not args.csv:
        import uvicorn

        print(f"Starting DataSense UI at http://{args.host}:{args.port}")
        uvicorn.run("datasense.server:app", host=args.host, port=args.port, reload=True)
        return

    csv_path = Path(args.csv)
    if not csv_path.exists():
        parser.error(f"CSV file not found: {csv_path}")

    outputs_dir = args.outputs_dir or str(Path(DEFAULT_OUTPUTS_DIR) / safe_dataset_id(csv_path.name))
    result = run_agent(
        csv_path=str(csv_path),
        user_goal=args.goal,
        outputs_dir=outputs_dir,
    )

    print("\nKEY FINDINGS")
    for finding in result["key_findings"]:
        print(f"- {finding}")

    print("\nSUMMARY")
    print(result["summary"])


if __name__ == "__main__":
    main()
