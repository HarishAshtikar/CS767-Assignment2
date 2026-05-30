"""Command-line orchestrator for DataSense."""

import argparse
from pathlib import Path

from agent import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DataSense CSV analysis agent.")
    parser.add_argument("--csv", default="sample_data.csv", help="Path to a CSV file to analyze.")
    parser.add_argument(
        "--goal",
        default="Understand the dataset, identify important patterns, and generate useful charts.",
        help="Natural-language analysis goal.",
    )
    parser.add_argument("--outputs-dir", default="outputs", help="Directory for generated charts.")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI web server instead.")
    parser.add_argument("--host", default="0.0.0.0", help="Host for --serve.")
    parser.add_argument("--port", type=int, default=8000, help="Port for --serve.")
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        uvicorn.run("server:app", host=args.host, port=args.port, reload=True)
        return

    result = run_agent(
        csv_path=str(Path(args.csv)),
        user_goal=args.goal,
        outputs_dir=args.outputs_dir,
    )

    print("\nKEY FINDINGS")
    for finding in result["key_findings"]:
        print(f"- {finding}")

    print("\nSUMMARY")
    print(result["summary"])


if __name__ == "__main__":
    main()
