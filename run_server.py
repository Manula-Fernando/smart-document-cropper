"""
Universal Launcher for Smart Document Cropper Enterprise Services.
Usage:
    python run_server.py --mode api       # Starts FastAPI REST API (port 8000)
    python run_server.py --mode mcp       # Starts Model Context Protocol (MCP) server
    python run_server.py --mode ui        # Starts Streamlit Interactive Dashboard (port 8501)
    python run_server.py --mode eval      # Runs evaluation benchmark on dataset
"""
import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Smart Document Cropper Service Launcher")
    parser.add_argument(
        "--mode",
        choices=["api", "mcp", "ui", "eval"],
        default="api",
        help="Service to launch: 'api' (FastAPI), 'mcp' (Model Context Protocol), 'ui' (Streamlit), 'eval' (Benchmark)"
    )
    parser.add_argument("--port", type=int, default=None, help="Port override")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    if args.mode == "api":
        port = args.port or 8008
        print(f"Starting Enterprise FastAPI REST server on http://{args.host}:{port} ...")
        print(f"Swagger API documentation: http://localhost:{port}/docs")
        cmd = [sys.executable, "-m", "uvicorn", "src.api.app:app", "--host", args.host, "--port", str(port)]
        subprocess.run(cmd)

    elif args.mode == "mcp":
        print("Starting Model Context Protocol (MCP) Server over stdio...")
        cmd = [sys.executable, "-m", "src.mcp.server"]
        subprocess.run(cmd)

    elif args.mode == "ui":
        port = args.port or 8501
        print(f"Starting Streamlit Enterprise Dashboard on http://localhost:{port} ...")
        cmd = [sys.executable, "-m", "streamlit", "run", "app_dashboard.py", "--server.port", str(port), "--server.address", args.host]
        subprocess.run(cmd)

    elif args.mode == "eval":
        print("Running comprehensive model benchmark on local dataset...")
        cmd = [sys.executable, "evaluate_models.py"]
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
