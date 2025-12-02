#!/usr/bin/env python3
"""
Generate OpenAPI clients for DRL Trading services.

This script finds OpenAPI specifications from the ingest service and generates
Python clients for dependent services.
"""

import subprocess
import sys
from pathlib import Path


def find_openapi_spec(ingest_dir: Path) -> Path:
    """Find the OpenAPI spec file in the ingest service directory."""
    specs_dir = ingest_dir / "specs"
    if not specs_dir.exists():
        raise FileNotFoundError(f"Specs directory not found: {specs_dir}")

    # Look for common OpenAPI spec filenames
    spec_names = ["openapi.yaml", "openapi.yml", "swagger.yaml", "swagger.yml"]

    for spec_name in spec_names:
        spec_path = specs_dir / spec_name
        if spec_path.exists():
            return spec_path

    raise FileNotFoundError(f"No OpenAPI spec found in {specs_dir}")


def generate_client(spec_path: Path, output_dir: Path, package_name: str) -> None:
    """Generate Python client from OpenAPI spec."""
    print(f"Generating client for {package_name} from {spec_path}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run OpenAPI Generator
    cmd = [
        "openapi-generator-cli", "generate",
        "-i", str(spec_path),
        "-g", "python",
        "-o", str(output_dir),
        "--package-name", package_name,
        "--additional-properties", "packageVersion=1.0.0",
        "--additional-properties", "projectName=" + package_name,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Client generation successful for {package_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating client for {package_name}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def main() -> None:
    """Main entry point."""
    # Get the project root (assuming script is in scripts/ directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Find the ingest service spec
    ingest_dir = project_root / "drl-trading-ingest"
    try:
        spec_path = find_openapi_spec(ingest_dir)
        print(f"Found OpenAPI spec: {spec_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Generate clients for dependent services
    services = [
        {
            "name": "drl-trading-training",
            "client_dir": "src/drl_trading_training/generated_clients/ingest",
            "package_name": "drl_trading_ingest_client"
        }
    ]

    for service in services:
        service_dir = project_root / service["name"]
        if not service_dir.exists():
            print(f"Warning: Service directory not found: {service_dir}")
            continue

        client_output_dir = service_dir / service["client_dir"]
        try:
            generate_client(spec_path, client_output_dir, service["package_name"])
        except Exception as e:
            print(f"Failed to generate client for {service['name']}: {e}")
            sys.exit(1)

    print("All client generation completed successfully")


if __name__ == "__main__":
    main()
