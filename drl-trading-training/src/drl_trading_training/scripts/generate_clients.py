"""
Script to generate REST API clients from OpenAPI specs.

This script uses OpenAPI Generator to create Python clients and DTOs
from the OpenAPI specification files.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Generate API clients from OpenAPI specs."""
    # Paths - script is at training/src/drl_trading_training/scripts/generate_clients.py
    script_dir = Path(__file__).parent
    training_src = script_dir.parent.parent
    project_root = training_src.parent.parent

    # Look for spec in ingest service first, then fallback to local
    ingest_specs_dir = project_root / "drl-trading-ingest" / "specs"
    local_specs_dir = training_src.parent / "specs"  # training/specs

    specs_dir = ingest_specs_dir if ingest_specs_dir.exists() else local_specs_dir
    output_dir = script_dir.parent / "adapter" / "rest" / "generated"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Spec file
    spec_file = specs_dir / "openapi.yaml"
    if not spec_file.exists():
        print(f"Error: OpenAPI spec not found at {spec_file}")
        print(f"Checked locations: {ingest_specs_dir}, {local_specs_dir}")
        sys.exit(1)

    # Generate client
    cmd = [
        "openapi-generator-cli", "generate",
        "-i", str(spec_file),
        "-g", "python",
        "-o", str(output_dir),
        "--additional-properties", "packageName=ingest_api_client",
        "--additional-properties", "projectName=DRL Trading Ingest API Client",
    ]

    print(f"Generating client from {spec_file} to {output_dir}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("Client generation completed successfully")
    else:
        print(f"Error generating client: {result.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
