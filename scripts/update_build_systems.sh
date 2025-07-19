#!/bin/bash
# Script to update all projects to use modern hatchling build system

# List of projects to update
projects=(
    "drl-trading-training"
    "drl-trading-inference"
    "drl-trading-ingest"
    "drl-trading-execution"
    "drl-trading-strategy-example"
)

echo "🔧 Updating build systems to modern hatchling backend..."

for project in "${projects[@]}"; do
    if [ -f "$project/pyproject.toml" ]; then
        echo "📦 Updating $project..."

        # Replace build-system section with hatchling
        sed -i 's/requires = \["setuptools.*"\]/requires = ["hatchling"]/' "$project/pyproject.toml"
        sed -i 's/build-backend = "setuptools.build_meta"/build-backend = "hatchling.build"/' "$project/pyproject.toml"

        # Remove setuptools-specific configuration sections
        sed -i '/# Package discovery configuration/,/exclude = \["tests\*"\]/d' "$project/pyproject.toml"

        echo "✅ Updated $project"
    else
        echo "⚠️  Skipped $project (no pyproject.toml found)"
    fi
done

echo "🎉 All projects updated to use modern hatchling build system!"
echo ""
echo "Benefits:"
echo "  • Faster builds"
echo "  • Better uv integration"
echo "  • Automatic package discovery"
echo "  • Cleaner configuration"
