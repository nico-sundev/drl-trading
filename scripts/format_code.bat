@echo off
echo Running code formatting tools...

echo.
echo Running isort...
pre-commit run isort --all-files --hook-stage manual

echo.
echo Running black...
pre-commit run black --all-files --hook-stage manual

echo.
echo Running ruff --fix...
pre-commit run ruff --all-files

echo.
echo Code formatting complete!
