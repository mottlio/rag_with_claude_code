#!/bin/bash
# Run linting checks

set -e

echo "🔍 Running ruff linting checks..."
ruff check backend/ main.py

echo "🧠 Running mypy type checking..."
mypy backend/ main.py

echo "✅ All linting checks passed!"