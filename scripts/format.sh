#!/bin/bash
# Format code using black and ruff

set -e

echo "🎨 Formatting Python code with black..."
black backend/ main.py

echo "📦 Sorting imports and fixing code issues with ruff..."
ruff check --fix backend/ main.py

echo "✅ Code formatting completed!"