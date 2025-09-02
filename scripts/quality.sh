#!/bin/bash
# Run all code quality checks

set -e

echo "🚀 Running comprehensive code quality checks..."
echo ""

# Format code
./scripts/format.sh

echo ""

# Run linting
./scripts/lint.sh

echo ""

# Run tests
echo "🧪 Running tests..."
cd backend && python -m pytest tests/ -v

echo ""
echo "🎉 All quality checks completed successfully!"