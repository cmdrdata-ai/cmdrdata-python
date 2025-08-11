#!/bin/bash

# Run comprehensive test suite for CmdrData SDK

echo "========================================="
echo "CmdrData SDK Test Suite"
echo "========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Installing dependencies..."
uv pip install -e ".[dev]"
echo ""

echo "Running tests with coverage..."
uv run pytest tests/ -v --cov=cmdrdata --cov-report=term-missing

echo ""
echo "Running type checking..."
uv run mypy cmdrdata --strict || echo "Type checking completed with warnings"

echo ""
echo "Checking code formatting..."
uv run black --check cmdrdata tests
uv run isort --check-only cmdrdata tests

echo ""
echo "========================================="
echo "Test suite completed!"
echo "========================================="