#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting to format the project at: $PROJECT_ROOT"

if command -v clang-format &> /dev/null
then
    echo "Formatting C++/CUDA files with clang-format..."
    find "$PROJECT_ROOT" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" -o -name "*.cc" -o -name "*.hpp" \) \
    -not -path "*/build/*" \
    -not -path "*/third_party/*" | xargs clang-format -i -style=file
else
    echo "Warning: clang-format not found. Skipping C++/CUDA formatting."
fi

if command -v black &> /dev/null
then
    echo "Formatting Python files with black..."
    black "$PROJECT_ROOT" --exclude "/(build|third_party|venv)/"
elif command -v autopep8 &> /dev/null
then
    echo "Formatting Python files with autopep8..."
    find "$PROJECT_ROOT" -type f -name "*.py" \
    -not -path "*/build/*" \
    -not -path "*/venv/*" | xargs autopep8 --in-place --aggressive --aggressive
else
    echo "Warning: Neither black nor autopep8 found. Skipping Python formatting."
fi

echo "Formatting complete!"