#!/usr/bin/env bash
# cleanup.sh — safe automated cleanup + reports for Python repo
# Usage: ./cleanup.sh
set -euo pipefail

# 1) Create a branch for the cleanup
git checkout -b cleanup/remove-unused || git checkout cleanup/remove-unused

# 2) Install or ensure dev tools are available
python -m pip install --upgrade pip
python -m pip install ruff autoflake isort black vulture pipreqs

# 3) Auto-fix imports, unused-imports and small style issues
# Ruff fixes many issues (imports, formatting, lint)
ruff check --fix .

# Remove unused imports/variables (autoflake is more aggressive; safe to run after tests)
autoflake --remove-all-unused-imports --remove-unused-variables --in-place -r .

# Sort imports and format
isort .
black .

# 4) Run tests (if present) — don't fail the script if tests fail; inspect failures manually
if command -v pytest >/dev/null 2>&1; then
  pytest || echo "pytest failed — check tests before committing"
else
  echo "pytest not found; skipping tests"
fi

# 5) Find likely-dead code (manual review required)
# Vulture can yield false positives for dynamic usage; review vulture_report.txt before deleting anything.
vulture . --min-confidence 60 > vulture_report.txt || true
echo "Vulture report written to vulture_report.txt — review before deleting code"

# 6) Rebuild requirements from imports (optional; review generated file)
pipreqs --force . --savepath requirements-rebuilt.txt || true
echo "Generated requirements-rebuilt.txt — compare to existing requirements files"

# 7) Find large binaries or files that likely don't belong in git history
find . -type f -size +5M -not -path "./.git/*" -print > large_files.txt || true
echo "Large files listed in large_files.txt"

# 8) Show git status and staged changes summary
git add -A
git status --porcelain > git_changes_summary.txt
git diff --staged > staged_diff.patch || true
echo "Summary of changes saved to git_changes_summary.txt and staged_diff.patch"

echo "Cleanup script complete. Review vulture_report.txt, requirements-rebuilt.txt, large_files.txt, and git_changes_summary.txt before committing."