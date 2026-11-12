#!/usr/bin/env bash
# generate_sync_report.sh — Generate a markdown sync report for TransformerEngine-FL
# Usage: bash scripts/generate_sync_report.sh [upstream_branch] [output_file]

set -euo pipefail

UPSTREAM_BRANCH="${1:-release_v2.14}"
OUTPUT="${2:-SYNC_REPORT.md}"

echo "Generating sync report..."

# Gather data
MERGE_COMMIT=$(git log --oneline --merges -1 --format="%H" 2>/dev/null || echo "N/A")
MERGE_COMMIT_SHORT=$(git log --oneline --merges -1 --format="%h" 2>/dev/null || echo "N/A")
UPSTREAM_SHA=$(git rev-parse "upstream/${UPSTREAM_BRANCH}" 2>/dev/null || echo "N/A")
UPSTREAM_SHA_SHORT=$(git rev-parse --short "upstream/${UPSTREAM_BRANCH}" 2>/dev/null || echo "N/A")
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "N/A")
SYNC_DATE=$(date +"%Y-%m-%d %H:%M:%S %Z")
MAIN_HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")

# Plugin directory status
PLUGIN_STATUS="❌ Missing"
if [ -d "transformer_engine/common/plugin" ]; then
    PCOUNT=$(find transformer_engine/common/plugin -type f -name "*.h" | wc -l | tr -d ' ')
    PLUGIN_STATUS="✅ Present ($PCOUNT header files)"
fi

# CUDA patches status
CUDA_STATUS="❌ Missing"
if [ -d "transformer_engine/common/cuda_patches" ]; then
    CCOUNT=$(find transformer_engine/common/cuda_patches -type f | wc -l | tr -d ' ')
    CUDA_STATUS="✅ Present ($CCOUNT files)"
fi

# OP API check
API_STATUS="❌ Not found"
if [ -d "transformer_engine/common/plugin" ]; then
    REG=$(grep -rl "register_plugin" transformer_engine/common/plugin/ 2>/dev/null | wc -l | tr -d ' ')
    UNREG=$(grep -rl "unregister_plugin" transformer_engine/common/plugin/ 2>/dev/null | wc -l | tr -d ' ')
    if [ "$REG" -gt 0 ] && [ "$UNREG" -gt 0 ]; then
        API_STATUS="✅ register_plugin() and unregister_plugin() present"
    elif [ "$REG" -gt 0 ]; then
        API_STATUS="⚠️ register_plugin() found, unregister_plugin() missing"
    fi
fi

# Build config check
SETUP_STATUS="N/A"
CMAKE_STATUS="N/A"
[ -f "setup.py" ] && { grep -qi "plugin" setup.py && SETUP_STATUS="✅ Plugin targets present" || SETUP_STATUS="⚠️ No plugin references"; }
[ -f "CMakeLists.txt" ] && { grep -qi "plugin" CMakeLists.txt && CMAKE_STATUS="✅ Plugin targets present" || CMAKE_STATUS="⚠️ No plugin references"; }

# Write report
cat > "$OUTPUT" << EOF
# TransformerEngine-FL Upstream Sync Report

## Sync Summary

| Field | Value |
|-------|-------|
| Date | ${SYNC_DATE} |
| Upstream | Nvidia/TransformerEngine |
| Upstream Branch | ${UPSTREAM_BRANCH} |
| Upstream Commit | \`${UPSTREAM_SHA_SHORT}\` (${UPSTREAM_SHA}) |
| Merge Commit | \`${MERGE_COMMIT_SHORT}\` (${MERGE_COMMIT}) |
| Current Branch | ${CURRENT_BRANCH} |
| HEAD | \`${MAIN_HEAD}\` |

## Plugin System Status

| Component | Status |
|-----------|--------|
| Plugin directory | ${PLUGIN_STATUS} |
| CUDA patches | ${CUDA_STATUS} |
| OP API interfaces | ${API_STATUS} |
| setup.py | ${SETUP_STATUS} |
| CMakeLists.txt | ${CMAKE_STATUS} |

## CI/CD Results

| Level | Test | Status |
|-------|------|--------|
| 1 | Compile test (\`pip install -e .\`) | ⬜ Not run |
| 2 | Unit tests (\`pytest tests/unit/\`) | ⬜ Not run |
| 3 | Integration tests (\`pytest tests/pytorch/\`) | ⬜ Not run |
| 4 | Plugin-specific tests | ⬜ Not run |
| 5 | End-to-end tests | ⬜ Not run |

> Update the CI/CD table above as each level completes.

## Conflict Resolution Summary

> Fill in after conflict resolution:
>
> - P0 conflicts resolved: _count_
> - P1 conflicts resolved: _count_
> - P2 conflicts resolved: _count_
> - Total files with conflicts: _count_

## Rollback Information

If issues are discovered post-merge:

\`\`\`bash
git revert -m 1 ${MERGE_COMMIT}
git push origin main
\`\`\`

## Notes

_Add any additional notes about the sync here._
EOF

echo "✅ Sync report written to: $OUTPUT"
echo "   Merge commit: $MERGE_COMMIT_SHORT"
echo "   Upstream SHA: $UPSTREAM_SHA_SHORT"
