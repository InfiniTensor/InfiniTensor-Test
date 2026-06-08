#!/usr/bin/env bash
# Run all standard suites sequentially and produce a comparison report.
#
# Usage:
#   ./scripts/run_all_suites.sh [output_dir]
#
# Each suite writes its result file under <output_dir>/. After all runs,
# a summary comparison report is written too.

set -euo pipefail

OUTPUT_DIR="${1:-data/results/batch_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"

echo "==> Output dir: $OUTPUT_DIR"
echo

# Each suite: name sources [extra args]
SUITES=(
    "smoke"
    "torchvision"
    "timm_small"
    "synthetic"
    "correctness"
)

# Suite config (which sources / limits)
declare -A SUITE_SOURCES=(
    [smoke]="--limit 10"
    [torchvision]="--sources torchvision"
    [timm_small]="--sources timm_small"
    [synthetic]="--sources synthetic_full"
    [correctness]="--limit 50 --check-correctness --tolerance 1e-3"
)

for suite in "${SUITES[@]}"; do
    echo "==> Running suite: $suite"
    out="$OUTPUT_DIR/${suite}.json"
    # shellcheck disable=SC2086
    infinitensor-test-run --suite "$suite" ${SUITE_SOURCES[$suite]} \
        2>"$OUTPUT_DIR/${suite}.log" \
        | tee "$OUTPUT_DIR/${suite}.summary.json"
    echo "    results: $out"
    echo
done

# Render reports
echo "==> Rendering reports"
for suite in "${SUITES[@]}"; do
    json="$OUTPUT_DIR/${suite}.json"
    if [ -f "$json" ]; then
        infinitensor-test-report "$json" -f both -o "$OUTPUT_DIR/${suite}_report"
    fi
done

# Compare smoke vs full if both exist
if [ -f "$OUTPUT_DIR/smoke.json" ] && [ -f "$OUTPUT_DIR/torchvision.json" ]; then
    echo "==> Comparing smoke vs torchvision"
    infinitensor-test-compare "$OUTPUT_DIR/smoke.json" "$OUTPUT_DIR/torchvision.json" \
        -o "$OUTPUT_DIR/compare_smoke_vs_torchvision.md"
fi

echo
echo "==> All done. See: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
