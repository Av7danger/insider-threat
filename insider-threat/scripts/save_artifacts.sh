#!/bin/bash
# Artifact Collection Script
#
# Purpose: Collect all models, metrics, and plots into a timestamped release directory
# This helps package everything needed for a release or deployment
#
# Usage:
#   bash scripts/save_artifacts.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RELEASE_DIR="artifacts/release-${TIMESTAMP}"

echo "Collecting artifacts for release: ${RELEASE_DIR}"
echo ""

# Create release directory
mkdir -p "${RELEASE_DIR}"

# Copy models
if [ -d "models" ]; then
    echo "Copying models..."
    cp -r models "${RELEASE_DIR}/"
    echo "✓ Models copied"
else
    echo "⚠ Warning: models/ directory not found"
fi

# Copy artifacts
if [ -d "artifacts" ]; then
    echo "Copying evaluation artifacts..."
    # Exclude release directories
    find artifacts -type f -not -path "artifacts/release-*/*" -exec cp --parent {} "${RELEASE_DIR}/" \;
    echo "✓ Artifacts copied"
else
    echo "⚠ Warning: artifacts/ directory not found"
fi

# Generate summary
echo ""
echo "Creating release summary..."

cat > "${RELEASE_DIR}/RELEASE_SUMMARY.txt" << EOF
Release Timestamp: ${TIMESTAMP}
Generated: $(date)

Contents:
EOF

# List models
if [ -d "${RELEASE_DIR}/models" ]; then
    echo "" >> "${RELEASE_DIR}/RELEASE_SUMMARY.txt"
    echo "Models:" >> "${RELEASE_DIR}/RELEASE_SUMMARY.txt"
    ls -lh "${RELEASE_DIR}/models"/*.pkl 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}' >> "${RELEASE_DIR}/RELEASE_SUMMARY.txt"
    ls -lh "${RELEASE_DIR}/models"/*.pt 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}' >> "${RELEASE_DIR}/RELEASE_SUMMARY.txt"
fi

# List key metrics
if [ -f "artifacts/summary_metrics.csv" ]; then
    echo "" >> "${RELEASE_DIR}/RELEASE_SUMMARY.txt"
    echo "Metrics:" >> "${RELEASE_DIR}/RELEASE_SUMMARY.txt"
    cat artifacts/summary_metrics.csv >> "${RELEASE_DIR}/RELEASE_SUMMARY.txt"
fi

echo ""
echo "Release Summary:"
cat "${RELEASE_DIR}/RELEASE_SUMMARY.txt"
echo ""
echo "✓ Release package created: ${RELEASE_DIR}"
echo ""
echo "Next steps:"
echo "  1. Review RELEASE_SUMMARY.txt"
echo "  2. Test models in release directory"
echo "  3. Create GitHub release if ready"

