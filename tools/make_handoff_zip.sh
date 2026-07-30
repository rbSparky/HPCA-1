#!/usr/bin/env bash
set -e

ZIPNAME="HPCA_QuotientFlow_Handoff_$(date +%Y%m%d).zip"
echo "Building: $ZIPNAME"
rm -f "$ZIPNAME"

zip -r "$ZIPNAME" \
  AGENTS.md README.md PAPER_RESULTS.md STARTING_PROMPT.md environment.yml \
  quotientflow/ flowadvantage/ scripts/ tests/ configs/ docs/ tools/ \
  results/paper_suite_v2/ \
  results/remote_runs/ \
  results/revision_v5b/ADAPTER_IMPLEMENTATION.md \
  results/revision_v5b/ADAPTER_VALIDATION.md \
  results/revision_v5b/CLUSTER_INCIDENT_20260729.md \
  results/revision_v5b/config_deviations.md \
  results/revision_v5b/CONTRACT_TRACE.md \
  results/revision_v5b/FRONTIER_RUNTIME_CALIBRATION.md \
  results/revision_v5b/gates.csv \
  results/revision_v5b/MEMORY_LAYOUT_AND_SIMULATION_PROVENANCE.md \
  results/revision_v5b/NATIVE_BASELINE_STATUS.md \
  results/revision_v5b/native_input_registry.json \
  results/revision_v5b/native_input_registry_v2.json \
  results/revision_v5b/native_input_registry_v3.json \
  results/revision_v5b/native_input_registry_v4.json \
  results/revision_v5b/NATIVE_PROPOSAL_IMPLEMENTATION.md \
  results/revision_v5b/NEGATIVE_TEST_REPORT.md \
  results/revision_v5b/PAPER_MAPPING_PROTOCOL.md \
  results/revision_v5b/PILOT_REPORT.md \
  results/revision_v5b/REPORT.md \
  results/revision_v5b/RUN_SUMMARY.txt \
  results/revision_v5b/generated_dfg \
  results/revision_v5b/generated_dfg_v2 \
  results/revision_v5b/length_medium_baseline_v1 \
  results/revision_v5b/length_medium_baseline_ext_v1 \
  results/revision_v5b/logs \
  -x "*/__pycache__/*" \
  -x "*/*.pyc" \
  -x "*/.pytest_cache/*" \
  -x "*/Zone.Identifier"

du -sh "$ZIPNAME"
echo "Created: $ZIPNAME"

# Copy to Windows Desktop
cp "$ZIPNAME" /mnt/c/Users/Rishabh/Desktop/
echo "Copied to Desktop: /mnt/c/Users/Rishabh/Desktop/$ZIPNAME"
