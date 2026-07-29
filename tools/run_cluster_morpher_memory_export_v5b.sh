#!/usr/bin/env bash
# Atomic Morpher export using an authoritative checked-in banked DATA_LAYOUT.
set -euo pipefail

toolchain="${1:?usage: run_cluster_morpher_memory_export_v5b.sh TOOLCHAIN KERNEL OUTPUT_DIR}"
kernel="${2:?missing KERNEL (fix_fft or gemm_nt)}"
output_dir="${3:?missing OUTPUT_DIR}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "$toolchain/bin/cgra_xml_mapper" ]]; then
  mapper="$toolchain/bin/cgra_xml_mapper"
else
  mapper="$toolchain/source/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper"
fi
source_root="$toolchain/source/Morpher_CGRA_Mapper"
test -x "$mapper"

case "$kernel" in
  fix_fft)
    dfg_rel="applications/fix_fft_npb/fix_fft_INNERMOST_LN111_PartPred_DFG.xml"
    layout_rel="json_arch/fft_various_mem_archs/stdnoc_mem_dual_port_two_banked.json"
    variables=(fr manupa1 l.1253 conv31252 fi shl29 conv.i conv.i241)
    ;;
  gemm_nt)
    dfg_rel="applications/gemm_nt/gemm_nt_INNERMOST_LN111_PartPred_DFG.xml"
    layout_rel="json_arch/gemm_various_mem_archs/stdnoc_mem_dual_port_two_banked.json"
    variables=(A B ALPHA shr mul mul12 add49)
    ;;
  *)
    echo "unsupported kernel: $kernel" >&2
    exit 51
    ;;
esac

dfg="$source_root/$dfg_rel"
layout="$source_root/$layout_rel"
template="$source_root/json_arch/hycube_original_updatemem.json"
test -f "$dfg"
test -f "$layout"
test -f "$template"

if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite memory export output: $output_dir" >&2
  exit 52
fi
tmp_dir="${output_dir}.tmp.$$"
mkdir -p "$tmp_dir/export"
trap 'printf "INCOMPLETE_MEMORY_EXPORT_DIR=%s\n" "$tmp_dir" >&2' ERR

required=()
for variable in "${variables[@]}"; do
  required+=(--required-variable "$variable")
done
python3 "$project_root/scripts/derive_morpher_mem_alloc.py" \
  --source-architecture "$layout" \
  --bank-size 2048 \
  --output-csv "$tmp_dir/${kernel}_mem_alloc.txt" \
  --provenance-json "$tmp_dir/memory_layout_provenance.json" \
  "${required[@]}"
python3 "$source_root/update_mem_alloc.py" \
  "$template" "$tmp_dir/${kernel}_mem_alloc.txt" 2048 2 \
  "$tmp_dir/architecture_exact_layout.json"
python3 -m json.tool "$tmp_dir/architecture_exact_layout.json" >/dev/null

start="$(date +%s.%N)"
(
  cd "$tmp_dir"
  /usr/bin/time -v "$mapper" \
    -d "$dfg" -x 4 -y 4 -j "$tmp_dir/architecture_exact_layout.json" \
    -i 4 -t HyCUBE_4REG -m 0 \
    --dump-flowadvantage-state "$tmp_dir/export"
) > "$tmp_dir/mapper.stdout.log" 2> "$tmp_dir/mapper.stderr.log" &
mapper_pid=$!

while kill -0 "$mapper_pid" 2>/dev/null; do
  now="$(date +%s.%N)"
  cpu_rss="$(ps -o cputime=,rss= -p "$mapper_pid" 2>/dev/null || true)"
  printf '{"timestamp":"%s","pid":%d,"wall_seconds":%.3f,"cpu_rss":"%s"}\n' \
    "$(date -Iseconds)" "$mapper_pid" \
    "$(awk -v a="$start" -v b="$now" 'BEGIN {print b-a}')" \
    "$(sed 's/^[[:space:]]*//;s/[[:space:]]\+/ /g' <<< "$cpu_rss")" \
    > "$tmp_dir/heartbeat.json.tmp"
  mv "$tmp_dir/heartbeat.json.tmp" "$tmp_dir/heartbeat.json"
  sleep 15
done

set +e
wait "$mapper_pid"
mapper_rc=$?
set -e
end="$(date +%s.%N)"
if (( mapper_rc != 0 )); then
  printf 'MAPPER_EXIT_CODE=%s\n' "$mapper_rc" > "$tmp_dir/failure.env"
  echo "native memory mapper failed with exit code $mapper_rc; preserved $tmp_dir" >&2
  exit "$mapper_rc"
fi

python3 - "$tmp_dir/export" "$start" "$end" > "$tmp_dir/summary.json" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
docs = {}
parsed = {}
for name in ("dfg.json", "mrrg.json", "mapping.json"):
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing export: {path}")
    payload = path.read_bytes()
    parsed[name] = json.loads(payload)
    docs[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
mapping = parsed["mapping.json"]
print(json.dumps({
    "schema": "cluster_morpher_memory_export_v5b",
    "wall_seconds": float(sys.argv[3]) - float(sys.argv[2]),
    "ii": mapping.get("ii"),
    "operation_count": len(mapping.get("operations", [])),
    "route_count": len(mapping.get("routes", [])),
    "documents": docs,
}, indent=2, sort_keys=True))
PY

{
  printf 'TOOLCHAIN=%q\n' "$toolchain"
  printf 'BINARY_SHA256=%q\n' "$(sha256sum "$mapper" | awk '{print $1}')"
  printf 'KERNEL=%q\n' "$kernel"
  printf 'DFG_REL=%q\nDFG_SHA256=%q\n' "$dfg_rel" "$(sha256sum "$dfg" | awk '{print $1}')"
  printf 'LAYOUT_REL=%q\nLAYOUT_SHA256=%q\n' "$layout_rel" "$(sha256sum "$layout" | awk '{print $1}')"
  printf 'TEMPLATE_SHA256=%q\n' "$(sha256sum "$template" | awk '{print $1}')"
  printf 'ARCH_SHA256=%q\n' "$(sha256sum "$tmp_dir/architecture_exact_layout.json" | awk '{print $1}')"
  printf 'THREAD_LIMITS=%q\n' \
    "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset},OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-unset},MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset},NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-unset}"
  printf 'FINISHED_AT=%q\n' "$(date -Iseconds)"
} > "$tmp_dir/manifest.env"

mv "$tmp_dir" "$output_dir"
trap - ERR
cat "$output_dir/summary.json"
printf 'MEMORY_EXPORT_OUTPUT=%s\n' "$output_dir"
