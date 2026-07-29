#!/usr/bin/env bash
# Run one native Morpher fixed14 export in an append-only output directory.
set -euo pipefail

toolchain="${1:?usage: run_cluster_morpher_smoke_v5b.sh TOOLCHAIN OUTPUT_DIR}"
output_dir="${2:?missing OUTPUT_DIR}"
if [[ -x "$toolchain/bin/cgra_xml_mapper" ]]; then
  mapper="$toolchain/bin/cgra_xml_mapper"
else
  mapper="$toolchain/source/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper"
fi
source_root="$toolchain/source/Morpher_CGRA_Mapper"

test -x "$mapper"
if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite smoke output: $output_dir" >&2
  exit 21
fi

tmp_dir="${output_dir}.tmp.$$"
mkdir -p "$tmp_dir/export"
trap 'printf "INCOMPLETE_SMOKE_DIR=%s\n" "$tmp_dir" >&2' ERR

dfg="$source_root/applications/array_add/array_add_INNERMOST_LN1_PartPred_DFG.xml"
arch="$source_root/applications/hycube/array_add/hycube_original_mem.json"
test -f "$dfg"
test -f "$arch"

start="$(date +%s.%N)"
(
  cd "$tmp_dir"
  /usr/bin/time -v "$mapper" \
    -d "$dfg" -x 4 -y 4 -j "$arch" -i 0 -t HyCUBE_4REG -m 0 \
    --dump-flowadvantage-state "$tmp_dir/export"
) > "$tmp_dir/mapper.stdout.log" 2> "$tmp_dir/mapper.stderr.log"
end="$(date +%s.%N)"

python3 - "$tmp_dir/export" "$start" "$end" > "$tmp_dir/summary.json" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
docs = {}
for name in ("dfg.json", "mrrg.json", "mapping.json"):
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing export: {path}")
    payload = path.read_bytes()
    docs[name] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "json_type": type(json.loads(payload)).__name__,
    }
mapping = json.loads((root / "mapping.json").read_text())
print(json.dumps({
    "schema": "cluster_morpher_smoke_v5b",
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
  printf 'DFG_SHA256=%q\n' "$(sha256sum "$dfg" | awk '{print $1}')"
  printf 'ARCH_SHA256=%q\n' "$(sha256sum "$arch" | awk '{print $1}')"
  printf 'THREAD_LIMITS=%q\n' \
    "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset},OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-unset},MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset},NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-unset}"
  printf 'FINISHED_AT=%q\n' "$(date -Iseconds)"
} > "$tmp_dir/manifest.env"

mv "$tmp_dir" "$output_dir"
trap - ERR
cat "$output_dir/summary.json"
printf 'SMOKE_OUTPUT=%s\n' "$output_dir"
