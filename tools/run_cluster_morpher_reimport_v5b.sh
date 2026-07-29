#!/usr/bin/env bash
# Atomically re-import a canonical mapping into native Morpher and require an
# exact semantic round trip.
set -euo pipefail

toolchain="${1:?usage: run_cluster_morpher_reimport_v5b.sh TOOLCHAIN DFG_REL ARCH_PATH MAPPING_PATH X Y II PE_TYPE OUTPUT_DIR}"
dfg_rel="${2:?missing DFG_REL}"
arch="${3:?missing ARCH_PATH}"
input_mapping="${4:?missing MAPPING_PATH}"
x_dim="${5:?missing X}"
y_dim="${6:?missing Y}"
ii="${7:?missing II}"
pe_type="${8:?missing PE_TYPE}"
output_dir="${9:?missing OUTPUT_DIR}"

if [[ -x "$toolchain/bin/cgra_xml_mapper" ]]; then
  mapper="$toolchain/bin/cgra_xml_mapper"
else
  mapper="$toolchain/source/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper"
fi
source_root="$toolchain/source/Morpher_CGRA_Mapper"
if [[ "$dfg_rel" = /* ]]; then dfg="$dfg_rel"; else dfg="$source_root/$dfg_rel"; fi
if [[ "$arch" != /* ]]; then arch="$source_root/$arch"; fi
test -x "$mapper"; test -f "$dfg"; test -f "$arch"; test -f "$input_mapping"
if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite reimport output: $output_dir" >&2
  exit 61
fi

tmp_dir="${output_dir}.tmp.$$"
mkdir -p "$tmp_dir/export"
trap 'printf "INCOMPLETE_REIMPORT_DIR=%s\n" "$tmp_dir" >&2' ERR

start="$(date +%s.%N)"
(
  cd "$tmp_dir"
  /usr/bin/time -v "$mapper" \
    -d "$dfg" -x "$x_dim" -y "$y_dim" -j "$arch" \
    -i "$ii" -t "$pe_type" -m 0 \
    --load-flowadvantage-mapping "$input_mapping" \
    --dump-flowadvantage-state "$tmp_dir/export"
) > "$tmp_dir/mapper.stdout.log" 2> "$tmp_dir/mapper.stderr.log"
end="$(date +%s.%N)"

grep -F "FlowAdvantage import PASS" "$tmp_dir/mapper.stdout.log" >/dev/null

python3 - "$input_mapping" "$tmp_dir/export/mapping.json" "$start" "$end" \
  > "$tmp_dir/semantic_comparison.json" <<'PY'
import json
import sys

before = json.load(open(sys.argv[1], encoding="utf-8"))
after = json.load(open(sys.argv[2], encoding="utf-8"))

def operations(document):
    return {
        str(item["native_node_key"]): (
            item["pe_id"], item["fu_id"], int(item["modulo_time"])
        )
        for item in document["operations"]
    }

def routes(document):
    return {
        str(item["edge_id"]): (
            tuple(item["ordered_resource_ids"]),
            tuple(item["ordered_link_ids"]),
            tuple(item.get("ordered_resource_latencies", [])),
            int(item["start_time"]),
            int(item["end_time"]),
        )
        for item in document["routes"]
    }

op_before, op_after = operations(before), operations(after)
route_before, route_after = routes(before), routes(after)
summary = {
    "schema": "morpher_semantic_roundtrip_v1",
    "wall_seconds": float(sys.argv[4]) - float(sys.argv[3]),
    "ii_before": int(before["ii"]),
    "ii_after": int(after["ii"]),
    "operation_count": len(op_before),
    "route_count": len(route_before),
    "placement_match": op_before == op_after,
    "route_match": route_before == route_after,
    "ii_match": int(before["ii"]) == int(after["ii"]),
}
print(json.dumps(summary, indent=2, sort_keys=True))
if not all(summary[key] for key in ("placement_match", "route_match", "ii_match")):
    raise SystemExit("semantic round-trip mismatch")
PY

{
  printf 'TOOLCHAIN=%q\nBINARY_SHA256=%q\n' \
    "$toolchain" "$(sha256sum "$mapper" | awk '{print $1}')"
  printf 'DFG=%q\nDFG_SHA256=%q\n' "$dfg" "$(sha256sum "$dfg" | awk '{print $1}')"
  printf 'ARCH=%q\nARCH_SHA256=%q\n' "$arch" "$(sha256sum "$arch" | awk '{print $1}')"
  printf 'INPUT_MAPPING=%q\nINPUT_MAPPING_SHA256=%q\n' \
    "$input_mapping" "$(sha256sum "$input_mapping" | awk '{print $1}')"
  printf 'II=%q\nPE_TYPE=%q\nFINISHED_AT=%q\n' "$ii" "$pe_type" "$(date -Iseconds)"
} > "$tmp_dir/manifest.env"

mv "$tmp_dir" "$output_dir"
trap - ERR
cat "$output_dir/semantic_comparison.json"
printf 'REIMPORT_OUTPUT=%s\n' "$output_dir"
