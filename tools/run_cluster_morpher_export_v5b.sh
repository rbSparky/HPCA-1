#!/usr/bin/env bash
# Atomic, append-only native Morpher export runner for cluster experiments.
set -euo pipefail

toolchain="${1:?usage: run_cluster_morpher_export_v5b.sh TOOLCHAIN DFG_REL ARCH_REL X Y INITIAL_II PE_TYPE METHOD OUTPUT_DIR}"
dfg_rel="${2:?missing DFG_REL}"
arch_rel="${3:?missing ARCH_REL}"
x_dim="${4:?missing X}"
y_dim="${5:?missing Y}"
initial_ii="${6:?missing INITIAL_II}"
pe_type="${7:?missing PE_TYPE}"
method="${8:?missing METHOD}"
output_dir="${9:?missing OUTPUT_DIR}"

if [[ -x "$toolchain/bin/cgra_xml_mapper" ]]; then
  mapper="$toolchain/bin/cgra_xml_mapper"
else
  mapper="$toolchain/source/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper"
fi
source_root="$toolchain/source/Morpher_CGRA_Mapper"
if [[ "$dfg_rel" = /* ]]; then dfg="$dfg_rel"; else dfg="$source_root/$dfg_rel"; fi
if [[ "$arch_rel" = /* ]]; then arch="$arch_rel"; else arch="$source_root/$arch_rel"; fi
test -x "$mapper"
test -f "$dfg"
test -f "$arch"

if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite export output: $output_dir" >&2
  exit 41
fi
tmp_dir="${output_dir}.tmp.$$"
mkdir -p "$tmp_dir/export"
trap 'printf "INCOMPLETE_EXPORT_DIR=%s\n" "$tmp_dir" >&2' ERR
# printHyCUBEBinary derives its output filename from the DFG argument.  Passing
# a toolchain-owned DFG path therefore mutates the immutable toolchain and
# leaves the bitstream outside the atomic result.  Run from an exact local
# copy and verify its content hash before invoking Morpher.
dfg_input="$tmp_dir/$(basename "$dfg")"
cp -p "$dfg" "$dfg_input"
[[ "$(sha256sum "$dfg_input" | awk '{print $1}')" == \
   "$(sha256sum "$dfg" | awk '{print $1}')" ]]
if [[ -n "${FLOWADVANTAGE_AUX_INPUT_DIR:-}" ]]; then
  test -d "$FLOWADVANTAGE_AUX_INPUT_DIR"
  mkdir "$tmp_dir/derived_inputs"
  cp -a "$FLOWADVANTAGE_AUX_INPUT_DIR"/. "$tmp_dir/derived_inputs/"
fi

start="$(date +%s.%N)"
(
  cd "$tmp_dir"
  /usr/bin/time -v "$mapper" \
    -d "$dfg_input" -x "$x_dim" -y "$y_dim" -j "$arch" \
    -i "$initial_ii" -t "$pe_type" -m "$method" \
    --dump-flowadvantage-state "$tmp_dir/export"
) > "$tmp_dir/mapper.stdout.log" 2> "$tmp_dir/mapper.stderr.log" &
mapper_pid=$!

while kill -0 "$mapper_pid" 2>/dev/null; do
  now="$(date +%s.%N)"
  native_pid="$(pgrep -P "$mapper_pid" -x cgra_xml_mapper 2>/dev/null | head -1 || true)"
  sample_pid="${native_pid:-$mapper_pid}"
  cpu_rss="$(ps -o cputime=,rss= -p "$sample_pid" 2>/dev/null || true)"
  printf '{"timestamp":"%s","wrapper_pid":%d,"native_pid":%d,"stage":"native_mapping","wall_seconds":%.3f,"cpu_rss":"%s"}\n' \
    "$(date -Iseconds)" "$mapper_pid" "${native_pid:-0}" \
    "$(awk -v a="$start" -v b="$now" 'BEGIN {print b-a}')" \
    "$(sed 's/^[[:space:]]*//;s/[[:space:]]\\+/ /g' <<< "$cpu_rss")" \
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
  echo "native mapper failed with exit code $mapper_rc; preserved $tmp_dir" >&2
  exit "$mapper_rc"
fi

if [[ "$pe_type" == "HyCUBE_4REG" ]]; then
  mapfile -t bitstreams < <(find "$tmp_dir" -maxdepth 1 -type f -name '*_binary.bin')
  if (( ${#bitstreams[@]} != 1 )); then
    echo "expected exactly one HyCUBE bitstream, found ${#bitstreams[@]}" >&2
    exit 42
  fi
fi

python3 - "$tmp_dir/export" "$tmp_dir" "$start" "$end" > "$tmp_dir/summary.json" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
run_root = pathlib.Path(sys.argv[2])
docs = {}
parsed = {}
for name in ("dfg.json", "mrrg.json", "mapping.json"):
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing export: {path}")
    payload = path.read_bytes()
    parsed[name] = json.loads(payload)
    docs[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
bitstreams = sorted(run_root.glob("*_binary.bin"))
m = parsed["mapping.json"]
print(json.dumps({
    "schema": "cluster_morpher_export_v5b",
    "wall_seconds": float(sys.argv[4]) - float(sys.argv[3]),
    "ii": m.get("ii"),
    "operation_count": len(m.get("operations", [])),
    "route_count": len(m.get("routes", [])),
    "bitstreams": [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in bitstreams
    ],
    "documents": docs,
}, indent=2, sort_keys=True))
PY

{
  printf 'TOOLCHAIN=%q\n' "$toolchain"
  printf 'BINARY_SHA256=%q\n' "$(sha256sum "$mapper" | awk '{print $1}')"
  printf 'DFG_REL=%q\nDFG_SHA256=%q\n' "$dfg_rel" "$(sha256sum "$dfg" | awk '{print $1}')"
  printf 'ARCH_REL=%q\nARCH_SHA256=%q\n' "$arch_rel" "$(sha256sum "$arch" | awk '{print $1}')"
  printf 'X=%q\nY=%q\nINITIAL_II=%q\nPE_TYPE=%q\nMETHOD=%q\n' \
    "$x_dim" "$y_dim" "$initial_ii" "$pe_type" "$method"
  printf 'THREAD_LIMITS=%q\n' \
    "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset},OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-unset},MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset},NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-unset}"
  printf 'FINISHED_AT=%q\n' "$(date -Iseconds)"
} > "$tmp_dir/manifest.env"

mv "$tmp_dir" "$output_dir"
trap - ERR
cat "$output_dir/summary.json"
printf 'EXPORT_OUTPUT=%s\n' "$output_dir"
