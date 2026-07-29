#!/usr/bin/env bash
# Atomic Morpher export for a compiler-generated DFG and DATA_LAYOUT contract.
set -euo pipefail

toolchain="${1:?usage: run_cluster_morpher_generated_memory_export_v5b.sh TOOLCHAIN DFG MEM_ALLOC OUTPUT_DIR}"
dfg="${2:?missing absolute DFG path}"
mem_alloc="${3:?missing absolute DATA_LAYOUT path}"
output_dir="${4:?missing output directory}"

if [[ -x "${toolchain}/bin/cgra_xml_mapper" ]]; then
  mapper="${toolchain}/bin/cgra_xml_mapper"
else
  mapper="${toolchain}/source/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper"
fi
mapper_root="${toolchain}/source/Morpher_CGRA_Mapper"
template="${mapper_root}/json_arch/hycube_original_updatemem.json"
update_script="${mapper_root}/update_mem_alloc.py"
test -x "${mapper}"
test -f "${dfg}"
test -f "${mem_alloc}"
test -f "${template}"
test -f "${update_script}"

if [[ -e "${output_dir}" ]]; then
  echo "refusing to overwrite memory export output: ${output_dir}" >&2
  exit 61
fi
tmp_dir="${output_dir}.tmp.$$"
mkdir -p "${tmp_dir}/export"
trap 'printf "INCOMPLETE_GENERATED_MEMORY_EXPORT=%s\n" "${tmp_dir}" >&2' ERR

cp -p "${dfg}" "${tmp_dir}/$(basename "${dfg}")"
cp -p "${mem_alloc}" "${tmp_dir}/$(basename "${mem_alloc}")"
dfg_input="${tmp_dir}/$(basename "${dfg}")"
mem_input="${tmp_dir}/$(basename "${mem_alloc}")"
[[ "$(sha256sum "${dfg_input}" | awk '{print $1}')" == \
   "$(sha256sum "${dfg}" | awk '{print $1}')" ]]
[[ "$(sha256sum "${mem_input}" | awk '{print $1}')" == \
   "$(sha256sum "${mem_alloc}" | awk '{print $1}')" ]]

python3 "${update_script}" \
  "${template}" "${mem_input}" 2048 2 \
  "${tmp_dir}/architecture_exact_layout.json"
python3 -m json.tool "${tmp_dir}/architecture_exact_layout.json" >/dev/null

start="$(date +%s.%N)"
(
  cd "${tmp_dir}"
  /usr/bin/time -v "${mapper}" \
    -d "${dfg_input}" -x 4 -y 4 \
    -j "${tmp_dir}/architecture_exact_layout.json" \
    -i 0 -t HyCUBE_4REG -m 0 \
    --dump-flowadvantage-state "${tmp_dir}/export"
) > "${tmp_dir}/mapper.stdout.log" 2> "${tmp_dir}/mapper.stderr.log" &
mapper_pid=$!

while kill -0 "${mapper_pid}" 2>/dev/null; do
  now="$(date +%s.%N)"
  native_pid="$(pgrep -P "${mapper_pid}" -x cgra_xml_mapper 2>/dev/null | head -1 || true)"
  sample_pid="${native_pid:-${mapper_pid}}"
  cpu_rss="$(ps -o cputime=,rss= -p "${sample_pid}" 2>/dev/null || true)"
  printf '{"timestamp":"%s","wrapper_pid":%d,"native_pid":%d,"stage":"native_mapping","wall_seconds":%.3f,"cpu_rss":"%s"}\n' \
    "$(date -Iseconds)" "${mapper_pid}" "${native_pid:-0}" \
    "$(awk -v a="${start}" -v b="${now}" 'BEGIN {print b-a}')" \
    "$(sed 's/^[[:space:]]*//;s/[[:space:]]\+/ /g' <<< "${cpu_rss}")" \
    > "${tmp_dir}/heartbeat.json.tmp"
  mv "${tmp_dir}/heartbeat.json.tmp" "${tmp_dir}/heartbeat.json"
  sleep 15
done

set +e
wait "${mapper_pid}"
mapper_rc=$?
set -e
end="$(date +%s.%N)"
if (( mapper_rc != 0 )); then
  printf 'MAPPER_EXIT_CODE=%s\n' "${mapper_rc}" > "${tmp_dir}/failure.env"
  echo "native generated-memory mapper failed; preserved ${tmp_dir}" >&2
  exit "${mapper_rc}"
fi

python3 - "${tmp_dir}/export" "${tmp_dir}" "${start}" "${end}" \
  > "${tmp_dir}/summary.json" <<'PY'
import hashlib
import json
import pathlib
import sys

export = pathlib.Path(sys.argv[1])
run_root = pathlib.Path(sys.argv[2])
documents = {}
parsed = {}
for name in ("dfg.json", "mrrg.json", "mapping.json"):
    path = export / name
    if not path.is_file():
        raise SystemExit(f"missing export: {path}")
    payload = path.read_bytes()
    parsed[name] = json.loads(payload)
    documents[name] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
mapping = parsed["mapping.json"]
print(json.dumps({
    "schema": "cluster_morpher_generated_memory_export_v5b",
    "wall_seconds": float(sys.argv[4]) - float(sys.argv[3]),
    "ii": mapping.get("ii"),
    "operation_count": len(mapping.get("operations", [])),
    "route_count": len(mapping.get("routes", [])),
    "documents": documents,
    "bitstreams": [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(run_root.glob("*_binary.bin"))
    ],
}, indent=2, sort_keys=True))
PY

{
  printf 'TOOLCHAIN=%q\n' "${toolchain}"
  printf 'BINARY_SHA256=%q\n' "$(sha256sum "${mapper}" | awk '{print $1}')"
  printf 'DFG=%q\nDFG_SHA256=%q\n' "${dfg}" "$(sha256sum "${dfg}" | awk '{print $1}')"
  printf 'MEM_ALLOC=%q\nMEM_ALLOC_SHA256=%q\n' \
    "${mem_alloc}" "$(sha256sum "${mem_alloc}" | awk '{print $1}')"
  printf 'TEMPLATE_SHA256=%q\n' "$(sha256sum "${template}" | awk '{print $1}')"
  printf 'ARCH_SHA256=%q\n' \
    "$(sha256sum "${tmp_dir}/architecture_exact_layout.json" | awk '{print $1}')"
  printf 'THREAD_LIMITS=%q\n' \
    "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset},OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-unset},MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset},NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-unset}"
  printf 'FINISHED_AT=%q\n' "$(date -Iseconds)"
} > "${tmp_dir}/manifest.env"

mv "${tmp_dir}" "${output_dir}"
trap - ERR
cat "${output_dir}/summary.json"
printf 'MEMORY_EXPORT_OUTPUT=%s\n' "${output_dir}"
