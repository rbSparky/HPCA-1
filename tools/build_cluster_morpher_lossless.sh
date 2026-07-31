#!/usr/bin/env bash
set -euo pipefail

# Build the lossless-route bridge from the already validated fixed17 source.
# Only route-record serialization is changed; native mapping semantics remain
# those of the validated fixed17 source.
base="${1:?fixed17 source directory}"
root="${2:?output toolchain directory}"
if [[ -f "$root/READY" ]]; then
  echo "READY_EXISTING=$root"
  exit 0
fi
if [[ -e "$root" ]]; then
  echo "refusing to overwrite incomplete build: $root" >&2
  exit 12
fi
mkdir -p "$root"
cp -a "$base" "$root/source"
src="$root/source"

python3 - "$src/include/morpher/arch/CGRA.h" "$src/src/arch/CGRA.cpp" <<'PY'
from pathlib import Path
import sys

header = Path(sys.argv[1])
source = Path(sys.argv[2])
h = header.read_text(encoding="utf-8")
needle = "\tstring json_file;\n"
if h.count(needle) != 1:
    raise SystemExit("CGRA.h anchor is not unique")
h = h.replace(needle, needle +
    "\t// Exact validated route records retained after external import.\n"
    "\tstd::map<std::string, json> importedFlowAdvantageRoutes;\n")
header.write_text(h, encoding="utf-8")

s = source.read_text(encoding="utf-8")
needle = ('\t\tjson rr; rr["edge_id"] = fa_node_key(&src) + "->" + fa_node_key(dstnode); '
          'rr["source_node"] = src.idx; rr["destination_node"] = dstnode->idx; '
          'rr["source_node_key"] = fa_node_key(&src); '
          'rr["destination_node_key"] = fa_node_key(dstnode); '
          'rr["start_time"] = start->getLat();\n')
if s.count(needle) != 1:
    raise SystemExit("CGRA.cpp route anchor is not unique")
replacement = ('\t\tstring imported_key = fa_node_key(&src) + "->" + fa_node_key(dstnode);\n'
    '\t\tauto imported_route = importedFlowAdvantageRoutes.find(imported_key);\n'
    '\t\tif (imported_route != importedFlowAdvantageRoutes.end()) {\n'
    '\t\t\tmapj["routes"].push_back(imported_route->second);\n'
    '\t\t\tcontinue;\n'
    '\t\t}\n' + needle.replace(
        'fa_node_key(&src) + "->" + fa_node_key(dstnode)', 'imported_key'))
s = s.replace(needle, replacement)
needle = ('\tif (m.value("architecture_hash", "") != fa_hash_text(fa_file_text(json_file) + ":" '
          '+ to_string(get_x_max()) + ":" + to_string(get_y_max()) + ":" + to_string(get_t_max()))) '
          '{ error = "architecture-hash mismatch"; return false; }\n')
if s.count(needle) != 1:
    raise SystemExit("CGRA.cpp import anchor is not unique")
s = s.replace(needle, needle + "\timportedFlowAdvantageRoutes.clear();\n")
needle = '\t\tsrc->routingPorts.push_back(make_pair(ports.back(),dstid));\n'
if s.count(needle) != 1:
    raise SystemExit("CGRA.cpp route-install anchor is not unique")
s = s.replace(needle,
    '\t\timportedFlowAdvantageRoutes[fa_node_key(src) + "->" + fa_node_key(dst)] = r;\n' + needle)
source.write_text(s, encoding="utf-8")
PY

sed -i \
  -e 's|set(CMAKE_C_COMPILER "gcc-7")|set(CMAKE_C_COMPILER "/usr/bin/gcc")|' \
  -e 's|set(CMAKE_CXX_COMPILER "/usr/bin/g++-7")|set(CMAKE_CXX_COMPILER "/usr/bin/g++")|' \
  "$src/CMakeLists.txt" "$src/src/CMakeLists.txt"
mkdir "$src/build"
cmake_bin="${CMAKE_BIN:-$(command -v cmake || true)}"
if [[ -z "$cmake_bin" ]]; then
  cmake_env="$root/cmake-env"
  /usr/bin/python3 -m venv "$cmake_env"
  "$cmake_env/bin/python" -m pip install --disable-pip-version-check --no-input cmake==3.31.6 \
    > "$root/cmake_install.log" 2>&1
  cmake_bin="$cmake_env/bin/cmake"
fi
test -x "$cmake_bin"
"$cmake_bin" -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  > "$root/cmake.log" 2>&1
"$cmake_bin" --build "$src/build" --parallel 2 > "$root/build.log" 2>&1
test -x "$src/build/src/cgra_xml_mapper"
sha256sum "$src/build/src/cgra_xml_mapper" > "$root/BINARY_SHA256"
printf 'READY\n' > "$root/READY"
echo "READY_NEW=$root"
