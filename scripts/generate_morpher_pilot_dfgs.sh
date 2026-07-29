#!/usr/bin/env bash
# Generate the frozen missing pilot DFGs with Morpher's native LLVM pass.
#
# The image is immutable; all compiler intermediates live in an ephemeral
# container and only the XML/DOT contracts plus a complete log are published.
set -euo pipefail

IMAGE="${MORPHER_GENERATOR_IMAGE:-quotientflow-morpher-v5-original-patched}"
OUTPUT="${1:-results/revision_v5b/generated_dfg}"
OUTPUT_ABS="$(realpath -m "${OUTPUT}")"
if [[ -d "${OUTPUT_ABS}" ]] && find "${OUTPUT_ABS}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite nonempty DFG artifact directory: ${OUTPUT_ABS}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ABS}"

docker image inspect "${IMAGE}" >/dev/null
docker run --rm \
  --volume "${OUTPUT_ABS}:/flowadvantage-output" \
  "${IMAGE}" \
  bash -lc '
set -euo pipefail
cd /home/user/morpher/Morpher_DFG_Generator
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
PASS=/home/user/morpher/Morpher_DFG_Generator/build/src/libdfggenPass.so
test -f "${PASS}"

for kernel in array_cond hpcg trmm; do
  source_dir="/home/user/morpher/Morpher_DFG_Generator/benchmarks/morpher_benchmarks/${kernel}"
  output_dir="/flowadvantage-output/${kernel}"
  mkdir -p "${output_dir}"
  cd "${source_dir}"
  rm -f -- "${kernel}.ll" "${kernel}_opt.ll" \
    "${kernel}_opt_instrument.ll" \
    "${kernel}_PartPredDFG.xml" "${kernel}_PartPredDFG.dot"
  clang -D CGRA_COMPILER -target i386-unknown-linux-gnu \
    -Wno-implicit-function-declaration -Wno-format \
    -Wno-main-return-type -c -emit-llvm -O2 \
    -fno-tree-vectorize -fno-unroll-loops \
    "${kernel}.c" -S -o "${kernel}.ll"
  opt -gvn -mem2reg -memdep -memcpyopt -lcssa -loop-simplify \
    -licm -loop-deletion -indvars -simplifycfg -mergereturn -indvars \
    "${kernel}.ll" -S -o "${kernel}_opt.ll"
  opt -load "${PASS}" -fn "${kernel}" -nobanks 2 -banksize 2048 \
    -type PartPred -skeleton "${kernel}_opt.ll" \
    -S -o "${kernel}_opt_instrument.ll"
  test -s "${kernel}_PartPredDFG.xml"
  cp "${kernel}_PartPredDFG.xml" "${output_dir}/"
  cp "${kernel}_PartPredDFG.dot" "${output_dir}/"
  test -s "${kernel}_mem_alloc.txt"
  cp "${kernel}_mem_alloc.txt" "${output_dir}/"
  (
    cd "${output_dir}"
    sha256sum "${kernel}_PartPredDFG.xml" \
      "${kernel}_PartPredDFG.dot" "${kernel}_mem_alloc.txt" \
      > SHA256SUMS
  )
done
'
