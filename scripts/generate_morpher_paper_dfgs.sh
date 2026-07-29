#!/usr/bin/env bash
# Generate the four additional compiler-derived paper-suite DFGs from the
# pinned Morpher LLVM pass.  Each source is compiled in a fresh ephemeral
# directory; the immutable image checkout is never modified.
set -euo pipefail

IMAGE="${MORPHER_GENERATOR_IMAGE:-quotientflow-morpher-v5-original-patched}"
OUTPUT="${1:-results/paper_suite_v2/generated_dfg}"
OUTPUT_ABS="$(realpath -m "${OUTPUT}")"

if [[ -d "${OUTPUT_ABS}" ]] &&
   find "${OUTPUT_ABS}" -mindepth 1 -print -quit | grep -q .; then
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
generator=/home/user/morpher/Morpher_DFG_Generator
mkdir -p /tmp/dfggen-build
cd /tmp/dfggen-build
cmake "${generator}"
make -j"$(nproc)"
pass=/tmp/dfggen-build/src/libdfggenPass.so
test -f "${pass}"

descriptors=(
  "fir|opencgra_benchmarks/fir/kernel.c|kernel"
  "conv2|cgrame_microbench/conv2/conv2.c|main"
  "conv3|cgrame_microbench/conv3/conv3.c|main"
  "mac|cgrame_microbench/mac/mac.c|main"
)

for descriptor in "${descriptors[@]}"; do
  IFS="|" read -r kernel relative_source function_name <<< "${descriptor}"
  work="/tmp/flowadvantage-dfg-${kernel}"
  output="/flowadvantage-output/${kernel}"
  mkdir -p "${work}" "${output}"
  cp "${generator}/benchmarks/${relative_source}" "${work}/input.c"
  cd "${work}"

  clang -D CGRA_COMPILER -target i386-unknown-linux-gnu \
    -Wno-implicit-function-declaration -Wno-format \
    -Wno-main-return-type -c -emit-llvm -O2 \
    -fno-tree-vectorize -fno-unroll-loops \
    input.c -S -o input.ll
  opt -gvn -mem2reg -memdep -memcpyopt -lcssa -loop-simplify \
    -licm -loop-deletion -indvars -simplifycfg -mergereturn -indvars \
    input.ll -S -o input_opt.ll
  opt -load "${pass}" -fn "${function_name}" -nobanks 2 -banksize 2048 \
    -type PartPred -skeleton input_opt.ll \
    -S -o input_instrument.ll

  mapfile -t xmls < <(find "${work}" -maxdepth 1 -type f \
    -name "*_PartPredDFG.xml" -print)
  mapfile -t memories < <(find "${work}" -maxdepth 1 -type f \
    -name "*_mem_alloc.txt" -print)
  (( ${#xmls[@]} == 1 )) ||
    { echo "${kernel}: expected one XML, found ${#xmls[@]}" >&2; exit 31; }
  dot="${xmls[0]%.xml}.dot"
  [[ -s "${dot}" ]] ||
    { echo "${kernel}: missing DOT matching ${xmls[0]}" >&2; exit 32; }
  (( ${#memories[@]} == 1 )) ||
    { echo "${kernel}: expected one memory contract, found ${#memories[@]}" >&2; exit 33; }

  cp "${xmls[0]}" "${output}/${kernel}_PartPredDFG.xml"
  cp "${dot}" "${output}/${kernel}_PartPredDFG.dot"
  cp "${memories[0]}" "${output}/${kernel}_mem_alloc.txt"
  cp input.c "${output}/${kernel}_source.c"
  {
    printf "kernel=%s\nfunction=%s\nsource=%s\n" \
      "${kernel}" "${function_name}" "${relative_source}"
    printf "morpher_commit=%s\n" \
      "9a9dce7aea521f1d5ef33686f57ca84864edb3c9"
    clang --version | head -1
    opt --version | head -1
  } > "${output}/GENERATION_MANIFEST.txt"
  (
    cd "${output}"
    sha256sum \
      "${kernel}_PartPredDFG.xml" \
      "${kernel}_PartPredDFG.dot" \
      "${kernel}_mem_alloc.txt" \
      "${kernel}_source.c" \
      GENERATION_MANIFEST.txt > SHA256SUMS
  )
done
'
