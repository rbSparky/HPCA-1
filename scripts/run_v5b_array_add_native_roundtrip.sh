#!/usr/bin/env bash
set -euo pipefail

# Runs the real Morpher compiler/trace/simulator pipeline and then replaces
# only its mapper configuration with a fresh-MRRG FlowAdvantage import.  All
# DFG, memory allocation, and trace artifacts therefore originate from the
# same compiler invocation.
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${FLOW_OUT_DIR:-$ROOT_DIR/results/revision_v5b/reference_mappings/array_add/A0_full_pipeline}"
MORPHER_IMAGE="${MORPHER_IMAGE:-quotientflow-morpher-v5b-native}"
mkdir -p "$OUT_DIR"

docker run --rm -v "$OUT_DIR:/out" "$MORPHER_IMAGE" bash -lc '
set -euo pipefail
cd /home/user/morpher
bash build_all.sh
python -u run_morpher.py morpher_benchmarks/array_add/array_add.c array_add

MAP=/home/user/morpher/Morpher_CGRA_Mapper/benchmarks/hycube/morpher_benchmarks/array_add
SIM=/home/user/morpher/hycube_simulator/benchmarks/morpher_benchmarks/array_add
DFG=array_add_PartPredDFG.xml
ARCH=hycube_original_mem.json
BASE_BIN=$(find "$MAP" -maxdepth 1 -name "*_binary.bin" -print -quit)
test -n "$BASE_BIN"
II=$(basename "$BASE_BIN" | sed -n "s/.*_II=\([0-9][0-9]*\)_.*/\1/p")
test -n "$II"

cd "$MAP"
/home/user/morpher/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper -d "$DFG" -x 4 -y 4 -j "$ARCH" -i "$II" -t HyCUBE_4REG -m 0 --dump-flowadvantage-state /out/native_dump
cp "$DFG" /out/array_add_PartPredDFG.xml
cp "$ARCH" /out/hycube_original_mem.json
cp "$MAP"/*_II="${II}"_*_binary.bin /out/native_mapper.bin
/home/user/morpher/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper -d "$DFG" -x 4 -y 4 -j "$ARCH" -i "$II" -t HyCUBE_4REG -m 0 --load-flowadvantage-mapping /out/native_dump/mapping.json --dump-flowadvantage-state /out/native_reimport
cp "$MAP"/*_II="${II}"_*_binary.bin /out/imported_mapper.bin
cmp -s /out/native_mapper.bin /out/imported_mapper.bin && echo identical > /out/bitstream_comparison.txt || echo different > /out/bitstream_comparison.txt

IMPORT_SIM=/tmp/flowadvantage_array_add_sim
mkdir -p "$IMPORT_SIM"
cp "$MAP"/*_II="${II}"_*_binary.bin "$IMPORT_SIM"/
TRACE=$(find /home/user/morpher/Morpher_DFG_Generator/benchmarks/morpher_benchmarks/array_add/memtraces -name "array_add_trace_*.txt" -print -quit)
test -n "$TRACE"
/home/user/morpher/hycube_simulator/src/build/hycube_simulator -x 4 -y 4 -c "$IMPORT_SIM"/*.bin -d "$TRACE" -a "$SIM/array_add_mem_alloc.txt" -m 4096 || SIM_STATUS=$?
# The simulator writes its result in the mapper working directory rather
# than beside the binary supplied through -c.  Preserve the authoritative
# result so the round-trip record is self-contained.
cp "$MAP/sim_result.txt" /out/sim_result.txt 2>/dev/null || true
cp "$MAP"/*.bin /out/
exit "${SIM_STATUS:-0}"
' 
