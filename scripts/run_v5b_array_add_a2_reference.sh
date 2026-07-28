#!/usr/bin/env bash
set -euo pipefail

# Produce the frozen A2 memory-organization reference from Morpher's own
# compiler output.  The memory architecture is generated with the exact data
# allocation emitted for this DFG; no hand-written allocation is accepted.
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/results/revision_v5b/reference_mappings/array_add/A2_hycube4x4_mem_variant"
mkdir -p "$OUT_DIR"

docker run --rm -v "$OUT_DIR:/out" quotientflow-morpher-v5b-native bash -lc '
set -euo pipefail
cd /home/user/morpher
bash build_all.sh
python -u run_morpher.py morpher_benchmarks/array_add/array_add.c array_add

MAP=/home/user/morpher/Morpher_CGRA_Mapper/benchmarks/hycube/morpher_benchmarks/array_add
DFG=array_add_PartPredDFG.xml
cd "$MAP"
python /home/user/morpher/Morpher_CGRA_Mapper/update_mem_alloc.py \
  /home/user/morpher/Morpher_CGRA_Mapper/json_arch/hycube_original_updatemem.json \
  array_add_mem_alloc.txt 2048 2 a2_hycube_mem.json
/home/user/morpher/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper \
  -d "$DFG" -x 4 -y 4 -j a2_hycube_mem.json -i 4 -t HyCUBE_4REG -m 0 \
  --dump-flowadvantage-state /out
cp "$DFG" /out/array_add_PartPredDFG.xml
cp a2_hycube_mem.json /out/a2_hycube_mem.json
'
