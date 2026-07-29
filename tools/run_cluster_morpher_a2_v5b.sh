#!/usr/bin/env bash
# Derive a frozen A2 memory architecture from Morpher's checked-in bank layout,
# then execute the atomic native export runner.
set -euo pipefail

toolchain="${1:?usage: run_cluster_morpher_a2_v5b.sh TOOLCHAIN KERNEL OUTPUT_DIR}"
kernel="${2:?missing kernel}"
output_dir="${3:?missing OUTPUT_DIR}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="$toolchain/source/Morpher_CGRA_Mapper"

case "$kernel" in
  fix_fft)
    source_arch="$source_root/json_arch/fft_various_mem_archs/stdnoc_mem_dual_port_two_banked.json"
    # This is Morpher's checked-in, natively mapped FFT DFG.  Unlike the
    # legacy fix_fft_npb artifact, it carries every BasePointerName required
    # to apply the checked-in memory layout.
    dfg_rel="applications/sample_xmls/fix_fft_INNERMOST_LN121_DFG.xml"
    required=(fr manupa1 l.1253 conv31252 fi shl29 conv.i conv.i241)
    ;;
  gemm_nt)
    source_arch="$source_root/json_arch/gemm_various_mem_archs/stdnoc_mem_dual_port_two_banked.json"
    dfg_rel="applications/gemm_nt/gemm_nt_INNERMOST_LN111_PartPred_DFG.xml"
    required=(A B ALPHA shr mul mul12 add49)
    ;;
  *)
    echo "unsupported A2 kernel: $kernel" >&2
    exit 51
    ;;
esac

test -f "$source_arch"
if [[ -e "$output_dir" || -e "${output_dir}.inputs" ]]; then
  echo "refusing to overwrite A2 output or input provenance: $output_dir" >&2
  exit 52
fi

input_tmp="${output_dir}.inputs.tmp.$$"
input_dir="${output_dir}.inputs"
mkdir -p "$input_tmp"
trap 'printf "INCOMPLETE_A2_INPUT_DIR=%s\n" "$input_tmp" >&2' ERR

derive_args=(
  "$repo_root/scripts/derive_morpher_mem_alloc.py"
  --source-architecture "$source_arch"
  --bank-size 2048
  --output-csv "$input_tmp/mem_alloc.csv"
  --provenance-json "$input_tmp/memory_layout_provenance.json"
)
for variable in "${required[@]}"; do
  derive_args+=(--required-variable "$variable")
done
python3 "${derive_args[@]}"

python3 "$source_root/update_mem_alloc.py" \
  "$source_root/json_arch/hycube_original_updatemem.json" \
  "$input_tmp/mem_alloc.csv" 2048 2 \
  "$input_tmp/architecture_exact_layout.json"

{
  printf 'KERNEL=%q\n' "$kernel"
  printf 'SOURCE_ARCH=%q\n' "$source_arch"
  printf 'SOURCE_ARCH_SHA256=%q\n' "$(sha256sum "$source_arch" | awk '{print $1}')"
  printf 'MEM_ALLOC_SHA256=%q\n' "$(sha256sum "$input_tmp/mem_alloc.csv" | awk '{print $1}')"
  printf 'PROVENANCE_SHA256=%q\n' "$(sha256sum "$input_tmp/memory_layout_provenance.json" | awk '{print $1}')"
  printf 'GENERATED_ARCH_SHA256=%q\n' "$(sha256sum "$input_tmp/architecture_exact_layout.json" | awk '{print $1}')"
  printf 'BANK_SIZE=2048\nBANK_COUNT=2\n'
} > "$input_tmp/manifest.env"
mv "$input_tmp" "$input_dir"
trap - ERR

FLOWADVANTAGE_AUX_INPUT_DIR="$input_dir" \
  exec bash "$repo_root/tools/run_cluster_morpher_export_v5b.sh" \
  "$toolchain" "$dfg_rel" "$input_dir/architecture_exact_layout.json" \
  4 4 4 HyCUBE_4REG 0 "$output_dir"
