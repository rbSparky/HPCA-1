#!/usr/bin/env bash
# Install the exact GCC-7 Morpher fixed14 binary extracted from the validated
# Docker image, together with its matching source, in an immutable prefix.
set -euo pipefail

archive="${1:?usage: install_cluster_morpher_binary_v5b.sh SOURCE_ARCHIVE SOURCE_SHA TREE_SHA BINARY BINARY_SHA}"
archive_sha="${2:?missing SOURCE_SHA}"
tree_sha="${3:?missing TREE_SHA}"
binary_input="${4:?missing BINARY}"
binary_sha="${5:?missing BINARY_SHA}"
cluster_root="${CLUSTER_TOOLCHAIN_ROOT:-$HOME/remote-work/HPCA/.cluster_toolchains}"
name="morpher-fixed14-gcc7-${binary_sha:0:12}"
final_dir="${cluster_root}/${name}"

[[ "$(sha256sum "$archive" | awk '{print $1}')" == "$archive_sha" ]]
[[ "$(sha256sum "$binary_input" | awk '{print $1}')" == "$binary_sha" ]]

if [[ -f "$final_dir/READY" ]]; then
  grep -Fx "BINARY_SHA256=$binary_sha" "$final_dir/manifest.env" >/dev/null
  printf 'READY_EXISTING=%s\n' "$final_dir"
  exit 0
fi
if [[ -e "$final_dir" ]]; then
  echo "refusing to overwrite incomplete toolchain: $final_dir" >&2
  exit 31
fi

mkdir -p "$cluster_root"
tmp_dir="${cluster_root}/.${name}.installing.$$.${RANDOM}"
mkdir -p "$tmp_dir/source" "$tmp_dir/bin"
trap 'printf "INCOMPLETE_INSTALL_DIR=%s\n" "$tmp_dir" >&2' ERR

tar -xzf "$archive" -C "$tmp_dir/source"
install -m 0555 "$binary_input" "$tmp_dir/bin/cgra_xml_mapper"
"$tmp_dir/bin/cgra_xml_mapper" --help > "$tmp_dir/help.txt"

{
  printf 'NAME=%q\n' "$name"
  printf 'MORPHER_BASE_COMMIT=%q\n' "9a9dce7aea521f1d5ef33686f57ca84864edb3c9"
  printf 'BRIDGE_PATCH_SHA256=%q\n' "4894054e67eb43f16ad0a40bcbf9783305b67612925a3ea5b0d7acdfd11fad54"
  printf 'SOURCE_TREE_SHA256=%q\n' "$tree_sha"
  printf 'SOURCE_ARCHIVE_SHA256=%q\n' "$archive_sha"
  printf 'BINARY_SHA256=%q\n' "$binary_sha"
  printf 'SOURCE_DOCKER_IMAGE=%q\n' \
    "quotientflow-morpher-v5b-native-fixed14@sha256:f1dee9de88d0935ccb1825200040e090abf78ff76140bb0e8d494c00862c9088"
  printf 'HOST=%q\n' "$(hostname)"
  printf 'INSTALLED_AT=%q\n' "$(date -Iseconds)"
} > "$tmp_dir/manifest.env"

printf 'ready\n' > "$tmp_dir/READY"
mv "$tmp_dir" "$final_dir"
trap - ERR
printf 'READY_NEW=%s\n' "$final_dir"
