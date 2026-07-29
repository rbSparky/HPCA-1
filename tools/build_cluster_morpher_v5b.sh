#!/usr/bin/env bash
# Build the pinned Morpher fixed14 bridge in an immutable, user-owned prefix.
#
# This script is intended to run through tools/cluster_queue.sh.  It does not
# require root, mutate an existing environment, or overwrite an existing
# toolchain.  The final directory name includes the source content hash.
set -euo pipefail

archive="${1:?usage: build_cluster_morpher_v5b.sh SOURCE_ARCHIVE SOURCE_SHA TREE_SHA}"
archive_sha="${2:?missing SOURCE_SHA}"
tree_sha="${3:?missing TREE_SHA}"
cluster_root="${CLUSTER_TOOLCHAIN_ROOT:-$HOME/remote-work/HPCA/.cluster_toolchains}"
name="morpher-fixed14-${tree_sha:0:12}"
final_dir="${cluster_root}/${name}"

actual_archive_sha="$(sha256sum "$archive" | awk '{print $1}')"
if [[ "$actual_archive_sha" != "$archive_sha" ]]; then
  echo "archive checksum mismatch: expected=$archive_sha actual=$actual_archive_sha" >&2
  exit 11
fi

if [[ -f "$final_dir/READY" ]]; then
  grep -Fx "TREE_SHA256=$tree_sha" "$final_dir/manifest.env" >/dev/null
  "$final_dir/source/Morpher_CGRA_Mapper/build/src/cgra_xml_mapper" --help \
    >/dev/null 2>&1 || true
  printf 'READY_EXISTING=%s\n' "$final_dir"
  exit 0
fi
if [[ -e "$final_dir" ]]; then
  echo "refusing to overwrite incomplete toolchain: $final_dir" >&2
  exit 12
fi

mkdir -p "$cluster_root"
tmp_dir="${cluster_root}/.${name}.building.$$.${RANDOM}"
mkdir "$tmp_dir"
trap 'printf "INCOMPLETE_BUILD_DIR=%s\n" "$tmp_dir" >&2' ERR

/usr/bin/python3 -m venv "$tmp_dir/cmake-env"
"$tmp_dir/cmake-env/bin/python" -m pip install \
  --disable-pip-version-check --no-input "cmake==3.31.6"

mkdir "$tmp_dir/source"
tar -xzf "$archive" -C "$tmp_dir/source"
source_dir="$tmp_dir/source/Morpher_CGRA_Mapper"

# The recorded upstream CMake file assigns Ubuntu-18.04-specific compiler
# paths *after* adding the source subdirectory.  The cluster is Ubuntu 24.04
# and intentionally has no system-wide gcc-7 package.  Change only those two
# tool-path assignments in this private build tree; the Morpher sources and
# mapping semantics remain byte-identical.
sed -i \
  -e 's|set(CMAKE_C_COMPILER "gcc-7")|set(CMAKE_C_COMPILER "/usr/bin/gcc")|' \
  -e 's|set(CMAKE_CXX_COMPILER "/usr/bin/g++-7")|set(CMAKE_CXX_COMPILER "/usr/bin/g++")|' \
  "$source_dir/CMakeLists.txt" "$source_dir/src/CMakeLists.txt"

(
  cd "$source_dir"
  mkdir build
  cd build
  "$tmp_dir/cmake-env/bin/cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    ..
  make -j2
) 2>&1 | tee "$tmp_dir/build.log"

binary="$source_dir/build/src/cgra_xml_mapper"
test -x "$binary"

{
  printf 'NAME=%q\n' "$name"
  printf 'MORPHER_BASE_COMMIT=%q\n' "9a9dce7aea521f1d5ef33686f57ca84864edb3c9"
  printf 'BRIDGE_PATCH_SHA256=%q\n' "4894054e67eb43f16ad0a40bcbf9783305b67612925a3ea5b0d7acdfd11fad54"
  printf 'TREE_SHA256=%q\n' "$tree_sha"
  printf 'ARCHIVE_SHA256=%q\n' "$archive_sha"
  printf 'CMAKE_VERSION=%q\n' "$("$tmp_dir/cmake-env/bin/cmake" --version | head -1)"
  printf 'CXX_VERSION=%q\n' "$(/usr/bin/g++ --version | head -1)"
  printf 'BUILD_COMPATIBILITY_CHANGE=%q\n' \
    "CMake compiler paths gcc-7/g++-7 replaced by /usr/bin/gcc and /usr/bin/g++"
  printf 'HOST=%q\n' "$(hostname)"
  printf 'BUILT_AT=%q\n' "$(date -Iseconds)"
  printf 'BINARY_SHA256=%q\n' "$(sha256sum "$binary" | awk '{print $1}')"
} > "$tmp_dir/manifest.env"

printf 'ready\n' > "$tmp_dir/READY"
mv "$tmp_dir" "$final_dir"
trap - ERR
printf 'READY_NEW=%s\n' "$final_dir"
