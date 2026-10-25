#!/usr/bin/env bash
# Copyright (c) 2026 Lee Gao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generates the Mali pass benchmark tables documented in mali-passes.md.
#
# Usage:
#   ./bench_mali_passes.sh [options] [shader.spv ...]
#
# Options:
#   -n ITERS   iterations per shader for section 2 (default: 10)
#   -N ITERS   iterations per shader for section 1 (default: 100)
#   -o PATH    spirv-opt binary to benchmark (default: ./build/tools/spirv-opt)
#   -b PATH    pre-hardening spirv-opt binary; enables section 1
#   -w FILE    write the report to FILE instead of stdout
#   -h         show this help
#
# Requires GNU date (for %N) and awk.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

OPT="./build/tools/spirv-opt"
BASELINE_OPT=""
INC_ITERS=10
HARD_ITERS=100
OUT=""

DEFAULT_SHADERS=(
  optbarrier.spv
  opselect.spv
  gta_frag.spv
  gta_vertex.spv
  failing_shader.spv
  clip_dist.spv
)

# Passes are added in this order for the cumulative benchmark.
PASS_ORDER=(
  "--fix-mali-spec-constant-composite"
  "--mali-optimization-barrier"
  "--mali-bitwise-scalar"
  "--mali-clamp-zero"
)
CONFIG_LABELS=("None" "+FixSpec" "+OptBarrier" "+BitwiseScalar" "+ClampZero")

usage() {
  sed -n '23,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while getopts ":n:N:o:b:w:h" opt; do
  case "$opt" in
    n) INC_ITERS=$OPTARG ;;
    N) HARD_ITERS=$OPTARG ;;
    o) OPT=$OPTARG ;;
    b) BASELINE_OPT=$OPTARG ;;
    w) OUT=$OPTARG ;;
    h) usage; exit 0 ;;
    :) die "option -$OPTARG requires an argument" ;;
    \?) die "unknown option -$OPTARG" ;;
  esac
done
shift $((OPTIND - 1))

if [ "$#" -gt 0 ]; then
  SHADERS=("$@")
else
  SHADERS=("${DEFAULT_SHADERS[@]}")
fi

[ -x "$OPT" ] || die "'$OPT' is not executable (build spirv-opt first)"
if [ -n "$BASELINE_OPT" ] && [ ! -x "$BASELINE_OPT" ]; then
  die "'$BASELINE_OPT' is not executable"
fi
case "$INC_ITERS$HARD_ITERS" in
  *[!0-9]*) die "iteration counts must be positive integers" ;;
esac
[ "$INC_ITERS" -gt 0 ] || die "iteration count must be positive"
[ "$HARD_ITERS" -gt 0 ] || die "iteration count must be positive"

now() { date +%s.%N; }

# run_bench ITERS OPT SHADER [PASS_FLAGS...]
# Prints elapsed seconds for ITERS invocations, after one warm-up run.
run_bench() {
  local iters=$1 opt=$2 shader=$3
  shift 3
  local start end i

  "$opt" "$@" "$shader" -o /dev/null >/dev/null 2>&1
  start=$(now)
  for ((i = 0; i < iters; i++)); do
    "$opt" "$@" "$shader" -o /dev/null >/dev/null 2>&1
  done
  end=$(now)

  awk -v s="$start" -v e="$end" 'BEGIN { printf "%.2f", e - s }'
}

# Collect the shaders that actually exist.
PRESENT_SHADERS=()
for shader in "${SHADERS[@]}"; do
  if [ -f "$shader" ]; then
    PRESENT_SHADERS+=("$shader")
  else
    printf 'warning: skipping missing shader %s\n' "$shader" >&2
  fi
done
[ "${#PRESENT_SHADERS[@]}" -gt 0 ] || die "no shaders found"

emit_section2() {
  printf 'Measurements taken on host x86_64 across the same shader corpus (**%s iterations per shader**). Passes are enabled cumulatively in the order shown, so each column adds exactly one pass to the previous column:\n\n' "$INC_ITERS"
  printf '%s\n' '- `FixSpec` = `--fix-mali-spec-constant-composite`'
  printf '%s\n' '- `OptBarrier` = `--mali-optimization-barrier`'
  printf '%s\n' '- `BitwiseScalar` = `--mali-bitwise-scalar`'
  printf '%s\n' '- `ClampZero` = `--mali-clamp-zero`'
  printf '\n'

  printf '| Corpus Shader |'
  for label in "${CONFIG_LABELS[@]}"; do
    printf ' %s |' "$label"
  done
  printf '\n%s' '|---|'
  for _ in "${CONFIG_LABELS[@]}"; do
    printf '%s' '---|'
  done
  printf '\n'

  for shader in "${PRESENT_SHADERS[@]}"; do
    printf '| `%s` |' "$(basename "$shader")"
    for ((c = 0; c <= ${#PASS_ORDER[@]}; c++)); do
      local elapsed
      elapsed=$(run_bench "$INC_ITERS" "$OPT" "$shader" "${PASS_ORDER[@]:0:c}")
      printf ' %ss |' "$elapsed"
    done
    printf '\n'
  done
}

emit_report() {
  printf '# Mali SPIR-V Pass Performance Benchmarks\n\n'
  emit_section2
}

if [ -n "$OUT" ]; then
  emit_report > "$OUT"
  printf 'wrote %s\n' "$OUT" >&2
else
  emit_report
fi
