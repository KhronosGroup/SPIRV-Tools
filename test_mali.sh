#!/bin/sh
# Host end-to-end checks for the Mali SPIR-V workaround passes.
#
# Builds the host tools, runs the same workaround pipeline the compatibility
# layer uses, then disassembles and validates the result.
#
# Usage:
#   sh test_mali.sh [module.spv ...]
#
# Environment:
#   SKIP_BUILD=1   use the existing install-x86-64 tools instead of rebuilding
#
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

if [ "${SKIP_BUILD:-0}" != "1" ]; then
time sh build_host.sh
fi

BIN="$ROOT/install-x86-64/bin"
OPT="$BIN/spirv-opt"
DIS="$BIN/spirv-dis"
VAL="$BIN/spirv-val"

for tool in "$OPT" "$DIS" "$VAL"; do
    if [ ! -x "$tool" ]; then
        echo "error: missing $tool (run without SKIP_BUILD=1)" >&2
        exit 1
    fi
done

DEFAULT_MODULES="
gta_frag.spv
gta_vertex.spv
gtav.spv
single_vs.spv
shader.spv
opselect.spv
optbarrier.spv
clip_dist.spv
failing_shader.spv
"

if [ "$#" -eq 0 ]; then
    # Intentional word splitting: these are simple space-separated file names.
    # shellcheck disable=SC2086
    set -- $DEFAULT_MODULES
fi

# Keep this in the same order as the compatibility layer's
# RunMaliSpirvWorkarounds(): interface changes first, then instruction-level
# rewrites, with clip/cull lowering last.
run_opt() {
    "$OPT" \
        --fix-mali-spec-constant-composite \
        --mali-pervertex-block \
        --mali-optimization-barrier \
        --mali-bitwise-scalar \
        --mali-clamp-zero \
        --mali-lockstep \
        --remove-clip-cull-dist \
        "$1" -o "$2"
}

failures=0

for module in "$@"; do
    if [ ! -f "$module" ]; then
        echo "error: no such module: $module" >&2
        failures=$((failures + 1))
        continue
    fi

    base=${module%.*}
    optimized="${base}_opt.spv"

    echo "== $module"

    if ! run_opt "$module" "$optimized"; then
        echo "error: spirv-opt failed for $module" >&2
        failures=$((failures + 1))
        continue
    fi
    if ! "$DIS" "$module" > "${module}.txt"; then
        echo "error: spirv-dis failed for $module" >&2
        failures=$((failures + 1))
        continue
    fi
    if ! "$DIS" "$optimized" > "${optimized}.txt"; then
        echo "error: spirv-dis failed for $optimized" >&2
        failures=$((failures + 1))
        continue
    fi
    if ! "$VAL" "$optimized"; then
        echo "error: spirv-val failed for $optimized" >&2
        failures=$((failures + 1))
        continue
    fi

    echo "ok: $module -> $optimized"
    echo
done

if [ "$failures" -ne 0 ]; then
    echo "$failures Mali end-to-end check(s) failed" >&2
    exit 1
fi

echo "all Mali end-to-end checks passed"
