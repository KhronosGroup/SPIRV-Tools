#!/bin/bash

# Copyright (c) 2020 The Khronos Group Inc.
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

set -e
set -x

NUM_CORES=$(nproc)
echo "Detected $NUM_CORES cores for building"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
VERSION=$(sed -n '0,/^v20/ s/^v\(20[0-9.]*\).*/\1/p' $DIR/../../CHANGES).${GITHUB_RUN_NUMBER:-0}
echo "Version: $VERSION"

make_package() {
  type=$1
  pkg=$2

  mkdir -p out/$type/$pkg

  # copy other js files
  cp source/wasm/$pkg.d.ts out/$type/$pkg
  sed -e 's/\("version"\s*:\s*\).*/\1"'$VERSION'",/' source/wasm/package-$pkg.json > out/$type/$pkg/package.json

  cp source/wasm/README.md out/$type/$pkg
  cp LICENSE out/$type/$pkg

  cp build/$type/$pkg.js out/$type/$pkg

  gzip -9 -k -f out/$type/$pkg/$pkg.js
  if [ -e build/$type/$pkg.wasm ] ; then
     cp build/$type/$pkg.wasm out/$type/$pkg
     gzip -9 -k -f out/$type/$pkg/$pkg.wasm
  fi
  wc -c out/$type/$pkg/*
}


build() {
    type=$1
    shift
    args=$@
    mkdir -p build/$type

    pushd build/$type
    echo $args
    emcmake cmake \
        -DCMAKE_BUILD_TYPE=Release \
        $args \
        ../..

    emmake make -j $(( $NUM_CORES )) SPIRV-Tools-static
    emmake make -j $(( $NUM_CORES )) SPIRV-Tools-opt

    echo Building js interface
    emcc \
        --bind \
        -I../../include \
        -std=c++11 \
        ../../source/wasm/spirv-tools.cpp \
        source/libSPIRV-Tools.a \
        -o spirv-tools.js \
        -s MODULARIZE \
        -Oz

    emcc \
        --bind \
        -I../../include \
        -std=c++11 \
        ../../source/wasm/spirv-tools-opt.cpp \
        source/libSPIRV-Tools.a \
        source/opt/libSPIRV-Tools-opt.a \
        -o spirv-tools-opt.js \
        -s MODULARIZE \
        -Oz

    popd

    make_package $type spirv-tools
    make_package $type spirv-tools-opt
}

if [ ! -d external/spirv-headers ] ; then
    echo "Fetching SPIRV-headers"
    git clone https://github.com/KhronosGroup/SPIRV-Headers.git external/spirv-headers
fi

echo Building ${BASH_REMATCH[1]}
build web\
    -DSPIRV_COLOR_TERMINAL=OFF\
    -DSPIRV_SKIP_TESTS=ON\
    -DSPIRV_SKIP_EXECUTABLES=ON

