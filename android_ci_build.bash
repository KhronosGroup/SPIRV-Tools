#!/bin/bash
set -e

python3 utils/git-sync-deps

export TOOLCHAIN=$ANDROID_NDK_LATEST_HOME/toolchains/llvm/prebuilt/linux-x86_64

cmake_build () {
  ANDROID_ABI=$1
  ANDROID_TARGET=$2
  mkdir -p build
  cd build
  cmake $GITHUB_WORKSPACE -DCMAKE_BUILD_TYPE=Release  -DANDROID_PLATFORM=26 -DPATHFINDER_TARGET=ANDROID_TARGET -DCMAKE_AR=$ANDROID_NDK_LATEST_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar -DCMAKE_RANLIB=$ANDROID_NDK_LATEST_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ranlib -DANDROID_ABI=$ANDROID_ABI -DCMAKE_SYSTEM_NAME=Android -DANDROID_TOOLCHAIN=clang -DANDROID_ARM_MODE=arm -DCMAKE_MAKE_PROGRAM=$ANDROID_NDK_LATEST_HOME/prebuilt/linux-x86_64/bin/make -DCMAKE_SYSTEM_NAME=Android -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_LATEST_HOME/build/cmake/android.toolchain.cmake
  cmake --build . --config Release --parallel 4
  cmake --install . --prefix .
  find ./ -name '*' -execdir ${TOOLCHAIN}/bin/llvm-strip --strip-debug {} \;
}

cmake_build arm64-v8a aarch64-linux-android
