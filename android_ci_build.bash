#!/bin/bash
# set -e

python3 utils/git-sync-deps

cmake_build () {
  ANDROID_ABI=$1
  mkdir -p build
  cd build
  cmake $GITHUB_WORKSPACE/src/main/cpp -DCMAKE_BUILD_TYPE=Release -DANDROID_PLATFORM=24 -DANDROID_ABI=$ANDROID_ABI -DCMAKE_SYSTEM_NAME=Android -DANDROID_TOOLCHAIN=clang -DANDROID_ARM_MODE=arm -DCMAKE_MAKE_PROGRAM=$ANDROID_NDK_LATEST_HOME/prebuilt/linux-x86_64/bin/make -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_LATEST_HOME/build/cmake/android.toolchain.cmake
  cmake --build . --config Release --parallel 4
  cmake --install . --prefix .
  # $ANDROID_NDK_LATEST_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip $GITHUB_WORKSPACE/build/libmobileglues.so
}

cmake_build arm64-v8a
