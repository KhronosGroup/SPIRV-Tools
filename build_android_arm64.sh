export ANDROID_NDK=$HOME/Android/Sdk/ndk/27.0.12077973

cmake -B build-android-arm64 -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  -DCMAKE_CXX_FLAGS="-ffunction-sections -fdata-sections -g0" \
  -DSPIRV_CHECK_CONTEXT=OFF \
  -DCMAKE_INSTALL_PREFIX=./install-android-arm64

cmake --build build-android-arm64 -j 4

cmake --install build-android-arm64
