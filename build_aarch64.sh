cmake -B build-aarch64 -G Ninja \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPIRV_SKIP_TESTS=ON \
  -DCMAKE_INSTALL_PREFIX=./install-aarch64

cmake --build build-aarch64 -j 4

cmake --install build-aarch64
