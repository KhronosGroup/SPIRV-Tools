cmake -B build-x86-64 -G Ninja \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPIRV_SKIP_TESTS=ON \
  -DCMAKE_INSTALL_PREFIX=./install-x86-64

cmake --build build-x86-64 -j 4

cmake --install build-x86-64
