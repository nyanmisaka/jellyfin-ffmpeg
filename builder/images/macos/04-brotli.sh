#!/bin/bash
ffbuild_macbase() {
  wget https://github.com/google/brotli/archive/refs/tags/v1.2.0.tar.gz -O broti.tar.gz
  tar xvf broti.tar.gz
  cd brotli-1.2.0

  mkdir build
  cd build

  cmake ../ -DCMAKE_INSTALL_PREFIX="$FFBUILD_PREFIX" \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_SHARED_LIBS=OFF
  make -j$(nproc)
  make install
}
