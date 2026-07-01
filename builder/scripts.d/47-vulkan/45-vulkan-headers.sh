#!/bin/bash

SCRIPT_REPO="https://github.com/KhronosGroup/Vulkan-Headers.git"
SCRIPT_COMMIT="v1.4.355"
SCRIPT_TAGFILTER="v?.*.*"

ffbuild_enabled() {
    [[ $TARGET == mac* ]] && return -1
    return 0
}

ffbuild_dockerbuild() {
    git-mini-clone "$SCRIPT_REPO" "$SCRIPT_COMMIT" vkheaders
    cd vkheaders

    mkdir build && cd build

    cmake -DCMAKE_TOOLCHAIN_FILE="$FFBUILD_CMAKE_TOOLCHAIN" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$FFBUILD_PREFIX" \
        -DVULKAN_HEADERS_ENABLE_MODULE=NO -DVULKAN_HEADERS_ENABLE_TESTS=NO -DVULKAN_HEADERS_ENABLE_INSTALL=YES ..
    make -j$(nproc)
    make install
}
