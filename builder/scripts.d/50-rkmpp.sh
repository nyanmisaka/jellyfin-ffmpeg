#!/bin/bash

SCRIPT_REPO="https://github.com/nyanmisaka/rk-mirrors.git"
SCRIPT_COMMIT="a9380ef333102ac318628f83b5f7a460d377749e"
SCRIPT_BRANCH="jellyfin-mpp-next"

ffbuild_enabled() {
    [[ $TARGET == linux* ]] && [[ $TARGET == *arm64 ]] && return 0
    return -1
}

ffbuild_dockerbuild() {
    git-mini-clone "$SCRIPT_REPO" "$SCRIPT_COMMIT" rkmpp
    cd rkmpp

    mkdir rkmpp_build && cd rkmpp_build

    cmake -GNinja \
        -DCMAKE_TOOLCHAIN_FILE="$FFBUILD_CMAKE_TOOLCHAIN" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$FFBUILD_PREFIX" \
        -DBUILD_TEST=OFF \
        -DBUILD_SHARED_LIBS=OFF ..

    ninja -j$(nproc)
    ninja install

    echo "Libs.private: -lstdc++" >> "$FFBUILD_PREFIX"/lib/pkgconfig/rockchip_mpp.pc
}

ffbuild_configure() {
    echo --enable-rkmpp
}

ffbuild_unconfigure() {
    [[ $TARGET != linux* ]] && return 0
    [[ $TARGET != *arm64 ]] && return 0
    echo --disable-rkmpp
}
