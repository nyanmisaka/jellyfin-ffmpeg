#!/bin/bash

SCRIPT_REPO="https://github.com/nyanmisaka/rk-mirrors.git"
SCRIPT_COMMIT="1d330cc28551943bed3380261a5a9c6fbd58ff53"
SCRIPT_BRANCH="jellyfin-rga-next"

ffbuild_enabled() {
    [[ $TARGET == linux* ]] && [[ $TARGET == *arm64 ]] && return 0
    return -1
}

ffbuild_dockerbuild() {
    git-mini-clone "$SCRIPT_REPO" "$SCRIPT_COMMIT" rkrga
    cd rkrga

    sed -i 's/shared_library/library/g' meson.build

    mkdir rkrga_build && cd rkrga_build

    local myconf=(
        --prefix="$FFBUILD_PREFIX"
        --buildtype=release
        --default-library=static
        -Dcpp_args=-fpermissive
        -Dlibdrm=false
        -Dlibrga_demo=false
    )

    if [[ $TARGET == linux* ]]; then
        myconf+=(
            --cross-file=/cross.meson
        )
    else
        echo "Unknown target"
        return -1
    fi

    meson "${myconf[@]}" ..

    ninja -j$(nproc)
    ninja install

    echo "Libs.private: -lstdc++" >> "$FFBUILD_PREFIX"/lib/pkgconfig/librga.pc
}

ffbuild_configure() {
    echo --enable-rkrga
}

ffbuild_unconfigure() {
    [[ $TARGET != linux* ]] && return 0
    [[ $TARGET != *arm64 ]] && return 0
    echo --disable-rkrga
}
