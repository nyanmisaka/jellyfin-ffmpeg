#!/bin/bash

SCRIPT_REPO="https://code.videolan.org/videolan/dav1d.git"
SCRIPT_COMMIT="a31e4bd75787c451301d66e888c76ada54544a0c"

ffbuild_enabled() {
    return 0
}

ffbuild_dockerbuild() {
    git-mini-clone "$SCRIPT_REPO" "$SCRIPT_COMMIT" dav1d
    cd dav1d

    mkdir build && cd build

    local myconf=(
        --prefix="$FFBUILD_PREFIX"
        --buildtype=release
        --default-library=static
    )

    if [[ $TARGET == win* || $TARGET == linux* ]]; then
        myconf+=(
            --cross-file=/cross.meson
        )
    elif [[ $TARGET == mac* ]]; then
        if [ "$MACOS_BUILDER_CPU_ARCH" = "arm64" ] && [ "$TARGET" = "mac64" ]; then
            myconf+=(
                --cross-file="$BUILDER_ROOT"/images/macos/cross/cross-x86_64.txt
            )
        fi
    else
        echo "Unknown target"
        return -1
    fi

    meson "${myconf[@]}" ..
    ninja -j$(nproc)
    ninja install
}

ffbuild_configure() {
    echo --enable-libdav1d
}

ffbuild_unconfigure() {
    echo --disable-libdav1d
}
