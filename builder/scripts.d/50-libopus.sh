#!/bin/bash

SCRIPT_REPO="https://github.com/xiph/opus.git"
SCRIPT_COMMIT="b5dc74f26ff9a9ef1f1af9a473b9ae07520eed7d"

ffbuild_enabled() {
    return 0
}

ffbuild_dockerbuild() {
    git-mini-clone "$SCRIPT_REPO" "$SCRIPT_COMMIT" opus
    cd opus

    # Fix AVX2 auto detction
    wget -q -O - https://github.com/xiph/opus/commit/9ec11c1.patch | git apply

    ./autogen.sh

    local myconf=(
        --prefix="$FFBUILD_PREFIX"
        --disable-shared
        --enable-static
        --disable-extra-programs
    )

    if [[ $TARGET == win* || $TARGET == linux* ]]; then
        myconf+=(
            --host="$FFBUILD_TOOLCHAIN"
        )
    elif [[ $TARGET == mac* ]]; then
        :
    else
        echo "Unknown target"
        return -1
    fi

    if [[ $TARGET == linux* || $TARGET == mac* ]] && [[ $TARGET == *arm64 ]]; then
        myconf+=(
            --with-NE10-libraries="$FFBUILD_PREFIX"/lib
            --with-NE10-includes="$FFBUILD_PREFIX"/include/libNE10
        )
    fi

    # Override previously set -O(n) option and the CC's default optimization options.
    CFLAGS="$CFLAGS -O3" ./configure "${myconf[@]}"
    make -j$(nproc)
    make install

    if [[ $TARGET == *arm64 ]]; then
        if [[ $TARGET == mac* ]]; then
            gsed -i 's/-lopus/-lopus -lNE10/' "$FFBUILD_PREFIX"/lib/pkgconfig/opus.pc
            gsed -i 's/-I${includedir}\/opus/-I${includedir}\/opus -I${includedir}\/libNE10/' "$FFBUILD_PREFIX"/lib/pkgconfig/opus.pc
        else
            sed -i 's/-lopus/-lopus -lNE10/' "$FFBUILD_PREFIX"/lib/pkgconfig/opus.pc
            sed -i 's/-I${includedir}\/opus/-I${includedir}\/opus -I${includedir}\/libNE10/' "$FFBUILD_PREFIX"/lib/pkgconfig/opus.pc
        fi
    fi
}

ffbuild_configure() {
    echo --enable-libopus
}

ffbuild_unconfigure() {
    echo --disable-libopus
}
