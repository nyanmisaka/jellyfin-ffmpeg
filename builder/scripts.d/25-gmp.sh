#!/bin/bash

SCRIPT_REPO="https://github.com/BtbN/gmplib.git"
SCRIPT_COMMIT="ece1292e8282ef2c9ca204d1b510ff2aca04a7a3"

ffbuild_enabled() {
    return 0
}

ffbuild_dockerbuild() {
    git-mini-clone "$SCRIPT_REPO" "$SCRIPT_COMMIT" gmp
    cd gmp

    ./.bootstrap

    local myconf=(
        --prefix="$FFBUILD_PREFIX"
        --enable-maintainer-mode
        --disable-shared
        --enable-static
        --with-pic
    )

    if [[ $TARGET == win* || $TARGET == linux* ]]; then
        myconf+=(
            --host="$FFBUILD_TOOLCHAIN"
        )
    elif [[ $TARGET == mac* ]]; then
        myconf+=(
            --enable-cxx
        )
        if [ "$MACOS_BUILDER_CPU_ARCH" = "arm64" ] && [ "$TARGET" = "mac64" ]; then
            myconf+=(
                --host="x86_64-apple-darwin" # Override GMP's autodetect
            )
        fi
        # The shipped configure script relies on an outdated libtool version
        # which causes linker errors due to name collisions on macOS
        # leads to wrongly compiled libraries
        # regenerate the configure ourselves as a workaroud
        autoreconf -i -s
    else
        echo "Unknown target"
        return -1
    fi

    ./configure "${myconf[@]}"
    make -j$(nproc)
    make install
}

ffbuild_configure() {
    echo --enable-gmp
}

ffbuild_unconfigure() {
    echo --disable-gmp
}
