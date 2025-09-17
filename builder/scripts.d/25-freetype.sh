#!/bin/bash

SCRIPT_REPO="https://github.com/freetype/freetype.git"
SCRIPT_COMMIT="03acd8923bd8a45ce8edaf06f5194b48becbad7f"

ffbuild_enabled() {
    return 0
}

ffbuild_dockerbuild() {
    git-mini-clone "$SCRIPT_REPO" "$SCRIPT_COMMIT" freetype
    cd freetype

    ./autogen.sh

    local myconf=(
        --prefix="$FFBUILD_PREFIX"
        --disable-shared
        --enable-static
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

    ./configure "${myconf[@]}"
    make -j$(nproc)
    make install

    # freetype2 does not link to macOS built-in iconv in its pkgconfig, add it
    if [[ $TARGET == mac* ]]; then
        sed -i '' '/^Libs:/ s/$/ -liconv/' "$FFBUILD_PREFIX"/lib/pkgconfig/freetype2.pc
    fi
}

ffbuild_configure() {
    echo --enable-libfreetype
}

ffbuild_unconfigure() {
    echo --disable-libfreetype
}
