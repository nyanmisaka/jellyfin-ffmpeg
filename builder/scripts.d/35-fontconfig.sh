#!/bin/bash

SCRIPT_REPO="https://chromium.googlesource.com/external/fontconfig"
SCRIPT_COMMIT="6d0a98982ec351c165c9224c8b7dbdfca3010e47"

ffbuild_enabled() {
    return 0
}

ffbuild_dockerbuild() {
    git-mini-clone "$SCRIPT_REPO" "$SCRIPT_COMMIT" fc
    cd fc

    mkdir build && cd build

    local myconf=(
        --prefix="$FFBUILD_PREFIX"
        --buildtype=release
        --wrap-mode=nofallback
        --default-library=static
        -Ddoc=disabled
        -Diconv=enabled
        -Dxml-backend=libxml2
        -Dtools=disabled
        -Dcache-build=disabled
        -Dtests=disabled
    )

    if [[ $TARGET == linux* ]]; then
        myconf+=(
            --sysconfdir=/etc
            --localstatedir=/var
            --cross-file=/cross.meson
        )
    elif [[ $TARGET == win* ]]; then
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

    meson setup "${myconf[@]}" ..
    ninja -j"$(nproc)"
    ninja install

    # Manually tell it to link against macOS builtin and static libintl
    if [[ $TARGET == mac* ]]; then
        sed -i '' '/^Libs:/ s/$/ -lintl -framework CoreFoundation/' "$FFBUILD_PREFIX"/lib/pkgconfig/fontconfig.pc
    fi
}

ffbuild_configure() {
    echo --enable-fontconfig
}

ffbuild_unconfigure() {
    echo --disable-fontconfig
}
