#!/bin/bash

SCRIPT_REPO="https://code.videolan.org/videolan/libbluray.git"
SCRIPT_COMMIT="d41111c1eb05a278f8dc37fb220d6ef9a13e3291"

ffbuild_enabled() {
    return 0
}

ffbuild_dockerbuild() {
    git-mini-clone "$SCRIPT_REPO" "$SCRIPT_COMMIT" libbluray
    cd libbluray

    if [[ $TARGET == mac* ]]; then
        gsed -i 's/dec_init/libbluray_dec_init/g' src/libbluray/disc/*.c src/libbluray/disc/*.h

        # stop the static library from exporting symbols when linked into a shared lib
        gsed -i 's/-DBLURAY_API_EXPORT/-DBLURAY_API_EXPORT_DISABLED/g' src/meson.build
    else
        sed -i 's/dec_init/libbluray_dec_init/g' src/libbluray/disc/*.c src/libbluray/disc/*.h

        # stop the static library from exporting symbols when linked into a shared lib
        sed -i 's/-DBLURAY_API_EXPORT/-DBLURAY_API_EXPORT_DISABLED/g' src/meson.build
    fi

    mkdir build && cd build

    local myconf=(
        --prefix="$FFBUILD_PREFIX"
        --buildtype=release
        -Ddefault_library=static
        -Denable_docs=false
        -Denable_tools=false
        -Denable_devtools=false
        -Denable_examples=false
        -Dbdj_jar=disabled
        -Dfontconfig=enabled
        -Dfreetype=enabled
        -Dlibxml2=enabled
    )

    if [[ $TARGET == win* || $TARGET == linux* ]]; then
        myconf+=(
            --cross-file=/cross.meson
        )
    elif [[ $TARGET == mac* ]]; then
        :
    else
        echo "Unknown target"
        return -1
    fi

    meson setup "${myconf[@]}" ..
    ninja -j$(nproc)
    ninja install
}

ffbuild_configure() {
    echo --enable-libbluray
}

ffbuild_unconfigure() {
    echo --disable-libbluray
}
