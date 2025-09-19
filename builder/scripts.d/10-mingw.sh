#!/bin/bash

SCRIPT_REPO="https://github.com/mingw-w64/mingw-w64.git"
SCRIPT_COMMIT="2e31630bce69ba975ad3d7ef551debada09434a4"

ffbuild_enabled() {
    [[ $TARGET == win* ]] || return -1
    return 0
}

ffbuild_dockerlayer() {
    to_df "COPY --link --from=${SELFLAYER} /opt/mingw/. /"
    to_df "COPY --link --from=${SELFLAYER} /opt/mingw/. /opt/mingw"
}

ffbuild_dockerfinal() {
    to_df "COPY --link --from=${PREVLAYER} /opt/mingw/. /"
}

ffbuild_dockerbuild() {
    retry-tool sh -c "rm -rf mingw && git clone '$SCRIPT_REPO' mingw"
    cd mingw
    git checkout "$SCRIPT_COMMIT"

    GCC_SYSROOT="$(${FFBUILD_CROSS_PREFIX}gcc -print-sysroot)"

    unset CC CXX LD AR CPP LIBS CCAS
    unset CFLAGS CXXFLAGS LDFLAGS CPPFLAGS CCASFLAGS
    unset PKG_CONFIG_LIBDIR

    ###
    ### mingw-w64-headers
    ###
    (
        cd mingw-w64-headers

        local myconf=(
            --prefix="$GCC_SYSROOT/usr/$FFBUILD_TOOLCHAIN"
            --host="$FFBUILD_TOOLCHAIN"
            --with-default-win32-winnt="0x601"
            --with-default-msvcrt="ucrt"
            --enable-idl
            --enable-sdk=all
            --enable-secure-api
        )

        ./configure "${myconf[@]}"
        make -j$(nproc)
        make install DESTDIR="/opt/mingw"
    )

    cp -a /opt/mingw/. /

    ###
    ### mingw-w64-crt
    ###
    (
        cd mingw-w64-crt

        local myconf=(
            --prefix="$GCC_SYSROOT/usr/$FFBUILD_TOOLCHAIN"
            --host="$FFBUILD_TOOLCHAIN"
            --with-default-msvcrt="ucrt"
            --enable-wildcard
        )

        ./configure "${myconf[@]}"
        make -j$(nproc)
        make install DESTDIR="/opt/mingw"
    )

    cp -a /opt/mingw/. /

    ###
    ### mingw-w64-libraries/winpthreads
    ###
    (
        cd mingw-w64-libraries/winpthreads

        local myconf=(
            --prefix="$GCC_SYSROOT/usr/$FFBUILD_TOOLCHAIN"
            --host="$FFBUILD_TOOLCHAIN"
            --with-pic
            --disable-shared
            --enable-static
        )

        ./configure "${myconf[@]}"
        make -j$(nproc)
        make install DESTDIR="/opt/mingw"
    )
}

ffbuild_configure() {
    echo --disable-w32threads --enable-pthreads
}
