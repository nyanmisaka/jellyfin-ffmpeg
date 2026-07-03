#!/bin/bash

# Builds the EXE/ZIP inside the Docker container

set -o errexit
set -o xtrace

# mingw-std-threads
mingw_threads_commit="c931bac289dd431f1dd30fc4a5d1a7be36668073"
git clone https://github.com/meganz/mingw-std-threads.git
pushd mingw-std-threads
git checkout ${mingw_threads_commit}
mkdir -p ${FF_DEPS_PREFIX}/include
mv *.h ${FF_DEPS_PREFIX}/include
popd

# OpenCL headers
git clone -b v2023.04.17 --depth=1 https://github.com/KhronosGroup/OpenCL-Headers.git
pushd OpenCL-Headers/CL
mkdir -p ${FF_DEPS_PREFIX}/include/CL
mv * ${FF_DEPS_PREFIX}/include/CL
popd

# OpenCL ICD loader
git clone -b v2023.04.17 --depth=1 https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
pushd OpenCL-ICD-Loader
mkdir build
pushd build
cmake \
    -DCMAKE_TOOLCHAIN_FILE=${FF_CMAKE_TOOLCHAIN} \
    -DCMAKE_INSTALL_PREFIX=${FF_DEPS_PREFIX} \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENCL_ICD_LOADER_HEADERS_DIR=${FF_DEPS_PREFIX}/include \
    -DOPENCL_ICD_LOADER_{PIC,DISABLE_OPENCLON12}=ON \
    -DOPENCL_ICD_LOADER_{BUILD_SHARED_LIBS,BUILD_TESTING,REQUIRE_WDK}=OFF \
    ..
make -j$(nproc)
make install
popd
mkdir -p ${FF_DEPS_PREFIX}/lib/pkgconfig
cat > ${FF_DEPS_PREFIX}/lib/pkgconfig/OpenCL.pc << EOF
prefix=${FF_DEPS_PREFIX}
exec_prefix=\${prefix}
libdir=\${prefix}/lib
includedir=\${prefix}/include
Name: OpenCL
Description: OpenCL ICD Loader
Version: 3.0
Libs: -L\${libdir} -l:OpenCL.a
Cflags: -I\${includedir}
Libs.private: -lole32 -lshlwapi -lcfgmgr32
EOF
popd

# FFNVCODEC
git clone -b n12.0.16.1 --depth=1 https://github.com/FFmpeg/nv-codec-headers.git
pushd nv-codec-headers
make PREFIX=${FF_DEPS_PREFIX} install
popd

# AMF
mkdir amf-headers
pushd amf-headers
amf_ver="1.5.2"
amf_link="https://github.com/GPUOpen-LibrariesAndSDKs/AMF/releases/download/v${amf_ver}/AMF-headers-v${amf_ver}.tar.gz"
wget ${amf_link} -O amf.tar.gz
tar xaf amf.tar.gz
pushd amf-headers-v${amf_ver}/AMF
mkdir -p ${FF_DEPS_PREFIX}/include/AMF
mv * ${FF_DEPS_PREFIX}/include/AMF
popd
popd

# VPL
git clone -b v2.17.0 --depth=1 https://github.com/intel/libvpl.git
pushd libvpl
mkdir build && pushd build
cmake \
    -DCMAKE_TOOLCHAIN_FILE=${FF_CMAKE_TOOLCHAIN} \
    -DCMAKE_INSTALL_PREFIX=${FF_DEPS_PREFIX} \
    -DCMAKE_INSTALL_BINDIR=${FF_DEPS_PREFIX}/bin \
    -DCMAKE_INSTALL_LIBDIR=${FF_DEPS_PREFIX}/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DINSTALL_{DEV,LIB}=ON \
    -DINSTALL_EXAMPLES=OFF \
    -DBUILD_{TESTS,EXAMPLES,EXPERIMENTAL}=OFF \
    ..
make -j$(nproc)
make install
echo "Libs.private: -lstdc++" >> ${FF_DEPS_PREFIX}/lib/pkgconfig/vpl.pc
popd
popd

# Jellyfin-FFmpeg
pushd ${SOURCE_DIR}
ffversion="$(dpkg-parsechangelog --show-field Version)"
if [[ -f "patches/series" ]]; then
    quilt push -a
fi
./configure \
    --prefix=${FF_PREFIX} \
    ${FF_TARGET_FLAGS} \
    --extra-version=Jellyfin \
    --disable-unstable \
    --disable-ffplay \
    --disable-debug \
    --disable-doc \
    --disable-sdl2 \
    --disable-w32threads \
    --enable-pthreads \
    --enable-shared \
    --enable-gpl \
    --enable-version3 \
    --enable-opencl \
    --enable-dxva2 \
    --enable-d3d11va \
    --enable-amf \
    --enable-libvpl \
    --enable-ffnvcodec \
    --enable-cuda \
    --enable-cuda-llvm \
    --enable-cuvid \
    --enable-nvdec \
    --enable-nvenc
make -j$(nproc)
make install
popd

# Zip and copy artifacts
mkdir -p ${ARTIFACT_DIR}/zip
pushd ${FF_PREFIX}/bin
ffpackage="jellyfin-ffmpeg_${ffversion}-portable_win64"
zip -9 -r ${ARTIFACT_DIR}/zip/${ffpackage}.zip ./*.{exe,dll}
pushd ${ARTIFACT_DIR}/zip
chown -Rc $(stat -c %u:%g ${ARTIFACT_DIR}) ${ARTIFACT_DIR}
popd
popd
