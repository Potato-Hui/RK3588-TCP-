#!/bin/bash
set -e

TOOL_CHAIN=/opt/atk-dlrk3588-toolchain
GCC_COMPILER=aarch64-buildroot-linux-gnu

export PATH=${TOOL_CHAIN}/bin:$PATH
export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++
export AR=${GCC_COMPILER}-ar
export RANLIB=${GCC_COMPILER}-ranlib

# 清空污染环境变量
unset PKG_CONFIG_PATH
unset PKG_CONFIG_LIBDIR
unset PKG_CONFIG_SYSROOT_DIR

ROOT_PWD=$( cd "$( dirname $0 )" && pwd )
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64

rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ../.. \
-DCMAKE_SYSTEM_NAME=Linux \
-DCMAKE_C_COMPILER=${CC} \
-DCMAKE_CXX_COMPILER=${CXX}

make -j$(nproc)
make install
cd -

echo "编译脚本执行完毕"
