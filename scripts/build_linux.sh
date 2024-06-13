rm -rf build
mkdir -p build
cd build

# Prefer to use ninja if found
if which ninja >/dev/null; then
    ninja_cmake_args="-G Ninja"
    export NINJA_STATUS="[%f/%t %r %es] "
fi

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCUTLASS_ENABLE=ON \
      -DCUTLASS_ENABLE_HEADERS_ONLY=ON \
      ${ninja_cmake_args} \
      ..
cmake --build . -j 8
