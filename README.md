# Building openvdb
cd ../openvdb/build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static -DUSE_BLOSC=OFF -DUSE_ZLIB=OFF -DOPENVDB_CORE_SHARED=OFF -A x64
cmake --build . --config Release --target install
Rename libopenvdb.lib in install dir to openvdb.lib


# Building shadowbox
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static
cmake --build . --config Release
