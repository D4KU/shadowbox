#!/bin/sh

dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$dir"
cd build
make -j8
cd ..
mv -f build/*.so shadowbox
mv -f build/Release/*.pyd shadowbox
rsync -av shadowbox "/mnt/c/Users/David/AppData/Roaming/Blender Foundation/Blender/3.3/scripts/addons"
