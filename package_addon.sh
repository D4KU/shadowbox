#!/bin/sh

dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$dir"

cd build
make -j8
cd ..

"/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/Msbuild/Current/Bin/msbuild.exe" "C:\shadowbox\build\shadowbox.sln" -target:core -property:Configuration=Release -verbosity:minimal

mv -f build/*.so shadowbox
mv -f build/Release/*.pyd shadowbox
rsync -a --mkpath shadowbox "/mnt/c/Users/David/AppData/Roaming/Blender Foundation/Blender/3.3/scripts/addons"

"/mnt/c/Program Files/Blender Foundation/Blender 3.3/blender.exe" "C:\Users\David\Desktop\xyzDebug.blend" -P "C:\Users\David\Desktop\shadowbox.py"
