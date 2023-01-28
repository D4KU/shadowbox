#!/bin/sh

cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
rsync -a --mkpath shadowbox "/mnt/c/Users/David/AppData/Roaming/Blender Foundation/Blender/3.4/scripts/addons"
"/mnt/c/Program Files/Blender Foundation/Blender 3.4/blender.exe" "C:\Users\David\Desktop\xyzDebug.blend" -P "C:\Users\David\Desktop\shadowbox.py"
