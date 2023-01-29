# Blender Shadowbox

This is a [Blender](https://github.com/blender/blender) add-on to generate a
mesh from 3 images describing its silhouette from each axis.

# How to use

Load the images you want to generate a mesh from into your project. Clear your
selection and add a new Shadowbox from `Add > Shadowbox`. In the settings
panel in the lower left corner, assign each axis an image. The images will we
stretched to fit a unit cube, but you can scale the box to the wanted aspect
ratio.

Every time you execute the Shadowbox function on a mesh object, that object's
geometry will be overwritten. If you don't want to loose any of your meshes,
execute the function without a selection and a new object will be created.

Because high mesh resolutions will quickly drain performance, it is
recommended to only work with the resolution you really need. When you decided
on one you can slightly improve performance by manually scaling your
images to the wished resolution. Then the add-on doesn't have to create a
temporary copy of your images and resize it every time you regenerate the
mesh.

This is especially useful when you use the *Run in background* option, which
regenerates the mesh every frame while you can paint into the images from
inside Blender's Image Editor. Stop the automatic update by pressing *Escape*.

# Installation

Download the archive from the `Releases` page. Inside Blender, navigate to
`Edit > Preferences > Add-ons` and click the `Install` button. Choose the
downloaded archive and confirm.

# Building from source

For optimal performance this add-on uses a custom Python module written in
C++ which has to be compiled for the Python version the targeted Blender
version ships with. It also depends on the following libraries:

* [openVDB](https://github.com/AcademySoftwareFoundation/openvdb)
* [pybind11](https://github.com/pybind/pybind11)
* [eigen3](https://eigen.tuxfamily.org)

[CMake](https://github.com/Kitware/CMake) is used to unify the build process
regardless of your operating system, but the way you install dependencies
varies.

## Windows

It is recommended to install dependencies with
[vcpkg](https://github.com/microsoft/vcpkg). Assuming it has been installed to
`C:/vcpkg`, execute the following commands where you want to build Shadowbox:

```
C:/vcpkg/vcpkg install openvdb pybind11 eigen3 --triplet=x64-windows-static
git clone https://github.com/D4KU/shadowbox.git
cd shadowbox
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static
cmake --build . --config Release
```

The end product is a `.pyd` file which you have to copy next to the Python
scripts into the `shadowbox` directory at the root of this repository: `copy
Release/*.pyd ../shadowbox`

Next, copy that directory to Blender's add-on directory. For Blender version
3.4 this would be: `%APPDATA%/Blender Foundation/Blender/3.4/scripts/addons`.
Lastly you follow the steps outlined under [Installation](#installation), but
select the `__init__.py` file instead of the archive.

### Building openvdb
Because the static OpenVDB build from vcpkg seems to be [broken](https://github.com/microsoft/vcpkg/issues/26993), I had to compile OpenVDB from source as well. This may be resolved in the future. In case that happens to you, here are my build instructions. The last command for the installation you have to execute on a command line with admin rights.
```
vcpkg install boost tbb --triplet=x64-windows-static
git clone https://github.com/AcademySoftwareFoundation/openvdb
cd openvdb
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static -DUSE_BLOSC=OFF -DUSE_ZLIB=OFF -DOPENVDB_CORE_SHARED=OFF -A x64
cmake --build . --config Release --target install
```

Lastly, you may have to rename `libopenvdb.lib` in the installation directory to `openvdb.lib`.
