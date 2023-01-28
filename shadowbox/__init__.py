bl_info = {
    "name": "Shadowbox",
    "description": (
        "Generate a mesh from 3 images describing its silhouette from "
        "each axis"
    ),
    "blender": (3, 3, 0),
    "category": "Mesh",
}

import os
import sys

mpath = os.path.dirname(os.path.realpath(__file__))
if not any(mpath in v for v in sys.path):
    sys.path.insert(0, mpath)

if 'bpy' in locals():
    import importlib
    importlib.reload(operator)
else:
    import bpy
    from . import operator


classes = (
    operator.Shadowbox,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
